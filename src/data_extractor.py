"""
data_extractor.py
─────────────────
Uses Claude to extract structured numerical / categorical data from document
text so it can be turned into Plotly charts.

Claude is asked to return a strict JSON schema describing the data.  The
structured output is then passed directly to visualizer.py.

Extraction types supported:
  - "timeseries"  : time-based metrics (e.g. revenue by year)
  - "comparison"  : compare named entities on a metric (e.g. product sales)
  - "distribution": part-of-whole proportions (e.g. budget breakdown)
  - "ranking"     : ranked list of items by numeric value
  - "multibar"    : multiple metrics per category
"""

import json
import logging
import re
from typing import Optional

from anthropic import Anthropic

logger = logging.getLogger(__name__)

# JSON schema Claude must follow when returning extracted data
DATA_SCHEMA = """
{
  "chart_type": "<timeseries | comparison | distribution | ranking | multibar>",
  "title": "<descriptive chart title>",
  "x_label": "<x-axis label>",
  "y_label": "<y-axis label>",
  "unit": "<e.g. USD millions, %, count — or empty string if none>",
  "series": [
    {
      "name": "<series / category name>",
      "values": [
        {"label": "<x-axis label or category>", "value": <numeric value>}
      ]
    }
  ],
  "insight": "<1-2 sentence plain-language insight about this data>"
}
"""

EXTRACT_SYSTEM = """You are a data extraction specialist.
Your job is to find numerical data in the provided document text and return it
as a single JSON object matching EXACTLY this schema (no extra keys, no markdown):

""" + DATA_SCHEMA + """

Rules:
- Extract real numbers from the text; do NOT invent data.
- If you cannot find data for the requested topic, return {"error": "<reason>"}.
- Values must be plain numbers (not strings). Strip currency symbols, commas, % signs.
- Keep series names short (≤ 30 chars).
- Include at least 3 data points if possible."""


class DataExtractor:
    """
    Prompts Claude to locate and structure quantitative information from text.

    Parameters
    ----------
    api_key : str
        Anthropic API key.
    model : str
        Claude model to use (haiku is fast enough for structured extraction).
    """

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    # ── Public API ───────────────────────────────────────────────────────────

    def extract(
        self,
        document_text: str,
        topic: Optional[str] = None,
        max_text_chars: int = 8000,
    ) -> dict:
        """
        Extract structured chart data from `document_text`.

        Parameters
        ----------
        document_text : str
            Full or partial text of the document.
        topic : str, optional
            Guide the extraction (e.g. "revenue", "market share", "expenses").
            If None, Claude selects the most prominent numerical dataset.
        max_text_chars : int
            Truncate text to this length to stay within token limits.

        Returns
        -------
        dict
            Parsed JSON matching DATA_SCHEMA, or {"error": "..."} on failure.
        """
        trimmed = document_text[:max_text_chars]

        topic_hint = (
            f"Focus specifically on data related to: {topic}."
            if topic
            else "Find the most prominent numerical dataset in this document."
        )

        user_msg = (
            f"{topic_hint}\n\n"
            f"Document text:\n\n{trimmed}"
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1200,
                system=EXTRACT_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()

            # Strip accidental markdown code fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)
            return data

        except json.JSONDecodeError as e:
            logger.warning("JSON parse error in DataExtractor: %s", e)
            return {"error": f"Could not parse Claude's response as JSON: {e}"}
        except Exception as e:
            logger.error("DataExtractor.extract failed: %s", e)
            return {"error": str(e)}

    def extract_key_stats(self, document_text: str, max_text_chars: int = 6000) -> dict:
        """
        Extract a set of key statistics (KPIs) from the document as plain
        key-value pairs.  Used for the Document Overview tab.

        Returns
        -------
        dict  {"stats": [{"metric": "...", "value": "...", "context": "..."}]}
        """
        trimmed = document_text[:max_text_chars]

        system = """Extract up to 10 key statistics or KPIs from the document.
Return ONLY valid JSON (no markdown):
{
  "stats": [
    {"metric": "<metric name>", "value": "<formatted value with unit>", "context": "<1 sentence context>"}
  ]
}
If no statistics are found, return {"stats": []}."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                system=system,
                messages=[{"role": "user", "content": f"Document:\n{trimmed}"}],
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            logger.error("extract_key_stats failed: %s", e)
            return {"stats": []}

    def suggest_visualizations(self, document_text: str, max_text_chars: int = 4000) -> list:
        """
        Ask Claude to suggest 3-5 chart topics for this document.

        Returns
        -------
        list of str  e.g. ["Revenue by year", "Expense breakdown", ...]
        """
        trimmed = document_text[:max_text_chars]
        system = """Given the document excerpt, suggest 3-5 specific chart topics
that would be visually interesting.  Return ONLY a JSON array of strings, e.g.:
["Revenue by quarter", "Market share by segment", "Employee headcount trend"]
Focus on topics where numeric data actually exists in the text."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system=system,
                messages=[{"role": "user", "content": trimmed}],
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception:
            return ["Revenue trend", "Category breakdown", "Year-over-year comparison"]
