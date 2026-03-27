"""
visualizer.py
─────────────
Converts the structured JSON output from DataExtractor into Plotly figures.

Each chart type maps to an appropriate Plotly chart:
  timeseries   → px.line  (with markers)
  comparison   → px.bar   (grouped if multiple series)
  distribution → px.pie   (donut variant for clarity)
  ranking      → horizontal px.bar
  multibar     → px.bar   (grouped)

All charts share a consistent dark-themed style that fits Streamlit's default theme.
"""

import logging
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ── Shared style constants ────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_white"
COLOR_SEQUENCE  = px.colors.qualitative.Set2   # Colorblind-friendly palette
FONT_FAMILY     = "Inter, sans-serif"


class Visualizer:
    """
    Builds interactive Plotly figures from DataExtractor JSON payloads.

    Usage
    -----
    viz = Visualizer()
    fig = viz.build(extracted_data)
    st.plotly_chart(fig)
    """

    def build(self, data: dict) -> Optional[go.Figure]:
        """
        Dispatch to the correct chart builder based on `data["chart_type"]`.

        Parameters
        ----------
        data : dict
            Output from DataExtractor.extract(). Must contain "chart_type"
            and "series".  Returns None if data contains an "error" key or
            cannot be rendered.

        Returns
        -------
        go.Figure or None
        """
        if "error" in data:
            logger.warning("Visualizer received error payload: %s", data["error"])
            return None

        chart_type = data.get("chart_type", "comparison")
        try:
            if chart_type == "timeseries":
                return self._line_chart(data)
            elif chart_type == "distribution":
                return self._pie_chart(data)
            elif chart_type == "ranking":
                return self._horizontal_bar(data)
            elif chart_type in ("comparison", "multibar"):
                return self._bar_chart(data)
            else:
                # Fallback: treat unknown types as bar chart
                logger.info("Unknown chart_type '%s', defaulting to bar.", chart_type)
                return self._bar_chart(data)
        except Exception as e:
            logger.error("Visualizer.build failed for type '%s': %s", chart_type, e)
            return None

    # ── Chart builders ────────────────────────────────────────────────────────

    def _line_chart(self, data: dict) -> go.Figure:
        """
        Line chart with markers for time-series data.
        Supports multiple series (one line per series).
        """
        df = self._to_dataframe(data)
        fig = px.line(
            df,
            x="label",
            y="value",
            color="series" if "series" in df.columns and df["series"].nunique() > 1 else None,
            markers=True,
            title=data.get("title", "Trend Over Time"),
            labels={"label": data.get("x_label", ""), "value": data.get("y_label", "")},
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=COLOR_SEQUENCE,
        )
        unit = data.get("unit", "")
        if unit:
            fig.update_yaxes(ticksuffix=f" {unit}" if not unit.startswith("%") else "")
        self._apply_style(fig)
        return fig

    def _bar_chart(self, data: dict) -> go.Figure:
        """
        Grouped bar chart for comparison or multi-series data.
        """
        df = self._to_dataframe(data)
        has_multi_series = "series" in df.columns and df["series"].nunique() > 1

        fig = px.bar(
            df,
            x="label",
            y="value",
            color="series" if has_multi_series else None,
            barmode="group" if has_multi_series else "relative",
            title=data.get("title", "Comparison"),
            labels={"label": data.get("x_label", ""), "value": data.get("y_label", "")},
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=COLOR_SEQUENCE,
            text_auto=".2s",
        )
        self._apply_style(fig)
        return fig

    def _pie_chart(self, data: dict) -> go.Figure:
        """
        Donut pie chart for distributions / part-of-whole breakdowns.
        Uses the first series only (additional series are ignored with a warning).
        """
        series = data.get("series", [])
        if not series:
            return self._empty_figure("No data for distribution chart.")

        if len(series) > 1:
            logger.warning("Distribution chart uses only the first series.")

        points = series[0].get("values", [])
        labels = [p["label"] for p in points]
        values = [p["value"] for p in points]

        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.40,          # Donut style
                textinfo="label+percent",
                hovertemplate="%{label}: %{value}<extra></extra>",
                marker=dict(colors=COLOR_SEQUENCE),
            )
        )
        fig.update_layout(
            title=data.get("title", "Distribution"),
            template=PLOTLY_TEMPLATE,
            font_family=FONT_FAMILY,
            showlegend=True,
            legend=dict(orientation="v", x=1.0, y=0.5),
        )
        return fig

    def _horizontal_bar(self, data: dict) -> go.Figure:
        """
        Horizontal bar chart for ranked lists.
        Items are sorted from highest to lowest for visual clarity.
        """
        df = self._to_dataframe(data)
        # Sort descending so highest value is at the top
        df = df.sort_values("value", ascending=True)

        fig = px.bar(
            df,
            x="value",
            y="label",
            orientation="h",
            title=data.get("title", "Ranking"),
            labels={"label": data.get("y_label", ""), "value": data.get("x_label", "")},
            template=PLOTLY_TEMPLATE,
            color="value",
            color_continuous_scale="Teal",
            text_auto=".2s",
        )
        fig.update_coloraxes(showscale=False)
        self._apply_style(fig)
        return fig

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_dataframe(data: dict) -> pd.DataFrame:
        """
        Flatten the nested series/values structure into a tidy DataFrame:
          columns: [label, value, series]  (series only if >1 series)
        """
        series_list = data.get("series", [])
        rows = []
        for series in series_list:
            name = series.get("name", "")
            for point in series.get("values", []):
                raw_val = point.get("value", 0)
                try:
                    # Strip commas/currency symbols that Claude sometimes includes
                    numeric = float(str(raw_val).replace(",", "").replace("$", "").strip())
                except (ValueError, TypeError):
                    numeric = 0.0
                rows.append({
                    "label": str(point.get("label", "")),
                    "value": numeric,
                    "series": name,
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # Drop series column when all values are identical (single series)
        if df["series"].nunique() <= 1:
            df = df.drop(columns=["series"])
        return df

    @staticmethod
    def _apply_style(fig: go.Figure) -> None:
        """Apply consistent axis / legend / font styling to all figures."""
        fig.update_layout(
            font_family=FONT_FAMILY,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#e2e8f0")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#e2e8f0")

    @staticmethod
    def _empty_figure(message: str) -> go.Figure:
        """Return a blank figure with an annotation message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    # ── Summary table ─────────────────────────────────────────────────────────

    @staticmethod
    def data_to_table(data: dict) -> Optional[pd.DataFrame]:
        """
        Convert extracted data to a pandas DataFrame for display as a table
        alongside the chart.
        """
        series_list = data.get("series", [])
        if not series_list:
            return None

        if len(series_list) == 1:
            # Simple two-column table
            points = series_list[0].get("values", [])
            label_col = data.get("x_label", "Category") or "Category"
            value_col = data.get("y_label", "Value") or "Value"
            unit      = data.get("unit", "")
            rows = [
                {label_col: p["label"], f"{value_col} ({unit})" if unit else value_col: p["value"]}
                for p in points
            ]
            return pd.DataFrame(rows)
        else:
            # Pivot: rows = labels, columns = series names
            all_labels = list(
                dict.fromkeys(
                    p["label"]
                    for s in series_list
                    for p in s.get("values", [])
                )
            )
            records = {label: {} for label in all_labels}
            for s in series_list:
                for p in s.get("values", []):
                    records[p["label"]][s["name"]] = p["value"]
            df = pd.DataFrame.from_dict(records, orient="index")
            df.index.name = data.get("x_label", "Category")
            return df.reset_index()
