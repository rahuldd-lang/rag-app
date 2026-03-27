# RAG Document Analyzer

A **Streamlit** application that analyzes PDF documents using a two-stage hybrid RAG pipeline powered by Anthropic Claude. Upload any PDF, ask questions, auto-generate charts from extracted data, and benchmark the pipeline with built-in evaluation metrics.

## Live Demo

> Deploy to [Streamlit Cloud](https://share.streamlit.io) — users enter their own **Application Access Key** via the sidebar.

---

## Features

| Feature | Details |
|---------|---------|
| **Hybrid Retrieval** | BM25 sparse + dense ONNX embedding search, fused with Reciprocal Rank Fusion (RRF) |
| **LLM** | Anthropic Claude (`claude-haiku-4-5-20251001`) for generation and LLM-as-judge evaluation |
| **Embeddings** | ChromaDB's built-in ONNX `all-MiniLM-L6-v2` — no torch or torchvision required |
| **Vector Store** | ChromaDB with persistent local storage |
| **Visualizations** | Auto-generated Plotly charts (bar, line, pie, radar) from AI-extracted structured data |
| **Evaluation** | Context Precision, Context Recall, Faithfulness, Answer Relevancy, Token Overlap F1 |
| **Authentication** | No pre-filled keys — each user enters an Application Access Key in the sidebar |

---

## GenAI Framework

| Component | Technology | Role |
|-----------|-----------|------|
| **Orchestration** | LangChain (`langchain-text-splitters`) | Text chunking, pipeline management |
| **Language Model** | Anthropic Claude Haiku 4.5 | Generation, data extraction, LLM-as-judge evaluation |
| **Embeddings** | ChromaDB ONNX (`all-MiniLM-L6-v2`) | Dense vector representation — no PyTorch dependency |
| **Vector Store** | ChromaDB (persistent) | Stores and queries document embeddings |
| **Sparse Retrieval** | `rank_bm25` | Keyword-based candidate retrieval |
| **Score Fusion** | Reciprocal Rank Fusion (RRF) | Merges BM25 + dense rankings without a cross-encoder |

---

## Architecture

```
PDF Upload → DocumentProcessor → ChromaDB (ONNX embeddings)
                                        ↓
User Query → RAGPipeline ←──── BM25 sparse retrieval
                    │  Stage 2: RRF fusion
                    ↓
            Claude (Haiku) → Answer
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
  [Q&A Chat]  [Visualizer]  [Evaluator]
               Plotly charts  Metrics dashboard
```

**Two-stage retrieval** (mirrors the [Kaggle RAG notebook](https://www.kaggle.com/code/warcoder/two-stage-retrieval-rag-using-rerank-models)):
- Stage 1a — Dense semantic search via ChromaDB ONNX embeddings
- Stage 1b — BM25 keyword search via `rank_bm25`
- Stage 2 — Reciprocal Rank Fusion (RRF) merges both ranked lists without needing a cross-encoder

---

## Quick Start

```bash
# 1. Clone and install dependencies
git clone https://github.com/YOUR_USERNAME/rag-app.git
cd rag-app
pip install -r requirements.txt

# 2. Generate the sample PDF (also auto-generates on first app launch)
python3 data/generate_sample_pdf.py

# 3. Run the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), enter your **Application Access Key** in the sidebar, and the TechCorp Annual Report sample will load automatically.

> No `.env` file needed. The access key is entered directly in the app UI and never stored on disk.

---

## App Tabs

| Tab | What it does |
|-----|-------------|
| **📋 Overview** | Hero banner, GenAI framework metadata, pipeline architecture, AI-extracted KPIs, suggested chart topics |
| **💬 Q&A Chat** | Ask questions; see the answer + source chunks with RRF scores |
| **📈 Visualizations** | Type a topic (e.g. "revenue by division") to generate a chart; or click "Build Full Dashboard" for 4 charts at once |
| **🧪 Evaluation** | Run the 10-question eval dataset; view per-question scores + 5-dimension radar chart |

---

## Evaluation Metrics

Five metrics assess pipeline quality, all scored `[0, 1]`:

| Metric | Type | How it works |
|--------|------|-------------|
| **Faithfulness** | LLM-as-judge | Claude rates whether every factual claim in the answer is grounded in the retrieved context (1–5 → normalised). Detects hallucination. |
| **Context Precision** | LLM-as-judge | Fraction of retrieved chunks flagged as relevant to the question. Measures retrieval signal-to-noise. |
| **Context Recall** | LLM-as-judge | Fraction of reference-answer statements supported by the retrieved context. Measures retrieval completeness — did we miss anything? |
| **Answer Relevancy** | Quantitative | Cosine similarity between question and answer embeddings. High score → answer stays on-topic. |
| **Token Overlap F1** | Quantitative baseline | Lexical F1 between expected and generated answer. Requires ground-truth, used as a simple baseline comparison. |

A 10-question evaluation dataset (`data/eval_dataset.json`) with expected answers targets the included TechCorp Annual Report sample.

---

## Sample Dataset

The app ships with a synthetic **TechCorp Annual Report 2023** PDF (`data/generate_sample_pdf.py`) containing:

- Revenue by division (Cloud, Enterprise, AI Platform) — 2021–2023
- Operating expense breakdown
- Geographic revenue distribution (NA, EMEA, APAC, LATAM)
- Quarterly performance (Q1–Q4 2023)
- Customer & employee metrics

To test with real documents, upload any PDF via the sidebar or download datasets from [Kaggle PDF datasets](https://www.kaggle.com/datasets?search=pdf).

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (must be **public** for the free tier)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set **Repository**, **Branch: main**, **Main file: app.py**
4. Click **Deploy** — no secrets required (users enter their Application Access Key via the sidebar)

> On first boot, ChromaDB downloads the ONNX model (~79 MB) to `~/.cache/chroma/`. Subsequent boots are instant.

---

## File Structure

```
rag-app/
├── app.py                          # Main Streamlit app — 4 tabs, session state, sidebar
├── src/
│   ├── document_processor.py       # PDF ingestion, chunking, ChromaDB ONNX indexing
│   ├── rag_pipeline.py             # Two-stage retrieval: BM25 + dense + RRF fusion
│   ├── data_extractor.py           # Claude-powered structured JSON data extraction
│   ├── visualizer.py               # Plotly chart builder (line, bar, pie, horizontal bar)
│   └── evaluator.py                # 5 evaluation metrics incl. Context Recall
├── data/
│   ├── generate_sample_pdf.py      # Generates TechCorp Annual Report 2023 PDF
│   └── eval_dataset.json           # 10 Q&A pairs with expected answers for benchmarking
├── .streamlit/
│   └── config.toml                 # Light theme configuration
└── requirements.txt
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | UI framework |
| `anthropic` | Claude API (generation + LLM-as-judge) |
| `chromadb` | Vector store with built-in ONNX embeddings |
| `onnxruntime` | Runs the ONNX embedding model — no torch/torchvision needed |
| `rank_bm25` | Sparse keyword retrieval (Stage 1b) |
| `langchain-text-splitters` | Recursive character text splitter |
| `pypdf` / `pdfplumber` | PDF text extraction |
| `plotly` | Interactive charts |
| `scikit-learn` | Cosine similarity for Answer Relevancy metric |
| `matplotlib` | Required by pandas Styler for evaluation table color gradients |
| `reportlab` | Generates the sample PDF |
