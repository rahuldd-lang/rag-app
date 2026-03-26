# RAG Document Analyzer

A Streamlit application for analyzing PDF documents using a **two-stage hybrid RAG pipeline** powered by Anthropic Claude.

## Features

| Feature | Details |
|---------|---------|
| **Hybrid Retrieval** | BM25 sparse + dense embedding search fused with Reciprocal Rank Fusion (RRF) |
| **LLM** | Anthropic Claude (claude-3-5-haiku) for generation and LLM-as-judge evaluation |
| **Embeddings** | `all-MiniLM-L6-v2` (local, no API key needed) |
| **Vector Store** | ChromaDB with persistent local storage |
| **Visualizations** | Auto-generated Plotly charts (bar, line, pie, radar) from AI-extracted data |
| **Evaluation** | Answer Relevancy, Faithfulness, Context Precision, Token Overlap F1 |

## Architecture

```
PDF Upload → DocumentProcessor → ChromaDB (dense embeddings)
                                      ↓
User Query → RAGPipeline ← BM25 (sparse)
                    ↓ RRF fusion
              Claude (Haiku) → Answer
                    ↓
          DataExtractor → Visualizer → Plotly Charts
                    ↓
            Evaluator → Metrics Dashboard
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
cp .env.example .env
# Edit .env and add your key

# 3. Generate the sample PDF (optional)
python data/generate_sample_pdf.py

# 4. Run the app
streamlit run app.py
```

## Evaluation Metrics

Three complementary metrics assess pipeline quality:

1. **Answer Relevancy** (quantitative): Cosine similarity between question and answer embeddings. Measures whether the answer stays on-topic.

2. **Faithfulness** (LLM-as-judge): Claude rates whether every factual claim in the answer is supported by the retrieved context (scale 1–5, normalised to 0–1). Detects hallucination.

3. **Context Precision** (LLM-as-judge): Fraction of retrieved chunks that contain information relevant to the question. Measures retrieval precision.

A 10-question evaluation dataset (`data/eval_dataset.json`) is included for the TechCorp Annual Report sample PDF.

## Dataset

The app includes a synthetic **TechCorp Annual Report 2023** PDF with:
- Revenue by division (2021–2023)
- Operating expense breakdown
- Geographic revenue distribution
- Quarterly performance data
- Customer & employee metrics

To use real PDFs, upload them via the sidebar or download from [Kaggle PDF datasets](https://www.kaggle.com/datasets?search=pdf).

**Reference notebook**: [Two-stage retrieval RAG using rerank models](https://www.kaggle.com/code/warcoder/two-stage-retrieval-rag-using-rerank-models)

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set `ANTHROPIC_API_KEY` in the app secrets
4. Deploy

## File Structure

```
rag-app/
├── app.py                          # Main Streamlit application (4 tabs)
├── src/
│   ├── document_processor.py       # PDF ingestion, chunking, ChromaDB indexing
│   ├── rag_pipeline.py             # Two-stage retrieval (BM25 + dense + RRF)
│   ├── data_extractor.py           # AI-powered structured data extraction
│   ├── visualizer.py               # Plotly chart generation
│   └── evaluator.py                # RAG evaluation metrics
├── data/
│   ├── generate_sample_pdf.py      # Generates TechCorp Annual Report
│   └── eval_dataset.json           # 10 Q&A pairs for evaluation
├── .streamlit/config.toml          # Dark theme configuration
└── requirements.txt
```
