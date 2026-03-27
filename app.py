"""
app.py  ──  RAG Document Analyzer
══════════════════════════════════════════════════════════════════════════════
A Streamlit application that combines:
  • Two-stage hybrid RAG (BM25 + dense retrieval with RRF fusion)
  • Claude-powered Q&A grounded in uploaded documents
  • Automated data extraction and interactive Plotly visualizations
  • Quantitative evaluation metrics (Answer Relevancy, Faithfulness, Context Precision)

Architecture overview:
  ┌─────────────┐    ┌──────────────────┐    ┌────────────────┐
  │  PDF Upload │───▶│ DocumentProcessor │───▶│   ChromaDB     │
  └─────────────┘    └──────────────────┘    │  (embeddings)  │
                                              └────────┬───────┘
  ┌─────────────┐    ┌──────────────────┐             │ dense
  │  User Query │───▶│   RAGPipeline    │◀────────────┘
  └─────────────┘    │  (BM25 + RRF)    │
                      └────────┬─────────┘
                               │ top-k chunks
                      ┌────────▼─────────┐
                      │  Claude (Haiku)   │  ← Generation
                      └────────┬─────────┘
                               │
                  ┌────────────┼────────────────┐
                  ▼            ▼                 ▼
            [Q&A Answer]  [Visualizer]    [Evaluator]

Dataset:  data/TechCorp_Annual_Report_2023.pdf  (or any user-uploaded PDF)
Eval set: data/eval_dataset.json  (10 Q&A pairs for the sample report)
══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import logging
import warnings
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Suppress noisy library warnings ─────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="RAG Document Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a polished look (light theme) ─────────────────────────────
st.markdown("""
<style>
  /* Metric cards */
  div[data-testid="metric-container"] {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 12px 16px;
  }
  /* Source expander styling */
  .source-chunk {
      background: #f1f5f9;
      border-left: 3px solid #2563eb;
      padding: 8px 12px;
      border-radius: 0 4px 4px 0;
      margin-bottom: 8px;
      font-size: 13px;
      color: #1e293b;
  }
  /* Score badge */
  .score-badge {
      display: inline-block;
      background: #dbeafe;
      border: 1px solid #93c5fd;
      border-radius: 12px;
      padding: 2px 10px;
      font-size: 12px;
      color: #1d4ed8;
  }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  Session-state initialisation
#  All heavy objects (processor, pipeline, evaluator) are cached so they
#  survive re-runs without re-initialisation.
# ════════════════════════════════════════════════════════════════════════════

def init_session():
    """Initialise keys in st.session_state on first run."""
    defaults = {
        "api_key":       "",   # Always empty — user must provide their own key
        "doc_processor": None,
        "rag_pipeline":  None,
        "data_extractor":None,
        "evaluator":     None,
        "chat_history":  [],    # List of {"role": "user"|"assistant", "content": str}
        "active_doc_id": None,  # Currently selected document for focused queries
        "eval_results":  None,  # Cached evaluation results
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ════════════════════════════════════════════════════════════════════════════
#  Lazy component initialisation
#  Components are created once after the user provides an API key.
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading document processor…")
def get_doc_processor(persist_dir: str):
    """Cache the DocumentProcessor + ChromaDB client across reruns."""
    from src.document_processor import DocumentProcessor
    return DocumentProcessor(persist_dir=persist_dir)


@st.cache_resource(show_spinner="Connecting to Claude…")
def get_rag_pipeline(api_key: str, persist_dir: str):
    """Cache the RAGPipeline so it reuses the same ChromaDB collection."""
    from src.rag_pipeline import RAGPipeline
    proc = get_doc_processor(persist_dir)
    return RAGPipeline(doc_processor=proc, api_key=api_key)


@st.cache_resource(show_spinner="Loading data extractor…")
def get_data_extractor(api_key: str):
    from src.data_extractor import DataExtractor
    return DataExtractor(api_key=api_key)


@st.cache_resource(show_spinner="Loading evaluator…")
def get_evaluator(api_key: str):
    from src.evaluator import Evaluator
    return Evaluator(api_key=api_key)


# ════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render the left sidebar with API key input and PDF uploader."""
    with st.sidebar:
        st.title("📊 RAG Analyzer")
        st.caption("Two-stage hybrid retrieval · Claude · Plotly")

        # ── API Key ──────────────────────────────────────────────────────────
        st.subheader("⚙️ Configuration")
        api_key_input = st.text_input(
            "Anthropic API Key",
            value=st.session_state.api_key,
            type="password",
            help="Get your key at console.anthropic.com",
        )
        if api_key_input:
            st.session_state.api_key = api_key_input

        if not st.session_state.api_key:
            st.warning("Enter your API key to get started.")
            st.stop()

        # Persist directory for ChromaDB
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

        # ── PDF Upload ───────────────────────────────────────────────────────
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyse.",
        )

        # ── Sample data shortcut ─────────────────────────────────────────────
        sample_path = Path("data/TechCorp_Annual_Report_2023.pdf")
        if not sample_path.exists():
            if st.button("🔧 Generate Sample PDF", help="Creates a synthetic annual report for demo"):
                with st.spinner("Generating sample PDF…"):
                    _generate_sample_pdf()
                st.success("Sample PDF ready!")
                st.rerun()
        else:
            use_sample = st.checkbox("📋 Use sample: TechCorp Annual Report 2023", value=True)

        # ── Process uploads ──────────────────────────────────────────────────
        proc = get_doc_processor(persist_dir)
        rag  = get_rag_pipeline(st.session_state.api_key, persist_dir)

        # Process user-uploaded files
        if uploaded_files:
            for f in uploaded_files:
                _index_pdf(proc, f.read(), f.name)

        # Process sample PDF if selected
        if sample_path.exists() and use_sample:
            with open(sample_path, "rb") as sf:
                _index_pdf(proc, sf.read(), sample_path.name)

        # ── Indexed documents list ────────────────────────────────────────────
        st.subheader("📚 Indexed Documents")
        docs = proc.get_all_documents()
        if docs:
            doc_options = {d["filename"]: d["doc_id"] for d in docs}
            selected_name = st.selectbox(
                "Active document (for focused Q&A)",
                options=["All documents"] + list(doc_options.keys()),
            )
            st.session_state.active_doc_id = (
                doc_options.get(selected_name) if selected_name != "All documents" else None
            )

            stats = proc.collection_stats()
            col1, col2 = st.columns(2)
            col1.metric("Documents", stats["total_documents"])
            col2.metric("Chunks", stats["total_chunks"])

            # Per-document details
            with st.expander("Document details"):
                for d in docs:
                    st.markdown(
                        f"**{d['filename']}**  \n"
                        f"{d['chunk_count']} chunks · {d['page_count']} pages"
                    )
                    if st.button(f"🗑 Remove", key=f"del_{d['doc_id']}"):
                        proc.delete_document(d["doc_id"])
                        st.rerun()
        else:
            st.info("Upload a PDF or generate the sample to begin.")

        # ── About ─────────────────────────────────────────────────────────────
        with st.expander("ℹ️ About this app"):
            st.markdown("""
**RAG Document Analyzer**
Built with LangChain, ChromaDB, sentence-transformers, and Anthropic Claude.

**Retrieval**: Two-stage hybrid search:
1. BM25 (keyword) + dense embedding search
2. Reciprocal Rank Fusion (RRF) merge

**Evaluation metrics**:
- Answer Relevancy (cosine similarity)
- Faithfulness (LLM-as-judge)
- Context Precision (LLM-as-judge)

**Dataset**: [Kaggle PDF datasets](https://www.kaggle.com/datasets?search=pdf)
**Reference**: [Two-stage RAG notebook](https://www.kaggle.com/code/warcoder/two-stage-retrieval-rag-using-rerank-models)
            """)


@st.cache_data(show_spinner=False)
def _index_pdf_cached(pdf_bytes: bytes, filename: str, persist_dir: str) -> tuple:
    """Cache PDF indexing so the same file isn't re-processed on every rerun."""
    proc = get_doc_processor(persist_dir)
    return proc.process_pdf(pdf_bytes, filename)


def _index_pdf(proc, pdf_bytes: bytes, filename: str):
    """Index a PDF with a progress indicator."""
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    with st.spinner(f"Indexing {filename}…"):
        chunks, doc_id = _index_pdf_cached(pdf_bytes, filename, persist_dir)
    return chunks, doc_id


def _generate_sample_pdf():
    """Generate the sample PDF using generate_sample_pdf.py."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "data/generate_sample_pdf.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        st.error(f"Failed to generate sample PDF: {result.stderr}")


# ════════════════════════════════════════════════════════════════════════════
#  Tab 1: Document Overview
# ════════════════════════════════════════════════════════════════════════════

def tab_overview():
    """
    Shows document statistics, key extracted metrics (KPIs), and a word-level
    topic analysis chart.
    """
    st.header("📋 Document Overview")

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    proc = get_doc_processor(persist_dir)
    docs = proc.get_all_documents()

    if not docs:
        st.info("No documents indexed. Upload a PDF or generate the sample report in the sidebar.")
        return

    # ── Document cards ────────────────────────────────────────────────────────
    st.subheader("Indexed Documents")
    cols = st.columns(min(len(docs), 3))
    for i, doc in enumerate(docs):
        with cols[i % 3]:
            st.markdown(f"""
<div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:14px;margin-bottom:8px">
  <b>{doc['filename']}</b><br>
  <span style="color:#94a3b8;font-size:13px">
    {doc['chunk_count']} chunks &nbsp;·&nbsp; {doc['page_count']} pages
  </span>
</div>
""", unsafe_allow_html=True)

    # ── KPI extraction ────────────────────────────────────────────────────────
    st.subheader("📌 Key Statistics (AI-Extracted)")
    extractor = get_data_extractor(st.session_state.api_key)

    # Pick the active document or the first available
    target_doc = docs[0] if not st.session_state.active_doc_id else next(
        (d for d in docs if d["doc_id"] == st.session_state.active_doc_id), docs[0]
    )

    with st.spinner("Extracting key statistics…"):
        doc_text = proc.get_full_text(target_doc["doc_id"])
        kpi_result = extractor.extract_key_stats(doc_text)

    stats = kpi_result.get("stats", [])
    if stats:
        # Display as metric grid (3 per row)
        rows = [stats[i:i+3] for i in range(0, len(stats), 3)]
        for row in rows:
            cols = st.columns(3)
            for col, stat in zip(cols, row):
                col.metric(
                    label=stat.get("metric", ""),
                    value=stat.get("value", ""),
                    help=stat.get("context", ""),
                )
    else:
        st.info("No key statistics could be extracted from this document.")

    # ── Suggested visualisations ──────────────────────────────────────────────
    st.subheader("💡 Suggested Visualizations")
    with st.spinner("Analysing document for chart opportunities…"):
        suggestions = extractor.suggest_visualizations(doc_text[:3000])

    if suggestions:
        st.write("The document contains data suitable for these charts:")
        for s in suggestions:
            st.markdown(f"- {s}")
        st.caption("→ Use the **Visualizations** tab to generate these charts.")


# ════════════════════════════════════════════════════════════════════════════
#  Tab 2: Q&A Chat
# ════════════════════════════════════════════════════════════════════════════

def tab_chat():
    """
    Chat interface for querying documents via the two-stage RAG pipeline.
    Displays retrieved source chunks alongside each answer.
    """
    st.header("💬 Chat with Documents")

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    proc = get_doc_processor(persist_dir)
    if not proc.get_all_documents():
        st.info("Index at least one document to start chatting.")
        return

    rag = get_rag_pipeline(st.session_state.api_key, persist_dir)

    # Active document indicator
    if st.session_state.active_doc_id:
        docs = proc.get_all_documents()
        fname = next(
            (d["filename"] for d in docs if d["doc_id"] == st.session_state.active_doc_id),
            "selected document"
        )
        st.caption(f"🎯 Focused on: **{fname}** (change in sidebar)")
    else:
        st.caption("🔍 Searching across **all documents** (narrow in sidebar)")

    # ── Chat history display ──────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show sources for assistant messages
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander(
                    f"📎 {len(msg['sources'])} source chunks (RRF score shown)",
                    expanded=False,
                ):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f'<div class="source-chunk">'
                            f'<span class="score-badge">#{i} · score {src["score"]:.4f}</span>'
                            f'<br><b>{src["filename"]}</b> · page {src["page"]}<br><br>'
                            f'{src["text"][:400]}{"…" if len(src["text"]) > 400 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # ── Input box ─────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your documents…"):
        # Add user message to history and display immediately
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving (BM25 + dense) → fusing → generating…"):
                result = rag.query(
                    question=prompt,
                    doc_id=st.session_state.active_doc_id,
                )

            st.markdown(result["answer"])

            # Retrieval stats
            st.caption(
                f"Retrieved: {result['dense_count']} dense + "
                f"{result['sparse_count']} BM25 → fused to {len(result['sources'])} chunks"
            )

            # Source expander
            if result["sources"]:
                with st.expander(
                    f"📎 {len(result['sources'])} source chunks",
                    expanded=False,
                ):
                    for i, src in enumerate(result["sources"], 1):
                        st.markdown(
                            f'<div class="source-chunk">'
                            f'<span class="score-badge">#{i} · RRF {src["score"]:.4f}</span>'
                            f'<br><b>{src["filename"]}</b> · page {src["page"]}<br><br>'
                            f'{src["text"][:400]}{"…" if len(src["text"]) > 400 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        # Persist message + sources in session history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑 Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  Tab 3: Visualizations
# ════════════════════════════════════════════════════════════════════════════

def tab_visualizations():
    """
    Extract structured numerical data from documents and render interactive
    Plotly charts.  Users can specify a topic or let the AI choose.
    """
    st.header("📈 Data Visualizations")

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    proc = get_doc_processor(persist_dir)
    docs = proc.get_all_documents()

    if not docs:
        st.info("Index at least one document to create visualizations.")
        return

    extractor = get_data_extractor(st.session_state.api_key)
    from src.visualizer import Visualizer
    viz = Visualizer()

    # Pick active document
    target_doc = (
        next((d for d in docs if d["doc_id"] == st.session_state.active_doc_id), docs[0])
        if st.session_state.active_doc_id
        else docs[0]
    )

    st.caption(f"Source document: **{target_doc['filename']}**")

    # ── Topic selector ────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input(
            "Chart topic (leave blank for AI to choose)",
            placeholder="e.g. revenue by division, expense breakdown, quarterly trend…",
        )
    with col2:
        generate_btn = st.button("🎨 Generate Chart", type="primary", use_container_width=True)

    # Quick-access preset buttons
    st.write("**Quick presets:**")
    preset_cols = st.columns(5)
    presets = [
        "Revenue by division",
        "Expense breakdown",
        "Quarterly revenue",
        "Geographic revenue",
        "Customer metrics",
    ]
    for i, preset in enumerate(presets):
        if preset_cols[i].button(preset, key=f"preset_{i}", use_container_width=True):
            topic = preset
            generate_btn = True

    # ── Chart generation ──────────────────────────────────────────────────────
    if generate_btn:
        doc_text = proc.get_full_text(target_doc["doc_id"])

        with st.spinner(f"Extracting {'data for: ' + topic if topic else 'best dataset'}…"):
            extracted = extractor.extract(doc_text, topic=topic if topic else None)

        if "error" in extracted:
            st.error(f"Extraction failed: {extracted['error']}")
            return

        # ── Insight callout ───────────────────────────────────────────────────
        if "insight" in extracted:
            st.info(f"💡 **Insight:** {extracted['insight']}")

        # ── Main chart ────────────────────────────────────────────────────────
        fig = viz.build(extracted)
        if fig:
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Could not render a chart for this data.")

        # ── Data table ────────────────────────────────────────────────────────
        with st.expander("📊 View raw data table"):
            df = viz.data_to_table(extracted)
            if df is not None:
                st.dataframe(df, width="stretch")
            else:
                st.json(extracted)

        # ── Raw JSON payload ──────────────────────────────────────────────────
        with st.expander("🔍 Extracted JSON (from Claude)"):
            st.json(extracted)

    # ── Multi-chart dashboard ─────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Auto Dashboard")
    st.caption("Generate multiple charts in one click")

    if st.button("🚀 Build Full Dashboard", type="secondary"):
        doc_text = proc.get_full_text(target_doc["doc_id"])
        dashboard_topics = [
            "Revenue by division across years",
            "Operating expense breakdown as distribution",
            "Revenue by geographic region distribution",
            "Quarterly revenue trend",
        ]

        chart_cols = st.columns(2)
        for i, dtopic in enumerate(dashboard_topics):
            with chart_cols[i % 2]:
                with st.spinner(f"Generating: {dtopic}…"):
                    ext = extractor.extract(doc_text, topic=dtopic)
                if "error" not in ext:
                    fig = viz.build(ext)
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                        if "insight" in ext:
                            st.caption(f"💡 {ext['insight']}")


# ════════════════════════════════════════════════════════════════════════════
#  Tab 4: Evaluation
# ════════════════════════════════════════════════════════════════════════════

def tab_evaluation():
    """
    Run and display RAG evaluation metrics against the eval dataset.

    Metrics computed:
      • Answer Relevancy   – embedding cosine similarity (quantitative)
      • Faithfulness       – LLM-as-judge 1-5 score, normalised to [0,1]
      • Context Precision  – fraction of retrieved chunks that are relevant
      • Token Overlap F1   – lexical overlap with expected answer (baseline)
    """
    st.header("🧪 Evaluation Dashboard")

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    proc = get_doc_processor(persist_dir)
    docs = proc.get_all_documents()

    # ── Metric explanations ───────────────────────────────────────────────────
    with st.expander("📖 Metric definitions", expanded=False):
        st.markdown("""
| Metric | Type | What it measures |
|--------|------|-----------------|
| **Answer Relevancy** | Quantitative | Cosine similarity between question and answer embeddings. High score → answer stays on topic. |
| **Faithfulness** | LLM-as-judge (0–1) | Claude rates whether every claim in the answer is supported by the retrieved context. Detects hallucination. |
| **Context Precision** | LLM-as-judge (0–1) | Fraction of retrieved chunks that actually contain relevant information. High score → retrieval is precise. |
| **Token Overlap F1** | Quantitative baseline | Token-level F1 between expected and generated answers (requires ground truth). |
""")

    if not docs:
        st.info("Index a document to run evaluation. The eval dataset targets the TechCorp sample PDF.")
        return

    eval_path = Path("data/eval_dataset.json")
    if not eval_path.exists():
        st.error("Evaluation dataset not found: data/eval_dataset.json")
        return

    # Load eval dataset for preview
    with open(eval_path) as f:
        eval_data = json.load(f)

    with st.expander("📋 Evaluation dataset preview"):
        q_rows = [
            {"ID": q["id"], "Question": q["question"], "Difficulty": q.get("difficulty", "")}
            for q in eval_data["questions"]
        ]
        st.dataframe(pd.DataFrame(q_rows), width="stretch")
        st.caption(f"Source: {eval_data.get('source_document', 'unknown')} · {len(eval_data['questions'])} questions")

    # ── Run evaluation ─────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])
    with col1:
        n_questions = st.slider(
            "Number of questions to evaluate",
            min_value=1,
            max_value=len(eval_data["questions"]),
            value=min(5, len(eval_data["questions"])),
            help="Fewer questions = faster evaluation",
        )

    with col2:
        run_btn = st.button("▶ Run Evaluation", type="primary")
        if st.session_state.eval_results:
            st.caption("Previous results shown below. Click Run to refresh.")

    if run_btn:
        rag       = get_rag_pipeline(st.session_state.api_key, persist_dir)
        evaluator = get_evaluator(st.session_state.api_key)

        # Subset the eval dataset
        eval_subset = {
            **eval_data,
            "questions": eval_data["questions"][:n_questions],
        }

        import tempfile
        # Write subset to a temp file so evaluator can load it
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(eval_subset, tmp)
            tmp_path = tmp.name

        progress_bar = st.progress(0, text="Evaluating…")
        results = []

        for i, item in enumerate(eval_subset["questions"]):
            progress_bar.progress(
                (i + 1) / n_questions,
                text=f"Evaluating question {i+1}/{n_questions}: {item['question'][:60]}…",
            )
            # Single-item evaluation
            try:
                rag_result = rag.query(
                    item["question"],
                    doc_id=st.session_state.active_doc_id,
                )
                metrics = evaluator.evaluate_response(
                    question=item["question"],
                    answer=rag_result["answer"],
                    context_chunks=rag_result["sources"],
                )
                from src.evaluator import Evaluator as Ev
                overlap = Ev._token_overlap(
                    item.get("expected_answer", ""),
                    rag_result["answer"],
                )
                results.append({
                    "id":                 item["id"],
                    "question":           item["question"],
                    "difficulty":         item.get("difficulty", ""),
                    "expected_answer":    item.get("expected_answer", ""),
                    "generated_answer":   rag_result["answer"],
                    "answer_relevancy":   round(metrics["answer_relevancy"], 3),
                    "faithfulness":       round(metrics["faithfulness"], 3),
                    "context_precision":  round(metrics["context_precision"], 3),
                    "token_overlap_f1":   round(overlap, 3),
                })
            except Exception as e:
                results.append({"id": item["id"], "question": item["question"], "error": str(e)})

        progress_bar.empty()
        st.session_state.eval_results = results

    # ── Display results ────────────────────────────────────────────────────────
    if st.session_state.eval_results:
        results = st.session_state.eval_results
        valid   = [r for r in results if "error" not in r]

        if not valid:
            st.error("All evaluations failed. Check your API key and indexed documents.")
            return

        # Aggregate metrics
        avg_ar = sum(r["answer_relevancy"] for r in valid) / len(valid)
        avg_fa = sum(r["faithfulness"]      for r in valid) / len(valid)
        avg_cp = sum(r["context_precision"] for r in valid) / len(valid)
        avg_to = sum(r["token_overlap_f1"]  for r in valid) / len(valid)

        # ── Summary metric cards ──────────────────────────────────────────────
        st.subheader("📊 Aggregate Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Answer Relevancy",  f"{avg_ar:.3f}",
                  help="Embedding cosine similarity [0–1]. ≥0.7 is good.")
        c2.metric("Faithfulness",       f"{avg_fa:.3f}",
                  help="LLM-as-judge grounding score [0–1]. ≥0.75 is good.")
        c3.metric("Context Precision",  f"{avg_cp:.3f}",
                  help="Fraction of retrieved chunks that are relevant [0–1].")
        c4.metric("Token Overlap F1",   f"{avg_to:.3f}",
                  help="Lexical F1 vs. expected answer (0–1 baseline metric).")

        # ── Radar / spider chart ──────────────────────────────────────────────
        radar_fig = _build_radar_chart(avg_ar, avg_fa, avg_cp, avg_to)
        st.plotly_chart(radar_fig, width="stretch")

        # ── Per-question table ────────────────────────────────────────────────
        st.subheader("Per-Question Results")
        df = pd.DataFrame(valid)[
            ["id", "question", "difficulty",
             "answer_relevancy", "faithfulness", "context_precision", "token_overlap_f1"]
        ]

        # Color-code the numeric columns
        st.dataframe(
            df.style.background_gradient(
                subset=["answer_relevancy","faithfulness","context_precision","token_overlap_f1"],
                cmap="RdYlGn", vmin=0, vmax=1,
            ),
            width="stretch",
        )

        # ── Metric distribution bar chart ─────────────────────────────────────
        st.subheader("Score Distribution by Question")
        chart_data = []
        for r in valid:
            for metric in ["answer_relevancy", "faithfulness", "context_precision"]:
                chart_data.append({"Question ID": r["id"], "Metric": metric, "Score": r[metric]})
        dist_df = pd.DataFrame(chart_data)
        dist_fig = px.bar(
            dist_df, x="Question ID", y="Score", color="Metric",
            barmode="group", template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Evaluation Scores per Question",
            labels={"Score": "Score (0–1)"},
        )
        dist_fig.update_layout(yaxis_range=[0, 1.1])
        st.plotly_chart(dist_fig, width="stretch")

        # ── Detailed answers expander ─────────────────────────────────────────
        with st.expander("🔍 View generated vs. expected answers"):
            for r in valid:
                st.markdown(f"**Q{r['id']}: {r['question']}**")
                col_a, col_b = st.columns(2)
                col_a.markdown("**Expected:**")
                col_a.markdown(r.get("expected_answer", "_no reference_"))
                col_b.markdown("**Generated:**")
                col_b.markdown(r.get("generated_answer", ""))
                st.divider()


def _build_radar_chart(ar: float, fa: float, cp: float, to: float) -> go.Figure:
    """
    Build a radar (spider) chart summarising all four evaluation metrics.
    """
    categories = [
        "Answer Relevancy", "Faithfulness",
        "Context Precision", "Token Overlap F1",
    ]
    values = [ar, fa, cp, to]
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed     = values + [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(37, 99, 235, 0.25)",
        line=dict(color="#2563EB", width=2),
        name="Average Scores",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont_size=10),
            angularaxis=dict(tickfont_size=12),
        ),
        template="plotly_white",
        title="Evaluation Metrics Radar",
        showlegend=False,
        height=420,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  Main layout
# ════════════════════════════════════════════════════════════════════════════

def main():
    render_sidebar()

    # Generate sample PDF if it doesn't exist yet (runs silently on first launch)
    sample_path = Path("data/TechCorp_Annual_Report_2023.pdf")
    if not sample_path.exists():
        try:
            _generate_sample_pdf()
        except Exception:
            pass  # User will see the "Generate" button in the sidebar

    # Main tab layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Overview",
        "💬 Q&A Chat",
        "📈 Visualizations",
        "🧪 Evaluation",
    ])

    with tab1:
        tab_overview()
    with tab2:
        tab_chat()
    with tab3:
        tab_visualizations()
    with tab4:
        tab_evaluation()


if __name__ == "__main__":
    main()
