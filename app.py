"""Phase 8: Streamlit web application.

Run:  streamlit run app.py

Sidebar: choose a statement (eval dataset or PDF upload), prompt strategy
(or all four), model backend, then Run. Main page: PDF preview, extracted
text, generated analysis, structured JSON, comparison table, charts, and a
1-5 rating widget stored to outputs/ratings.csv.
"""

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import settings
from config.constants import STRATEGY_LABELS
from src.engine.analyzer import analyze_statement, run_all_strategies
from src.evaluation.evaluator import evaluate_record, load_ground_truth
from src.evaluation.ratings import append_rating
from src.extraction.pdf_extractor import load_processed_text, process_pdf

st.set_page_config(page_title="Financial LLM Prompting Lab", layout="wide")

RUN_ALL = "Run all strategies"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def manifest_rows() -> list[dict]:
    import csv

    path = settings.EVAL_DATASET_DIR / "manifest.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@st.cache_data(show_spinner="Rendering PDF preview...")
def pdf_preview_png(pdf_path: str) -> bytes:
    import fitz

    with fitz.open(pdf_path) as doc:
        return doc[0].get_pixmap(dpi=90).tobytes("png")


@st.cache_data(show_spinner="Extracting PDF text...")
def extract_uploaded(file_bytes: bytes, filename: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    return process_pdf(Path(tmp_path), doc_name=f"upload_{Path(filename).stem}")


def metric_row(evaluation: dict) -> dict:
    return {
        "Strategy": STRATEGY_LABELS.get(evaluation["strategy"], evaluation["strategy"]),
        "Fact accuracy": evaluation.get("fact_accuracy"),
        "Completeness": evaluation.get("completeness"),
        "Hallucination rate": evaluation.get("hallucination_rate"),
        "Response time (s)": evaluation.get("response_time"),
        "Precision": evaluation.get("precision"),
        "Recall": evaluation.get("recall"),
        "F1": evaluation.get("f1"),
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Financial LLM Lab")

source = st.sidebar.radio("Statement source", ["Evaluation dataset", "Upload PDF"])

statement_id = None
doc_name = ""
pdf_path = None
statement_text = None
ground_truth = None

if source == "Evaluation dataset":
    rows = manifest_rows()
    if not rows:
        st.sidebar.error("No eval dataset found - run scripts/build_ground_truth.py")
    else:
        options = {f"{r['statement_id']} - {r['doc_name']}": r for r in rows}
        chosen = st.sidebar.selectbox("Statement", list(options))
        row = options[chosen]
        statement_id, doc_name = row["statement_id"], row["doc_name"]
        pdf_path = settings.RAW_STATEMENTS_DIR / f"{doc_name}.pdf"
        try:
            statement_text = load_processed_text(doc_name)
        except FileNotFoundError:
            st.sidebar.warning(
                "Not extracted yet - run scripts/extract_all.py (or click Run, "
                "which extracts on the fly)."
            )
        try:
            ground_truth = load_ground_truth(statement_id)
        except FileNotFoundError:
            pass
else:
    uploaded = st.sidebar.file_uploader("Upload a financial statement PDF", type="pdf")
    if uploaded is not None:
        record = extract_uploaded(uploaded.getvalue(), uploaded.name)
        statement_id = record["doc_name"]
        doc_name = record["doc_name"]
        pdf_path = Path(record["pdf_path"])
        statement_text = record["text"]

strategy_choice = st.sidebar.selectbox(
    "Prompt strategy", [RUN_ALL] + list(STRATEGY_LABELS.values())
)

backend = st.sidebar.selectbox(
    "Model", ["gemini (Gemini 2.5 Flash)", "gpt41mini (stub)", "llama (stub)"]
)
if not backend.startswith("gemini"):
    st.sidebar.info(
        "Only the Gemini backend is live in this project; the others are "
        "adapter stubs (see src/llm/adapters.py)."
    )

run_clicked = st.sidebar.button(
    "Run analysis", type="primary",
    disabled=statement_text is None and pdf_path is None,
)

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("Financial Statement Analysis - Prompt Strategy Lab")
st.caption(
    f"Model: {settings.GEMINI_MODEL} | temperature {settings.TEMPERATURE} | "
    f"max tokens {settings.MAX_OUTPUT_TOKENS} | timeout {settings.TIMEOUT_SECONDS}s"
)

if statement_id is None:
    st.info("Pick a statement from the evaluation dataset or upload a PDF.")
    st.stop()

left, right = st.columns([1, 2])
with left:
    st.subheader("PDF preview")
    if pdf_path and Path(pdf_path).exists():
        st.image(pdf_preview_png(str(pdf_path)))
    else:
        st.write("No PDF available.")

with right:
    st.subheader("Extracted text")
    if statement_text is None and pdf_path and Path(pdf_path).exists():
        if st.button("Extract now"):
            record = process_pdf(Path(pdf_path), doc_name)
            statement_text = record["text"]
    if statement_text:
        st.text_area(
            "Clean statement text (input to the LLM)",
            statement_text[:20000],
            height=340,
        )
        st.caption(f"{len(statement_text):,} characters total "
                   f"(prompt uses first {settings.MAX_INPUT_CHARS:,})")

if run_clicked:
    if statement_text is None:
        record = process_pdf(Path(pdf_path), doc_name)
        statement_text = record["text"]

    if not backend.startswith("gemini"):
        st.error("Selected backend is a stub - switch to Gemini to run.")
        st.stop()

    label_to_key = {v: k for k, v in STRATEGY_LABELS.items()}
    with st.spinner("Calling Gemini..."):
        if strategy_choice == RUN_ALL:
            results = run_all_strategies(statement_text, statement_id, doc_name)
        else:
            key = label_to_key[strategy_choice]
            results = {
                key: analyze_statement(statement_text, key, statement_id, doc_name)
            }
    st.session_state["results"] = results

results = st.session_state.get("results")
if results:
    st.divider()
    st.header("Results")

    evaluations = []
    for strategy, record in results.items():
        if record.get("error"):
            st.error(f"{STRATEGY_LABELS[strategy]}: {record['error']}")
            continue
        if ground_truth is not None:
            try:
                evaluations.append(evaluate_record(record, ground_truth))
            except Exception as exc:  # evaluation must never break the app
                st.warning(f"Evaluation failed for {strategy}: {exc}")

    tabs = st.tabs([STRATEGY_LABELS[s] for s in results])
    for tab, (strategy, record) in zip(tabs, results.items()):
        with tab:
            if record.get("error"):
                st.error(record["error"])
                continue
            st.markdown(record["response"])
            with st.expander("Structured JSON"):
                st.json(record.get("structured_json") or {})
            st.caption(
                f"latency {record['latency']}s | input tokens "
                f"{record['input_tokens']} | output tokens {record['output_tokens']}"
            )
            st.download_button(
                "Download report",
                record["response"] or "",
                file_name=f"{statement_id}_{strategy}_report.md",
                key=f"dl_{strategy}",
            )

            st.markdown("**Rate this analysis**")
            rating = st.radio(
                "Rating (1 = poor, 5 = excellent)",
                [1, 2, 3, 4, 5],
                horizontal=True,
                key=f"rating_{strategy}",
                index=None,
            )
            if st.button("Save rating", key=f"save_rating_{strategy}"):
                if rating is None:
                    st.warning("Choose a rating first.")
                else:
                    append_rating(statement_id, strategy, rating, doc_name)
                    st.success("Rating saved to outputs/ratings.csv")

    if evaluations:
        st.subheader("Comparison")
        import pandas as pd

        table = pd.DataFrame([metric_row(e) for e in evaluations])
        st.dataframe(table, hide_index=True)

        chart_df = table.set_index("Strategy")[
            ["Fact accuracy", "Completeness", "Hallucination rate"]
        ]
        st.bar_chart(chart_df)
    elif ground_truth is None:
        st.info(
            "Uploaded statements have no ground truth, so accuracy metrics "
            "are unavailable - completeness and latency are shown per tab."
        )
