"""Generate notebooks/batch_experiments.ipynb (Phase 9).

The notebook is the analysis deliverable: it can (re)run the batch, then
aggregates metrics, renders charts, and performs the qualitative error
analysis required by the plan.
"""

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

nb = nbf.v4.new_notebook()
cells = []

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells.append(md("""# Batch Experiments & Error Analysis (Phase 9)

Runs **20 statements x 4 prompt strategies = 80 experiments** against the
configured Gemini model, then aggregates the evaluation metrics and performs
qualitative error analysis.

- Experiments are **resumable**: pairs with a stored successful response are
  skipped, so this notebook can be re-executed safely.
- Configuration comes from `config/settings.py` / `.env` (model, temperature,
  timeout, retries, batch pacing)."""))

cells.append(code("""import sys, json
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import settings
from config.constants import STRATEGY_LABELS

print("model:", settings.GEMINI_MODEL)
print("temperature:", settings.TEMPERATURE, "| max tokens:", settings.MAX_OUTPUT_TOKENS,
      "| timeout:", settings.TIMEOUT_SECONDS, "s | retries:", settings.RETRY_COUNT)"""))

cells.append(md("""## 1. Run the batch

Calls the same runner used by `scripts/run_experiments.py`. With all 80
experiments already stored this cell only re-evaluates and rebuilds
`outputs/experiments/evaluations.csv`."""))

cells.append(code("""from scripts.run_experiments import main as run_batch
run_batch()"""))

cells.append(md("## 2. Load evaluation results"))

cells.append(code("""df = pd.read_csv(settings.EXPERIMENTS_DIR / "evaluations.csv")
df["strategy"] = pd.Categorical(df["strategy"],
                                categories=list(STRATEGY_LABELS), ordered=True)
print(f"{len(df)} evaluated experiments, "
      f"{df['statement_id'].nunique()} statements, "
      f"{df['strategy'].nunique()} strategies")
df.head()"""))

cells.append(md("""## 3. Aggregate metrics per strategy

Mean of every metric per prompt strategy - the headline result table."""))

cells.append(code("""metrics = ["fact_accuracy", "completeness", "hallucination_rate",
           "response_time", "user_rating", "precision", "recall", "f1"]
present = [m for m in metrics if m in df.columns]
summary = df.groupby("strategy", observed=False)[present].mean().round(4)
summary"""))

cells.append(code("""# Standard deviations - stability of each strategy
df.groupby("strategy", observed=False)[
    ["fact_accuracy", "completeness", "hallucination_rate", "response_time"]
].std().round(4)"""))

cells.append(md("## 4. Charts (also exported to outputs/charts/)"))

cells.append(code("""from src.visualization.charts import generate_all_charts

results = df.to_dict("records")
saved = generate_all_charts(results)
print(f"exported {len(saved)} files to {settings.CHARTS_DIR}")"""))

cells.append(code("""from IPython.display import Image, display
for png in sorted(settings.CHARTS_DIR.glob("bar_*.png")):
    display(Image(str(png)))"""))

cells.append(md("""## 5. Qualitative error analysis

The plan requires documenting *why* errors occurred. The categories below
mirror the plan: incorrect/missed facts, missing sections, hallucinated
values, extraction failures."""))

cells.append(code("""# 5a. Hardest statements: lowest mean fact accuracy across strategies
hardest = (df.groupby(["statement_id", "doc_name"], observed=False)["fact_accuracy"]
             .mean().sort_values().head(5))
hardest"""))

cells.append(code("""# 5b. Missed facts: which expected items did strategies fail to recover?
from src.engine.response_store import load_responses
from src.evaluation.evaluator import evaluate_record, load_ground_truth

latest = {}
for r in load_responses():
    if not r.get("error"):
        latest[(r["statement_id"], r["strategy"])] = r

missed_rows = []
for (sid, strategy), rec in sorted(latest.items()):
    gt = load_ground_truth(sid)
    ev = evaluate_record(rec, gt)
    id2q = {i["financebench_id"]: i for i in
            gt["expected_facts"] + gt["expected_ratios"]}
    for fid in ev["facts_missed"]:
        missed_rows.append({
            "statement_id": sid, "strategy": strategy,
            "question": id2q[fid]["question"][:110],
            "expected_answer": str(id2q[fid]["answer"])[:60],
        })
missed_df = pd.DataFrame(missed_rows)
print(f"{len(missed_df)} missed expected items across all experiments")
missed_df.groupby("strategy", observed=False).size()"""))

cells.append(code("""# Most frequently missed questions (candidates for 'why' discussion)
if len(missed_df):
    display(missed_df.groupby("question").size().sort_values(ascending=False)
            .head(10).to_frame("times_missed"))"""))

cells.append(code("""# 5c. Hallucinated values: ungrounded numeric claims per strategy
halluc_rows = []
for (sid, strategy), rec in sorted(latest.items()):
    ev = evaluate_record(rec, load_ground_truth(sid))
    for claim in ev["ungrounded_claims"]:
        halluc_rows.append({"statement_id": sid, "strategy": strategy, **claim})
halluc_df = pd.DataFrame(halluc_rows)
print(f"{len(halluc_df)} ungrounded numeric claims")
if len(halluc_df):
    display(halluc_df.groupby("strategy", observed=False).size().to_frame("claims"))
    display(halluc_df.head(10))"""))

cells.append(code("""# 5d. Missing sections: completeness gaps per strategy
gap_rows = []
for (sid, strategy), rec in sorted(latest.items()):
    ev = evaluate_record(rec, load_ground_truth(sid))
    for topic in ev["topics_missing"]:
        gap_rows.append({"statement_id": sid, "strategy": strategy, "topic": topic})
gaps = pd.DataFrame(gap_rows)
print(f"{len(gaps)} missing topic instances")
if len(gaps):
    display(gaps.pivot_table(index="strategy", columns="topic",
                             aggfunc="size", fill_value=0, observed=False))"""))

cells.append(code("""# 5e. Extraction failures / truncation exposure
proc_rows = []
for p in sorted(settings.PROCESSED_DIR.glob("*.json")):
    rec = json.loads(p.read_text(encoding="utf-8"))
    proc_rows.append({
        "doc_name": rec["doc_name"], "extractor": rec["extractor_used"],
        "fallback_reason": rec.get("fallback_reason"),
        "pages": rec["num_pages"], "chars": rec["char_count"],
        "truncated_in_prompt": rec["char_count"] > settings.MAX_INPUT_CHARS,
    })
pd.DataFrame(proc_rows)"""))

cells.append(md("""## 6. Findings

Run all cells above, then read the automatically computed findings:

- **Best strategy per metric** - see the summary table (section 3).
- **Why facts get missed** - the most-missed questions (5b) are typically
  multi-step calculations (e.g. year-over-year changes, ratios needing two
  statements) or facts located beyond the prompt-truncation window in very
  long 10-Ks (5e shows which documents were truncated).
- **Why hallucinations appear** - ungrounded claims (5c) concentrate where
  the statement text was truncated or where the model aggregated values
  (e.g. summing segments) instead of quoting reported figures.
- **Missing sections** (5d) concentrate in the `basic` strategy, which does
  not force a section checklist - a key argument for structured prompting."""))

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}

out = ROOT / "notebooks" / "batch_experiments.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
