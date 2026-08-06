# Financial LLM Prompting Lab

Compares four prompt engineering strategies (basic, role-based, structured,
few-shot) for LLM-based financial statement analysis on real SEC filings,
measuring fact accuracy, completeness, hallucination rate, response time,
and user ratings. Built on Google Gemini + the FinanceBench dataset.

## Quick start

```bash
# 1. Install dependencies (Python 3.11+)
pip install -r requirements.txt

# 2. Configure secrets
copy .env.example .env       # then put your Gemini API key in .env

# 3. Get the dataset (once): clone FinanceBench next to this README
git clone --depth 1 https://github.com/patronus-ai/financebench.git

# 4. Build the evaluation dataset (20 statements + ground truth)
python scripts/build_ground_truth.py

# 5. Extract + preprocess the PDFs
python scripts/extract_all.py

# 6. Run the web app
streamlit run app.py

# 7. Or run the full batch: 20 statements x 4 strategies = 80 experiments
python scripts/run_experiments.py     # resumable; paced for rate limits
```

## Project structure

```
config/          settings.py (env-driven config), constants.py, prompts.yaml
data/            raw_statements/, processed/, eval_dataset/ (ground truth),
                 expected_outputs/ (CSV fallbacks), clean_text_backup/
src/
  extraction/    pdfplumber primary + PyMuPDF fallback
  preprocessing/ header/footer removal, financial-section selector
  prompts/       versioned prompt builder (prompts.yaml)
  llm/           Gemini client (retries/timeout/token logging) + adapter stubs
  engine/        analyzer pipeline + response store (full metadata JSON)
  evaluation/    metrics: accuracy, completeness, hallucination, ratings, F1
  visualization/ bar/line/box charts -> PNG/PDF/CSV
outputs/         responses/, experiments/, charts/, reports/, logs/
notebooks/       batch_experiments.ipynb (batch run + error analysis)
scripts/         build_ground_truth.py, extract_all.py, run_experiments.py
app.py           Streamlit application
```

## Configuration

Everything tunable lives in `config/settings.py`, overridable via `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | - | required |
| `GEMINI_MODEL` | `gemini-2.5-flash` | see note below |
| `LLM_TEMPERATURE` | `0.2` | generation temperature |
| `LLM_MAX_OUTPUT_TOKENS` | `4096` | response cap |
| `LLM_TIMEOUT_SECONDS` | `60` | per-call timeout |
| `LLM_RETRY_COUNT` | `3` | retries with exponential backoff |
| `LLM_MAX_INPUT_CHARS` | `60000` | statement truncation for prompts |
| `LLM_BACKEND` | `gemini` | `gpt41mini` / `llama` are adapter stubs |
| `BATCH_SLEEP_SECONDS` | `10` | pacing between batch API calls |

> **Model note:** `gemini-2.5-flash` is retired for new API keys. This
> project's experiments used `GEMINI_MODEL=gemini-3.5-flash-lite` (set in
> `.env`), the closest Flash-tier model that reliably accepted the workload.

## Results

**80/80 experiments succeeded** (20 statements x 4 strategies,
`gemini-3.5-flash-lite`, temperature 0.2):

| Strategy | Fact accuracy | Completeness | Hallucination rate | Response time (s) | F1 |
|---|---|---|---|---|---|
| Basic | 0.492 | 1.000 | 0.033 | 4.52 | 0.544 |
| Role-based | 0.507 | 1.000 | 0.042 | 5.19 | 0.557 |
| **Structured** | **0.626** | 1.000 | 0.033 | 7.62 | **0.693** |
| Few-shot | 0.536 | 1.000 | **0.017** | **4.48** | 0.593 |

Key findings:

- **Structured prompting** (forced nine-section report) recovers the most
  facts: +27% relative accuracy over the basic prompt, at ~1.7x the latency.
- **Few-shot prompting** is the most faithful (half the hallucination rate
  of basic, highest precision 0.983) and the fastest.
- A **role-based persona alone changes almost nothing** versus the basic
  prompt - structure beats persona.
- Long filings are handled by a financial-section selector that keeps the
  highest-density financial text within the 60k-char prompt budget
  (44/46 ground-truth values retained vs 37/46 with naive truncation).
- Pipeline verified on an unseen statement (NIKE 2021 10-K): extracted and
  analyzed correctly with real reported figures.

Full details: metrics per experiment in `outputs/experiments/evaluations.csv`,
charts in `outputs/charts/`, research report in `outputs/reports/report.md`,
analysis notebook in `notebooks/batch_experiments.ipynb`.

## Security

The API key lives only in `.env`, which is gitignored; `.env.example`
documents the expected variables without secrets.
