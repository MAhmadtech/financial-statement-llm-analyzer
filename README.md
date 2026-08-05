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
  preprocessing/ header/footer removal, whitespace normalization
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

After a batch run: metrics per experiment in
`outputs/experiments/evaluations.csv`, per-strategy summary and charts in
`outputs/charts/`, full report in `outputs/reports/report.md`, and the
analysis notebook in `notebooks/batch_experiments.ipynb`.

## Security

The API key lives only in `.env`, which is gitignored; `.env.example`
documents the expected variables without secrets.
