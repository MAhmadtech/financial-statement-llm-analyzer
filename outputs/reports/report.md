# Evaluating Prompt Engineering Strategies for LLM-Based Financial Statement Analysis

**Project report - Financial LLM Prompting Lab**
Date: August 2026

---

## 1. Introduction

Large language models (LLMs) are increasingly used to read and interpret
corporate financial disclosures. How a task is *prompted* materially changes
output quality, yet quantified comparisons on real filings remain scarce.
This project builds an end-to-end experimental pipeline that analyzes real
SEC filings with a single LLM under **four prompt strategies** - basic,
role-based, structured, and few-shot - and measures their effect on
**fact-extraction accuracy, completeness, hallucination rate, response time,
and user preference**.

## 2. Literature Review

**BloombergGPT** (Wu et al., 2023) demonstrated that a 50B-parameter model
trained on a mixed general + financial corpus (FinPile) outperforms
similarly-sized general models on financial NLP tasks (sentiment, NER,
QA) while remaining competitive on general benchmarks. It motivates
domain-aware evaluation but its weights are proprietary, which limits
reproducibility for downstream research.

**FinGPT** (Yang et al., 2023) took the opposite, open-source route:
lightweight adaptation (LoRA) of open base models on rapidly refreshed
financial data streams. Its central argument - that data pipelines and
task-specific adaptation matter more than parameter count - supports the
premise here: careful *prompting* of an accessible commercial model may
recover much of the benefit of specialized models at negligible cost.

**Prompt Engineering Guide** (DAIR.AI, ongoing) systematizes strategies
including zero-shot, role/persona, output-structuring, and few-shot
prompting. It reports that structural constraints and worked examples
generally improve faithfulness on extraction-style tasks - a hypothesis this
project tests quantitatively in the financial domain.

**FinanceBench** (Islam et al., 2023, Patronus AI) supplies the evaluation
substrate: 10,231 question-answer pairs grounded in real filings (150 open
source), with evidence excerpts. The authors found frontier-model failure
rates of 60-80% on retrieval-free financial QA, underscoring the difficulty
of the task and the value of measured evaluation. This project derives its
ground truth from the open-source subset.

Additional context: FinQA (Chen et al., 2021) and TAT-QA (Zhu et al., 2021)
established numerical-reasoning-over-tables benchmarks and consistently show
that multi-step arithmetic is the dominant failure mode - anticipating the
error patterns observed here.

## 3. Methodology

### 3.1 Dataset and ground truth
- Source: FinanceBench open-source subset (150 QA pairs over 361 documents).
- Selection: the 20 documents with the richest question coverage
  (10-Ks, 10-Qs, an 8-K, and earnings releases across sectors) form the
  evaluation set (`data/eval_dataset/manifest.csv`).
- Each statement's ground truth file (`statement_XX.json`) contains:
  `pdf_path`, `expected_summary`, `expected_facts`, `expected_ratios`,
  `expected_risks`, `expected_recommendations`.
- Derivation rules: QA pairs are classified into **facts** vs **ratios** by
  ratio-keyword matching on the question; **risks** are QA pairs whose
  question or answer flags risk language. FinanceBench contains no analyst
  recommendations, so `expected_recommendations` holds coverage
  *expectations* derived from each document's ratio/risk topics; they feed
  the completeness metric only, never accuracy.

### 3.2 PDF extraction
- Primary extractor **pdfplumber** (text + tables); **PyMuPDF** fallback
  triggers when average text yield falls below 80 chars/page or on error.
- Preprocessing removes recurring page headers/footers (fingerprinted edge
  lines on >= 50% of pages), normalizes whitespace, and preserves negative
  numbers `(1,234)` and currency formatting `$1,577`.
- Outputs: `data/processed/<doc>.json` (+ `.txt`), with clean-text backups in
  `data/clean_text_backup/`.

### 3.3 Prompt design (four strategies, versioned)
Prompts live in `config/prompts.yaml` as `strategy_vN` versions; experiments
record the exact version used (`basic_v1`, `role_based_v1`, `structured_v1`,
`few_shot_v1`).

1. **Basic** - direct "analyze this statement" instruction.
2. **Role-based** - CPA/CFA/senior-analyst persona with professional
   skepticism instruction.
3. **Structured** - forces nine sections (Executive Summary, Revenue,
   Expenses, Profitability, Liquidity, Solvency, Financial Ratios, Risks,
   Recommendations).
4. **Few-shot** - one hand-written worked example (fictional manufacturer,
   arithmetically consistent) followed by the target statement.

All four strategies share one output contract: a human-readable report plus
a fenced JSON block with keys Revenue, Expenses, Net Profit, Current Ratio,
Debt to Equity, ROE, Risks, Recommendations - with an explicit instruction
to write "not reported" rather than invent values.

### 3.4 LLM configuration
- Backend: Google Gemini via the `google-genai` SDK; adapter interface with
  configuration flag allows plugging GPT-4.1 Mini / Llama 3.1 8B without
  pipeline changes (stubs in this project scope).
- **Model: `gemini-3.5-flash-lite`** (see Deviations, section 7).
- Temperature 0.2; max output tokens 4,096; timeout (180 s on the
  experiment machine; 60 s default); 3 retry attempts with exponential
  backoff; token usage logged per call.
- Statements longer than 60,000 characters are reduced by a **financial
  section selector**: the clean text is split into 3,000-character blocks,
  each scored by financial-keyword frequency and digit density, and the
  top-scoring blocks are kept in document order. On the 15 over-budget
  filings this keeps 44/46 ground-truth values inside the prompt budget vs
  37/46 for naive head-truncation (e.g. Verizon 0/2 -> 2/2, PepsiCo
  2/4 -> 4/4). The reduction flag is stored per experiment.

### 3.5 Evaluation metrics
1. **Fact extraction accuracy** - share of expected facts + ratios whose
   answer numbers appear in the response (1% relative tolerance,
   unit-scale-aware: millions/billions/thousands). Qualitative answers match
   at >= 40% significant-term overlap.
2. **Completeness** - checklist coverage of revenue, expenses, profit,
   liquidity, solvency, risks, recommendations (keyword families).
3. **Hallucination rate** - ungrounded numeric claims among the structured
   JSON's factual fields (Revenue, Expenses, Net Profit) versus the source
   statement text. Derived ratio fields are excluded because correct ratios
   rarely appear verbatim in filings.
4. **Response time** - wall-clock latency per API call.
5. **User rating** - 1-5 ratings collected in the Streamlit app
   (`outputs/ratings.csv`).
6. **Advanced (optional)** - precision (grounded JSON claims / all JSON
   claims), recall (= fact accuracy), F1.

### 3.6 Experimental setup
- Hardware: Windows 10 Pro workstation (consumer laptop-class), Python 3.12.
- Software: pdfplumber, PyMuPDF, google-genai, pandas, matplotlib,
  Streamlit; full list in `requirements.txt`.
- Design: 20 statements x 4 strategies = **80 experiments**, single model,
  fixed generation parameters; batch runner is resumable and paced
  (10 s between calls) for API rate limits.
- Every experiment stores complete metadata (statement id, strategy, prompt
  version, model, extracted text, full prompt, response, parsed JSON,
  latency, timestamp, token counts, error) in `outputs/responses/`.

## 4. Results

> This section is generated from `outputs/experiments/evaluations.csv` and
> `outputs/charts/strategy_summary.csv` after the batch run. See
> `notebooks/batch_experiments.ipynb` for the full analysis and charts
> (accuracy / completeness / hallucination / response time comparisons).

All 80 experiments (20 statements x 4 strategies) completed without API
failures on `gemini-3.5-flash-lite`.

| Strategy | Fact accuracy | Completeness | Hallucination rate | Response time (s) | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Basic | 0.492 | 1.000 | 0.033 | 4.52 | 0.967 | 0.492 | 0.544 |
| Role-based | 0.507 | 1.000 | 0.042 | 5.19 | 0.958 | 0.507 | 0.557 |
| **Structured** | **0.626** | 1.000 | 0.033 | 7.62 | 0.967 | **0.626** | **0.693** |
| Few-shot | 0.536 | 1.000 | **0.017** | **4.48** | **0.983** | 0.536 | 0.593 |

**Key findings**

1. **Structured prompting is the clear accuracy winner**: 0.626 mean fact
   accuracy, a +27% relative improvement over the basic prompt (0.492), and
   the best F1 (0.693). Forcing the nine-section skeleton with an explicit
   "compute ratios and show your inputs" instruction makes the model surface
   more of the expected figures. The cost is latency: 7.6 s mean vs ~4.5 s,
   as responses are roughly twice as long.
2. **Few-shot prompting is the faithfulness winner**: lowest hallucination
   rate (0.017, half the basic rate), highest precision (0.983), and the
   fastest responses (4.48 s). The worked example appears to anchor the
   model to quote-reported-figures behavior, but it recovers fewer expected
   facts than structured prompting (0.536).
3. **The role-based persona adds almost nothing over basic** (0.507 vs
   0.492 accuracy, slightly *higher* hallucination rate) - persona wording
   alone, without structural constraints, does not change what the model
   extracts.
4. **Completeness saturated at 1.0 for all strategies** - the shared JSON
   output contract (which demands Risks and Recommendations keys) already
   guarantees topic coverage, so the checklist metric does not discriminate
   between strategies in this setup. A stricter, depth-weighted completeness
   measure is future work.
5. **Accuracy variance is high across statements** (std ~0.33-0.40):
   statements whose expected answers require multi-step arithmetic (CAGR,
   ratio changes across years) score near zero for every strategy, while
   direct-lookup statements score near one (see error analysis).

Charts exported to `outputs/charts/` (PNG + PDF), summary table to
`outputs/charts/strategy_summary.csv`.

## 5. Error Analysis

Computed in sections 5a-5e of the notebook:

- **Missed facts** concentrate in multi-step calculations (year-over-year
  changes, ratios requiring values from two statements) and in facts located
  beyond the 60k-character truncation window of very long 10-Ks.
- **Hallucinated values** cluster where the model aggregates segment values
  itself instead of quoting reported totals, and on truncated inputs.
- **Missing sections** occur predominantly under the basic strategy, which
  imposes no section checklist.
- **Extraction failures**: pdfplumber succeeded on all 20 documents (no
  fallback needed); the risk that remains is table-to-text linearization
  noise in dense financial tables.

## 6. Limitations

- **PDF extraction quality** bounds everything downstream; linearized tables
  lose some structure even when extraction succeeds.
- **Small evaluation dataset** (20 documents, 150 QA pairs total in the
  open-source subset) limits statistical power.
- **Single-model dependence**: findings are for one Gemini Flash-tier model;
  strategy rankings may not transfer across models.
- **Automated hallucination labeling** is a proxy (numeric grounding of JSON
  fields); it misses qualitative false claims and can flag legitimate
  derived values - true labeling is partly subjective.
- **Prompt budget** of 60k characters means very long 10-Ks are analyzed
  from their highest-density financial sections only; the section selector
  still leaves ~4% of ground-truth values outside the prompt (2/46).

## 7. Deviations from the original plan

- **Model**: the plan specified `gemini-2.5-flash`, which Google retired for
  new API keys ("no longer available to new users") before experiments
  began; the mainline `gemini-3.5-flash` also rejected large prompts with
  persistent 503 capacity errors for this key. Experiments therefore use
  **`gemini-3.5-flash-lite`**, the closest Flash-tier model that reliably
  accepted the workload. The model name is configuration, not code
  (`GEMINI_MODEL` in `.env`).
- **Timeout**: raised from 60 s to 180 s via `.env` override on the
  experiment machine due to slow network I/O on ~56 KB prompts.
- **Jupyter stack**: installed `ipykernel`/`nbclient` instead of the full
  JupyterLab metapackage, which fails to install on Windows without
  long-path support; the notebook itself is standard `.ipynb`.
- **Second backend**: GPT-4.1 Mini / Llama adapters are interface stubs by
  scope decision; enabling them is a config flag plus one class.

## 8. Future Work

- Larger evaluation datasets (full FinanceBench, ECTSum, earnings calls).
- Multi-model comparison through the existing adapter interface.
- Retrieval-augmented generation over full filings to remove the truncation
  limitation.
- Fine-tuning / distillation of a small financial model on the stored
  experiment corpus.
- OCR (e.g., Tesseract) for scanned statements.
- Automated financial ratio validation: recompute ratios from extracted
  tables and check the model's arithmetic exactly.

## References

- Wu, S. et al. (2023). *BloombergGPT: A Large Language Model for Finance.*
  arXiv:2303.17564.
- Yang, H., Liu, X.-Y., Wang, C. D. (2023). *FinGPT: Open-Source Financial
  Large Language Models.* arXiv:2306.06031.
- Islam, P. et al. (2023). *FinanceBench: A New Benchmark for Financial
  Question Answering.* arXiv:2311.11944.
- DAIR.AI. *Prompt Engineering Guide.* https://www.promptingguide.ai
- Chen, Z. et al. (2021). *FinQA: A Dataset of Numerical Reasoning over
  Financial Data.* EMNLP 2021.
- Zhu, F. et al. (2021). *TAT-QA: A Question Answering Benchmark on a Hybrid
  of Tabular and Textual Content in Finance.* ACL 2021.
