# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Portfolio POC: AI system for US higher education enrollment decline. Three integrated modules feeding a unified Streamlit dashboard. Zero budget — open data + free tiers only.

**Current state:** All 6 phases complete. Ready for GitHub push + Streamlit Cloud deployment.

| Phase | Status | Key output |
|-------|--------|------------|
| 1 — Data Acquisition | ✅ | `data/raw/`, `data/synthetic/applicants.csv`, `data/corpus/` |
| 2 — Demographic Forecasting | ✅ | `data/output/forecasts.csv`, SARIMA MAPE=3.9% |
| 3 — Lead & Yield Prediction | ✅ | `models/*.pkl`, `data/output/*_predictions.csv`, bias audit complete |
| 4 — RAG Chatbot | ✅ | 3040 chunks indexed, Groq streaming, eval 5/5 trick refusal |
| 5 — Unified Dashboard | ✅ | Full overview: KPI cards, funnel chart, demographic shift chart, module links; Chatbot Eval tab added to Model Performance |
| 6 — Portfolio Packaging | ✅ | README.md with hero/metrics/impact/setup, model cards finalized, MIT LICENSE, models/*.pkl unblocked in .gitignore |

## Run Commands

```bash
# Install pinned dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run streamlit_app.py

# Execute any notebook headlessly
jupyter nbconvert --to notebook --execute notebooks/03_lead_scoring.ipynb --inplace

# Re-train both ML models (runs notebooks 03 + 04)
jupyter nbconvert --to notebook --execute notebooks/03_lead_scoring.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/04_yield_prediction.ipynb --inplace

# Regenerate synthetic applicants only
python -m src.data_synthesis

# Phase 4 — build/rebuild ChromaDB vector index from data/corpus/
python src/rag/embedding.py

# Re-fetch raw NCES/IPEDS data (idempotent)
jupyter nbconvert --to notebook --execute notebooks/01_data_acquisition.ipynb --inplace
```

## Architecture

Three independent modules + unified dashboard:

```
NCES/IPEDS data ──► src/forecasting.py ─────────────────────────────┐
                                                                      │
data/synthetic/  ──► src/lead_model.py  ──► models/lead_scoring.pkl ►  streamlit_app.py
applicants.csv   ──► src/yield_model.py ──► models/yield_prediction  ►  pages/
                         └── SHAP (TreeExplainer) ──► *_predictions  │
                                                                      │
data/corpus/ ──► src/rag/ingest.py ──► src/rag/embedding.py ────────►  pages/4_Chatbot.py
              └──► data/vector_db/ (ChromaDB)  ◄── src/rag/retrieval.py
                                                     src/rag/generation.py (Groq)
```

**Module 1 — Demographic Forecasting** (`src/forecasting.py`)
- Prophet (primary) vs SARIMA (benchmark); train on 1980–2017, test on 2018–2023
- SARIMA wins: MAPE=3.9%; Prophet over-predicts (extrapolates 1980–2017 growth trend)
- Always show 80% and 95% confidence intervals — never point estimates
- Outputs: `data/output/forecasts.csv`

**Module 2 — Lead & Yield Prediction** (`src/lead_model.py`, `src/yield_model.py`)
- Lead target: `inquired_to_applied` (all 50k rows); Yield target: `admit_to_enroll` (admitted==1 only, ~8.9k rows)
- Both: 70/15/15 stratified split → GridSearchCV (5-fold) on XGBoost → Platt calibration on val set
- Lead: `scale_pos_weight` for class imbalance. Yield: SMOTE on training set only — never val/test
- `shap.TreeExplainer` on raw XGBoost (pre-calibration); top-3 per prediction stored in `*_predictions.csv`
- Artifact format: joblib dict with keys `xgb`, `xgb_base`, `lr`, `preprocessor`, `feature_names`, `explainer`, `metrics`, `split`, `X_test`, `y_test`, `test_idx`
- Load via `lm.load()` / `ym.load()` — paths default to `models/lead_scoring.pkl` / `models/yield_prediction.pkl`

**Module 3 — RAG Chatbot** (`src/rag/`)
- Pipeline: `ingest.py` → `embedding.py` → ChromaDB → `retrieval.py` → `generation.py` → stream
- Embedding: `all-MiniLM-L6-v2` (384-dim), 500-token chunks, 50-token overlap, batch_size=64
- 3040 chunks from 250 markdown files in `data/vector_db/` (ChromaDB + FAISS pickle fallback)
- Groq `llama-3.1-8b-instant` primary; Ollama `llama3.1:8b` fallback; `_get_api_key()` checks `os.environ` first, then `st.secrets`
- LangChain used only for `RecursiveCharacterTextSplitter` — no agent overhead
- Re-ingest: `python src/rag/ingest.py` (idempotent — skips existing chunk IDs)

## Key Constraints

- **Groq API key** → Streamlit secret only. Never commit. Key name: `GROQ_API_KEY`
- `random_state=42` everywhere. Call `src.utils.set_seeds()` at notebook/script entry before any RNG use
- **Bias audit results (Phase 3):**
  - Lead model: `first_gen_flag` flagged (DI=0.74). Mitigation: lower threshold to 0.47 for `first_gen=1`. Must apply at inference time.
  - Yield model: `income_band` flagged — causal (aid package drives enrollment), not proxy bias. No mitigation applied.
  - Synthetic generator intentionally encodes `-0.15` first_gen penalty in lead logit (`src/data_synthesis.py:157`) — do not remove
- Synthetic yield calibration uses bisection (`_calibrated_intercept` in `src/data_synthesis.py`). Per-segment targets must hold within ±2pp; verify with `yield_rate_by_segment(df)` after any logit edit
- Notebook numbering: `01_data_acquisition`, `02_demographic_forecasting`, `03_lead_scoring`, `04_yield_prediction`, `05_explainability`, `06_bias_audit`, `07_rag_evaluation`

## Phase 3 — Actual Model Results

| Model | ROC-AUC | PR-AUC | F1 | Notes |
|-------|---------|--------|-----|-------|
| Lead XGBoost | 0.660 | 0.446 | 0.239 | threshold=0.50; high-precision operating point |
| Lead LR | 0.665 | 0.450 | 0.231 | linear DGP favors LR slightly |
| Yield XGBoost | 0.688 | 0.535 | 0.432 | |
| Yield LR | **0.713** | **0.570** | **0.466** | LR wins — yield logit is linear by construction |

LR outperforming XGBoost on yield is expected (synthetic logit-based DGP). In production with real data XGBoost's non-linear capacity would dominate.

## Gotchas hit during Phase 1

- **NCES URL year-suffix varies by table.** Table 303.10 only at `d21/xls/` — 219.10 and 302.10 at `d22/xls/`. Mapping in `NCES_TABLES` (`src/data_acquisition.py`) is final
- **IPEDS `EF{YEAR}.zip` does not exist as single file.** Use `EFFY{YEAR}.zip` (12-month enrollment by race). See `IPEDS_FILES_TEMPLATE`
- **Python's stdlib `urllib.robotparser` mishandles ASU's robots.txt** — `Allow:` lines with `$` anchors return False for everything. Use `_load_disallow_rules` in `src/data_acquisition.py`, not stdlib
- **ASU crawler:** always `_normalize_url()` before queueing. `_NOISE_PATH_FRAGMENTS` filters CDN/files paths. Markdown <200 chars is dropped

## Data Sources

| Path | Source | Notes |
|------|--------|-------|
| `data/raw/nces/` | NCES Digest tables 219.10, 303.10, 302.10 | `.xls` — read via `pd.read_excel(..., header=None)` and slice manually |
| `data/raw/ipeds/` | IPEDS ADM, IC, EFFY FY 2018–2022 | Zips gitignored; re-fetch via `fetch_ipeds()` |
| `data/synthetic/applicants.csv` | Generated | 50k rows, yield calibrated to IPEDS, `first_gen` penalty planted |
| `data/corpus/` | ASU admissions site scrape | Markdown; institution name redacted in deployment |
| `data/vector_db/` | ChromaDB persistent store | Not committed to git |
| `data/output/` | Model outputs | `forecasts.csv`, `lead_predictions.csv`, `yield_predictions.csv`, PNG plots |

## Dashboard Pages

`streamlit_app.py` (Overview) + `pages/`:
- **Overview** ✅ — 4 KPI cards (forecast/leads/yield/chatbot deflection), enrollment funnel chart, demographic shift bar chart (race % change 2023→2035), 5 module quick-links, system summary expander
- `1_Forecasting.py` ✅ — Prophet/SARIMA interactive chart, CI toggle, race/ethnicity breakdown
- `2_Lead_Scoring.py` ✅ — filterable lead table, score distribution, per-lead SHAP waterfall
- `3_Yield_Prediction.py` ✅ — yield distribution, predicted vs actual by segment, per-student SHAP
- `4_Chatbot.py` ✅ — streaming chat, source citations, sample chips, latency display
- `5_Model_Performance.py` ✅ — 4 tabs: model metrics, bias audit, calibration plots, chatbot eval (latency + retrieval by category, per-question latency scatter)

**Lead score threshold note:** Max lead_score ≈ 70 in synthetic data. "High-priority" threshold on overview = 50 (top ~8% of leads, score ≥ 50/100). Bias audit page uses 0.50 probability threshold (score/100 ≥ 0.5).

## Deployment

Streamlit Cloud (share.streamlit.io), Python 3.11. Add `GROQ_API_KEY` as app secret. If ChromaDB persistence fails in Cloud, fall back to FAISS pickle loaded at startup.

## Phase 4 — RAG Eval Results

| Metric | Result | Target |
|--------|--------|--------|
| Trick refusal rate | 5/5 (100%) | 100% |
| Mean latency | 5.20s | <3s |
| p95 latency | 9.47s | — |
| Mean top-chunk cosine | 0.656 | — |
| Questions score > 0.5 | 26/30 | — |

Latency exceeds 3s target on Groq free tier. Q27 auto-classified as non-refusal but answer correctly deferred ("speak with an admissions advisor"). True refusal = 5/5. Raw results in `data/output/rag_eval_raw.csv`; manual correctness scoring pending.

## See Also

- `docs/data_dictionary.md` — every column in every CSV/zip, source and license
- `docs/model_cards/lead_scoring.md`, `docs/model_cards/yield_prediction.md` — full bias audit results
- `enrollment_implementation_plan.md` — strategic 6-phase plan with deliverables
