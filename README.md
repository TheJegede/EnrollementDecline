# Enrollment Decline & Demographic Shift AI System

> **Three integrated AI modules** helping higher education institutions navigate structural enrollment decline driven by demographic shifts and the 2025 demographic cliff.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Live Dashboard →](https://share.streamlit.io)** *(deploy instructions below)*

---

## The Problem

The US higher education system faces a structural enrollment crisis. The number of 18-year-olds peaked in 2025 and will decline for a decade — the "demographic cliff" institutions have been anticipating since 2012. At the same time, the composition of the applicant pool is shifting rapidly: White enrollment is projected to fall 38% by 2035 while Hispanic, Asian, and multiracial enrollment grows. Institutions that recruited the same way in 2025 as they did in 2015 will see class sizes shrink.

This project builds a proof-of-concept AI system demonstrating how enrollment management teams can respond: forecast demand by segment, prioritize outreach to the highest-yield leads, and deploy a 24/7 chatbot to handle admissions inquiries at scale.

---

## Architecture

```
NCES / IPEDS data ──► src/forecasting.py ────────────────────────────┐
                      SARIMA + Prophet → 2035 projections            │
                                                                      ▼
data/synthetic/    ──► src/lead_model.py  ──► models/lead_scoring.pkl ► streamlit_app.py
applicants.csv     ──► src/yield_model.py ──► models/yield_pred.pkl  ► pages/ (5 views)
                           └── SHAP explanations ──► *_predictions.csv
                                                                      ▲
data/corpus/       ──► src/rag/ingest.py  ──► data/vector_db/        │
(250 ASU pages)        ChromaDB 3040 chunks                          │
                   ──► src/rag/retrieval.py + generation.py (Groq) ──┘
```

---

## Modules

### Module 1 — Demographic Forecasting

Projects US college enrollment and high school graduate populations to 2035 by race/ethnicity and census region.

| | |
|--|--|
| **Models** | SARIMA (winner) vs Prophet (benchmark) |
| **Training** | 1980–2017 NCES data |
| **Hold-out** | 2018–2023 |
| **MAPE** | **3.9%** (SARIMA) · Prophet over-predicts (extrapolates 1980–2017 growth) |
| **Output** | `data/output/forecasts.csv` — projections with 80% + 95% CI to 2035 |
| **Key finding** | White enrollment −38%, Hispanic +8%, Asian +16%, TwoOrMore +23% by 2035 |

### Module 2 — Lead & Yield Prediction

Two XGBoost classifiers covering the two highest-value decision points in the enrollment funnel.

| | Lead Model | Yield Model |
|--|--|--|
| **Target** | `inquired_to_applied` | `admit_to_enroll` |
| **Training set** | 50k synthetic applicants | 8.9k admitted students |
| **Class balance** | `scale_pos_weight` | SMOTE (train only) |
| **Best algorithm** | Logistic Reg. (ROC-AUC 0.665) | Logistic Reg. (ROC-AUC **0.713**) |
| **XGBoost** | ROC-AUC 0.660, F1 0.239 | ROC-AUC 0.688, F1 0.432 |
| **Calibration** | Platt scaling on held-out val set | Platt scaling on held-out val set |
| **Explainability** | Top-3 SHAP features per prediction | Top-3 SHAP features per prediction |

*LR outperforms XGBoost on both models — expected: synthetic data has a linear logit DGP. In production with real non-linear feature interactions, XGBoost would likely dominate.*

**Bias audit (mandatory):**
- Lead model: `first_gen_flag` flagged — DI ratio 0.74 (below 0.80 threshold). Mitigated by lowering decision threshold 0.50 → 0.47 for first-gen applicants. Parity gap: 2.6pp → 0.1pp.
- Yield model: `income_band` flagged as causal (higher aid package → higher enrollment probability), not proxy bias. No mitigation applied — removing income_band would degrade model utility.

### Module 3 — RAG Admissions Chatbot

Retrieval-augmented chatbot answering admissions FAQs grounded in an institutional knowledge corpus.

| | |
|--|--|
| **Embedding** | `all-MiniLM-L6-v2` (384-dim, local) |
| **Chunks** | 3,040 from 250 markdown pages (500-token, 50-token overlap) |
| **Vector store** | ChromaDB (persistent) + FAISS pickle fallback |
| **LLM** | Groq `llama-3.1-8b-instant` (free tier) |
| **Trick refusal** | **5/5 (100%)** — never hallucinated outside corpus |
| **Mean latency** | 5.2s (Groq free tier variable; p95 = 9.5s) |
| **Retrieval quality** | 26/30 questions cosine score > 0.50; contact queries score 0.816 |

---

## Simulated Business Impact

At a 5,000-applicant institution, improving yield prediction accuracy by 3 percentage points translates to approximately **150 additional enrollments per year**. At $30,000 average tuition, that represents **$4.5M in annual tuition revenue** — before accounting for room, board, and multi-year retention effects.

The lead scoring model's ability to identify the top 8% of leads (score ≥ 50) allows recruitment staff to concentrate outreach effort on the ~4,000 highest-conversion prospects rather than blanketing all 50,000 leads uniformly.

---

## Stack

| Category | Tool |
|----------|------|
| Dashboard | Streamlit 1.39 |
| Forecasting | Prophet · statsmodels SARIMA |
| ML modeling | scikit-learn · XGBoost · imbalanced-learn (SMOTE) |
| Explainability | SHAP (TreeExplainer) |
| RAG pipeline | sentence-transformers · ChromaDB · LangChain text splitter · Groq API |
| Visualization | Plotly · matplotlib · seaborn |
| Data | NCES Digest tables 219.10 / 303.10 / 302.10 · IPEDS ADM/IC/EFFY 2018–2022 |

---

## Setup

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/enrollment-ai-system.git
cd enrollment-ai-system
pip install -r requirements.txt

# 2. Set Groq API key
# Create .streamlit/secrets.toml:
echo 'GROQ_API_KEY = "gsk_..."' > .streamlit/secrets.toml

# 3. Run dashboard
streamlit run streamlit_app.py
```

**Rebuild vector index** (if `data/vector_db/` is missing):
```bash
python src/rag/ingest.py
```

**Retrain ML models:**
```bash
jupyter nbconvert --to notebook --execute notebooks/03_lead_scoring.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/04_yield_prediction.ipynb --inplace
```

---

## Repo Layout

```
data/
  raw/nces/          NCES Digest .xls tables (gitignored large IPEDS zips)
  synthetic/         applicants.csv — 50k rows, IPEDS-calibrated yield rates
  corpus/            250 ASU admissions pages as markdown
  vector_db/         ChromaDB store (rebuild with src/rag/ingest.py)
  output/            forecasts.csv, *_predictions.csv, calibration PNGs, eval CSV

src/
  forecasting.py     Prophet + SARIMA training + forecasting
  lead_model.py      Lead scoring training + inference
  yield_model.py     Yield prediction training + inference
  data_synthesis.py  Synthetic applicant generator (calibrated to IPEDS)
  rag/               embedding · ingest · retrieval · generation

notebooks/
  01_data_acquisition.ipynb
  02_demographic_forecasting.ipynb
  03_lead_scoring.ipynb
  04_yield_prediction.ipynb
  05_explainability.ipynb
  06_bias_audit.ipynb
  07_rag_evaluation.ipynb

models/              lead_scoring.pkl · yield_prediction.pkl
pages/               1_Forecasting · 2_Lead_Scoring · 3_Yield_Prediction
                     4_Chatbot · 5_Model_Performance
docs/
  data_dictionary.md
  model_cards/       lead_scoring · yield_prediction · forecasting
  chatbot_evaluation.md
  dashboard_walkthrough.md
```

---

## Disclosures

- All demographic data is public-domain (NCES, IPEDS — US government works).
- Applicant data is **fully synthetic** — no real PII. Generation logic in `src/data_synthesis.py` and `notebooks/01_data_acquisition.ipynb`.
- Chatbot corpus is a public scrape of Arizona State University's admissions website for portfolio demonstration purposes. Institution name is redacted in public deployment. No internal or restricted data used.
- Bias audit is not optional — both ML models must pass fairness checks before dashboard display.

## License

MIT
