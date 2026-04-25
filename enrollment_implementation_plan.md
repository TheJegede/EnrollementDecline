  
**IMPLEMENTATION PLAN**

**Enrollment Decline & Demographic Shift AI System**

*Proof of Concept — Independent Portfolio Project*

| Timeline | 10–12 Weeks |
| :---- | :---- |
| **Budget** | $0 (open data \+ free tiers) |
| **Modules** | Forecasting \+ ML \+ RAG Chatbot |
| **Stack** | Python, Streamlit, Groq API, ChromaDB |
| **Deliverable** | Live integrated dashboard \+ chatbot |

# **Project Overview**

This project addresses the second-most-severe pain point in US higher education: structural enrollment decline driven by demographic shifts and an increased reliance on adult learners. The 2025 demographic cliff has materialized, traditional 18–22 applicant pools are shrinking, and institutions are competing harder than ever for fewer students.

The system is built as an independent proof of concept with no institutional access, zero budget, and an open-source-only stack. It demonstrates three integrated AI capabilities: demographic forecasting at the national level, machine learning models scoring leads and predicting yield across the funnel, and an NLP chatbot that handles 24/7 admissions inquiries. All three feed a single unified dashboard.

*Three integrated modules in one portfolio project is significantly more complex than the prior retention POC. Timeline reflects this: 10–12 weeks vs 7–8. Each module is independently usable but most powerful when integrated.*

## **System Architecture**

The three modules align to different stages of the enrollment funnel:

* Module 1 — Demographic Forecasting: top-of-funnel strategic planning. Uses NCES and IPEDS data to project future high school graduate populations by region and demographic segment.

* Module 2 — Lead and Yield Prediction: mid-funnel operational scoring. Two ML classifiers — one ranks prospective leads by enrollment probability, one predicts yield (admitted student enrollment likelihood).

* Module 3 — NLP Chatbot: always-on prospect support. Retrieval-augmented generation answering admissions FAQs, application questions, deadline lookups, and program inquiries.

* Unified Dashboard: single Streamlit app with four pages — one per module plus an integrated overview. Deployed publicly to Streamlit Cloud.

## **Technology Stack**

| Category | Tool / Service | Purpose |
| :---- | :---- | :---- |
| Data manipulation | pandas, numpy | All data processing |
| Forecasting | Prophet, statsmodels | Time-series demographic projection |
| ML modeling | scikit-learn, XGBoost | Lead scoring \+ yield classifiers |
| Class imbalance | imbalanced-learn | SMOTE for minority class handling |
| Explainability | SHAP | Per-applicant feature contribution |
| LLM API | Groq (free tier) | Fast LLM inference for chatbot |
| Vector DB | ChromaDB | Local embedding storage for RAG |
| Embeddings | sentence-transformers | Free local embedding generation |
| Visualization | matplotlib, seaborn, plotly | EDA and dashboard charts |
| Dashboard | Streamlit | Unified UI for all modules |
| Deployment | Streamlit Cloud | Free public hosting |
| Version control | GitHub | Code, notebooks, model artifacts |

| Phase 1  Data Acquisition | Weeks 1–2 |
| :---- | :---: |

## **Objective**

Acquire all three data streams the project needs — demographic time-series, applicant-level data for predictive modeling, and document corpora for the chatbot knowledge base. All data must be public, openly licensed, and require no institutional credentials.

## **Source 1 — NCES Digest of Education Statistics**

The National Center for Education Statistics publishes annual digest tables containing historical and projected high school graduate counts, college enrollment by demographic segment, and projections through 2031\. This is the primary input for the demographic forecasting module.

| Attribute | Detail |
| :---- | :---- |
| Source URL | https://nces.ed.gov/programs/digest/ |
| Specific tables | Table 219.10 (high school graduates by state), Table 303.10 (enrollment projections), Table 302.10 (enrollment by age and attendance) |
| License | Public domain — US government work, no restrictions |
| Format | Excel (.xls) and CSV downloads |
| Coverage | 1970–present, with NCES projections to 2031 |
| Granularity | National and state-level, by race/ethnicity and age band |

## **Source 2 — IPEDS (Integrated Postsecondary Education Data System)**

IPEDS provides institution-level enrollment data for every Title IV-eligible college and university in the US. Used to model trends across institution types and as feature engineering input for the predictive models.

| Attribute | Detail |
| :---- | :---- |
| Source URL | https://nces.ed.gov/ipeds/use-the-data |
| Access | Free download via IPEDS Data Center — no login required for public files |
| Key surveys | EF (Fall Enrollment), ADM (Admissions), IC (Institutional Characteristics) |
| License | Public domain — US government work |
| Format | CSV exports per survey-year combination |
| Coverage | All \~6,000 Title IV institutions, 1980–present |
| Key fields | Total applications, admits, enrollees, demographic breakdowns, yield rates by institution |

## **Source 3 — Synthetic Applicant Dataset**

Real applicant-level data (with personal characteristics, behavior, and enrollment outcomes) is not publicly available — and would not be ethical to use even if it were. The predictive modeling module therefore uses a synthetic dataset generated in Python with realistic distributions calibrated to published IPEDS yield rates.

* Sample size: 50,000 synthetic applicants across 5 simulated institution types

* Features: high school GPA, SAT/ACT score, distance from institution, income band, first-generation flag, race/ethnicity, intended major, application date relative to deadline, campus visit flag, financial aid inquiry flag, email engagement score

* Lead label: binary 'inquired\_to\_applied' — did this lead become an applicant?

* Yield label: binary 'admit\_to\_enroll' — did this admit choose this institution?

* Calibration: yield rates per institution segment match published IPEDS averages (R1: \~33%, regional state: \~22%, private liberal arts: \~28%)

* Bias-relevant features included to enable a meaningful fairness audit

* Reproducibility: random\_state=42, generation logic fully documented in 01\_data\_synthesis.ipynb

*Synthetic data is the standard portfolio approach for this exact pain point. Document the generation logic transparently — it shows methodological awareness, not a shortcut.*

## **Source 4 — Chatbot Knowledge Corpus**

The RAG chatbot needs a document corpus to retrieve from. Use a representative public university's admissions website as the source — content is publicly accessible and the structure mirrors what a real institution's chatbot would index.

| Attribute | Detail |
| :---- | :---- |
| Source | Public admissions sites (e.g. Arizona State, Penn State, Ohio State) — pick one as the canonical corpus |
| Method | Web scraping with BeautifulSoup or trafilatura — robots.txt respected |
| Sections | Application requirements, deadlines, financial aid, housing, programs, FAQ, contact information |
| Volume | \~200–400 page sections, chunked into \~1,500 retrievable passages |
| Storage | Markdown files in data/corpus/ — committed to repo |
| Disclosure | README explicitly notes this is a public-data scrape for portfolio purposes; institution name redacted in public deployment |

## **Phase 1 Deliverables**

* data/raw/nces/ — downloaded NCES tables

* data/raw/ipeds/ — IPEDS survey CSVs

* data/synthetic/applicants.csv — 50,000-row synthetic applicant set

* data/corpus/ — markdown knowledge corpus

* notebooks/01\_data\_acquisition.ipynb — reproducible download \+ synthesis

* docs/data\_dictionary.md — every field, source, and license documented

| Phase 2  Demographic Forecasting Module | Weeks 2–4 |
| :---- | :---: |

## **Objective**

Build a time-series forecasting model that projects US high school graduate populations and college enrollment trends through 2035\. The forecast directly addresses the strategic question: 'Which student segments are growing, which are shrinking, and where should the institution focus recruitment effort?'

## **Forecasting Approach**

Two modeling approaches will be compared, with the better-performing model selected for the dashboard.

| Model | Why use it | Role |
| :---- | :---- | :---- |
| Prophet (Meta) | Handles seasonality, trend changepoints, missing data; minimal tuning required | Primary |
| SARIMA (statsmodels) | Classical time-series benchmark; well-documented; produces interpretable parameters | Benchmark |

* Train/test split: hold out 2018–2023 as test set, train on 1980–2017

* Metrics: MAPE (mean absolute percentage error), RMSE, and direction accuracy on year-over-year change

* Forecast horizon: project to 2035 with 80% and 95% confidence intervals

* Segmentation: forecast separately for total population, by race/ethnicity (5 categories), by region (4 census regions), and by age band (18–24 traditional, 25–34 adult)

## **Key Outputs**

* Decade-level projection: total US high school graduates 2024–2035

* Demographic shift heatmap: which segments grow, which decline, by region

* Adult learner trajectory: 25–34 enrollment trend — the segment institutions are pivoting toward

* Confidence bands visualized on every projection — never present a forecast as a point estimate without uncertainty

## **Phase 2 Deliverables**

* src/forecasting.py — reusable forecasting module

* notebooks/02\_demographic\_forecasting.ipynb — model comparison, validation, projection plots

* data/output/forecasts.csv — projection table 2024–2035 with confidence intervals

* Streamlit dashboard page 1 (forecasting view) wired and functional

| Phase 3  Lead Scoring & Yield Prediction Module | Weeks 4–7 |
| :---- | :---: |

## **Objective**

Build two complementary classifiers covering the two most operational decision points in the funnel: which leads should the recruitment team pursue, and which admitted students will actually enroll. Both models output interpretable probability scores with SHAP-explained feature contributions.

## **Model 1 — Lead Scoring**

Predicts the probability that an inquiry/lead converts into a completed application. Used to prioritize recruitment outreach effort across thousands of leads.

* Target: binary inquired\_to\_applied (0/1)

* Features: source channel, demographic data, intended major, distance, campus visit flag, email engagement score, days since first inquiry, financial aid inquiry flag

* Algorithms: logistic regression (baseline), XGBoost (challenger), Random Forest (secondary)

* Class balance: \~30% inquiry-to-app rate in synthetic data — moderate imbalance, handled with class weighting

* Output: lead\_score (0–100), top-3 SHAP feature contributions per lead

## **Model 2 — Yield Prediction**

Predicts the probability that an admitted student enrolls. Used to prioritize yield-stage outreach and to forecast incoming class size.

* Target: binary admit\_to\_enroll (0/1)

* Features: all lead-scoring features plus admission decision flag, financial aid package amount, scholarship offer flag, days from admit to deposit deadline, peer admit comparison features

* Algorithms: XGBoost as primary (yield is non-linear and feature-interaction-heavy), logistic regression as interpretable baseline

* Class balance: \~25% yield rate — apply SMOTE to training set only

* Output: yield\_probability (0–100), top-3 SHAP contributions, predicted enrollment label

## **Validation Protocol**

* Split: 70% train, 15% validation, 15% test — stratified by target class

* Cross-validation: 5-fold stratified on training set

* Hyperparameter tuning: GridSearchCV on XGBoost (n\_estimators, max\_depth, learning\_rate, subsample, min\_child\_weight)

* Metrics: Precision, Recall, F1 (positive class), ROC-AUC, Precision-Recall AUC, Confusion Matrix

* Calibration: Platt scaling so probabilities are interpretable as true likelihoods

## **Bias Audit**

Both models touch admissions decisions — the highest-stakes domain in higher ed AI. Bias audit is mandatory and visible.

| Check | Method |
| :---- | :---- |
| Demographic parity | Compare positive prediction rates across race/ethnicity, gender, first-gen status, income band — flag if any group differs by more than 5pp |
| Equalized odds | Compare TPR (recall) across groups — model must catch positive outcomes equally well |
| Disparate impact ratio | Ratio of positive prediction rates between groups — flag if outside the 0.8–1.25 band |
| Proxy feature audit | Check correlation of every feature with protected attributes — distance, ZIP-derived features, and HS GPA can act as proxies |
| Mitigation if bias found | Threshold adjustment per group OR re-weighting OR feature exclusion — document the decision and the before/after metrics |

## **Phase 3 Deliverables**

* src/lead\_model.py and src/yield\_model.py — training and prediction modules

* models/lead\_scoring.pkl and models/yield\_prediction.pkl

* notebooks/03\_lead\_scoring.ipynb and 04\_yield\_prediction.ipynb

* notebooks/05\_explainability.ipynb — SHAP analysis for both models

* notebooks/06\_bias\_audit.ipynb — fairness audit with documented mitigation

* docs/model\_cards/ — standardized model cards for each model

* data/output/lead\_predictions.csv and yield\_predictions.csv

| Phase 4  RAG Chatbot Module | Weeks 6–9 |
| :---- | :---: |

## **Objective**

Build a retrieval-augmented chatbot that answers admissions inquiries using the institutional knowledge corpus. The chatbot demonstrates that prospects can get accurate, 24/7 answers without straining admissions staff. Key design constraint: must run entirely on free-tier services.

## **RAG Architecture**

Retrieval-Augmented Generation works in two stages. First, the user's question is converted to an embedding and used to retrieve the most relevant chunks from the knowledge base. Then those chunks are injected into the LLM prompt as context, and the LLM generates an answer grounded in the retrieved content. This approach prevents hallucination — the LLM can only cite information that was actually retrieved from authoritative sources.

## **Component Stack**

| Component | Tool | Why |
| :---- | :---- | :---- |
| Embedding model | sentence-transformers (all-MiniLM-L6-v2) | Free, local, fast, 384-dim vectors |
| Vector store | ChromaDB (local persistent) | Free, embedded, no server required |
| Document chunking | LangChain RecursiveCharacterTextSplitter | Standard 500-token chunks with 50-token overlap |
| LLM inference | Groq API (Llama 3.1 8B or Mixtral) | Free tier, sub-second response time |
| Orchestration | Direct Python (no LangChain agent overhead) | Simpler, more debuggable for portfolio |
| Interface | Streamlit chat component | Native chat UI, no extra deps |

*Groq's free tier provides generous rate limits (\~30 requests/minute) and very fast inference. API key is stored as a Streamlit secret — never committed to the repo.*

## **Knowledge Base Pipeline**

* Ingest: load all markdown files from data/corpus/

* Chunk: split each document into \~500-token passages with 50-token overlap

* Embed: generate vector embedding for each chunk using sentence-transformers

* Store: persist to ChromaDB at data/vector\_db/

* Re-ingestion script: update knowledge base when corpus files change

## **Query Pipeline**

* Receive user question via Streamlit chat input

* Embed question with the same sentence-transformer model

* Retrieve top-5 most similar chunks from ChromaDB

* Construct prompt: system instructions \+ retrieved chunks \+ user question

* Send to Groq API, stream response back to chat UI

* Display answer with source citations (which chunks were used)

## **System Prompt Design**

The system prompt is critical for grounding behavior. Three rules enforced:

* Answer only from provided context — refuse if context does not contain the answer

* Cite source sections explicitly — every claim must reference a retrieved chunk

* Defer to human advisors for complex cases — application status, individual financial aid questions, deadline exceptions

## **Evaluation**

* Build a test set of 30 admissions questions with known correct answers

* Manual scoring: answer correctness, source citation accuracy, refusal appropriateness

* Latency benchmark: average end-to-end response time under 3 seconds

* Hallucination check: 5 'trick' questions where the answer is not in the corpus — chatbot must refuse, not invent

## **Phase 4 Deliverables**

* src/rag/ — embedding, retrieval, generation modules

* data/vector\_db/ — populated ChromaDB

* notebooks/07\_rag\_evaluation.ipynb — test set scoring

* Streamlit dashboard chatbot page wired to live Groq API

* docs/chatbot\_evaluation.md — test results and known limitations

| Phase 5  Unified Dashboard & Integration | Weeks 8–11 |
| :---- | :---: |

## **Objective**

Combine all three modules into a single integrated Streamlit application. The dashboard simulates what an enrollment management team would actually use day-to-day — a single pane of glass spanning strategic forecasting, operational scoring, and prospect-facing chat support.

## **Dashboard Page Structure**

### **Page 1 — Executive Overview**

* KPI cards: 5-year graduate forecast trend, current funnel volume, predicted yield class size, chatbot deflection rate

* Demographic shift summary chart — which segments are growing

* Funnel visualization: leads → applicants → admits → enrolls with percentages at each stage

* Quick links to each module page

### **Page 2 — Demographic Forecasting**

* Interactive projection chart with selectable segments (national, region, race/ethnicity, age band)

* Confidence band toggle — 80% and 95% intervals

* Decade comparison: 2015–2025 actual vs 2025–2035 projected

* Export forecast data button

### **Page 3 — Lead Scoring**

* Searchable, filterable lead table sorted by score descending

* Score distribution histogram

* Click-through to per-lead detail with SHAP waterfall plot

* Threshold slider to focus on highest-priority leads

### **Page 4 — Yield Prediction**

* Admitted student list with predicted enrollment probability

* Predicted class composition by major, region, demographic

* Per-student detail: yield probability, key drivers, recommended outreach action

* Class size forecast with uncertainty range

### **Page 5 — Admissions Chatbot**

* Streamlit chat interface with message history

* Source citations displayed alongside each response

* Sample question prompts shown above input

* Latency indicator and 'powered by Groq \+ Llama 3.1' attribution

### **Page 6 — Model Performance & Bias Audit**

* Performance metrics for both ML models (Precision, Recall, F1, ROC-AUC)

* Forecasting model error metrics (MAPE, RMSE)

* Chatbot evaluation summary (correctness rate, refusal rate, latency)

* Bias audit results across demographic groups for both ML models

## **Deployment to Streamlit Cloud**

* Push final repo to GitHub

* Connect repo at share.streamlit.io

* Add Groq API key as Streamlit secret in app settings

* Verify ChromaDB persistence works in Streamlit Cloud environment

* Public URL goes in README and on resume

*Streamlit Cloud free tier handles small ChromaDB instances well. If the corpus grows large, fall back to FAISS index loaded from a pickle. Document the fallback in the README.*

## **Phase 5 Deliverables**

* streamlit\_app.py — full multi-page application

* requirements.txt with all pinned versions

* Live public URL on Streamlit Cloud

* docs/dashboard\_walkthrough.md with annotated screenshots of each page

| Phase 6  Portfolio Packaging | Weeks 11–12 |
| :---- | :---: |

## **Objective**

Polish the project into a recruiter-ready, hiring-manager-ready portfolio asset. The repository must communicate scope, technical depth, and business impact within five minutes of someone landing on the README.

## **GitHub Repository Structure**

| Path | Contents |
| :---- | :---- |
| README.md | Hero section, problem framing, architecture diagram, key results, live link, setup |
| notebooks/01\_data\_acquisition.ipynb | Sources, downloads, synthesis logic |
| notebooks/02\_demographic\_forecasting.ipynb | Prophet vs SARIMA comparison, projections, confidence intervals |
| notebooks/03\_lead\_scoring.ipynb | Lead model training, validation, metrics |
| notebooks/04\_yield\_prediction.ipynb | Yield model training and validation |
| notebooks/05\_explainability.ipynb | SHAP analysis for both models |
| notebooks/06\_bias\_audit.ipynb | Fairness audit \+ mitigation documentation |
| notebooks/07\_rag\_evaluation.ipynb | Chatbot test set scoring |
| src/ | Reusable modules: forecasting, lead\_model, yield\_model, rag/ |
| streamlit\_app.py \+ pages/ | Multi-page dashboard |
| models/ | Serialized .pkl artifacts |
| data/ | raw, synthetic, processed, corpus, vector\_db |
| docs/ | data\_dictionary, model\_cards, dashboard\_walkthrough, chatbot\_evaluation |
| requirements.txt | Pinned dependencies |

## **README Requirements**

* Hero: one-sentence problem, one-sentence solution, live dashboard button, GitHub stars/license badges

* Pain point context: 2–3 sentences on the demographic cliff and why enrollment AI matters

* Architecture diagram: forecasting \+ ML \+ RAG → unified dashboard

* Three-module summary with key metrics for each (forecast MAPE, model F1, chatbot correctness rate)

* Tech stack badges across all three modules

* Simulated business impact: at a 5,000-applicant institution, a 3pp yield prediction improvement \= \~150 additional enrollments \= \~$4.5M tuition revenue annually

* Setup: clone, install, set Groq key, streamlit run

## **Optional Write-up**

A 1,000–1,500 word Medium or LinkedIn article significantly extends portfolio reach. Structure:

* Hook: the demographic cliff in one chart from the forecasting module

* Problem framing: why enrollment is the second-highest-severity higher ed pain point

* How AI maps to each funnel stage (forecasting \+ ML \+ chatbot)

* One technical deep-dive per module — pick the most interesting decision

* What real institutional deployment would require beyond the POC

* Link to GitHub and live dashboard

## **Phase 6 Deliverables**

* Polished GitHub repo with comprehensive README

* All notebooks cleaned of dead code and debug output

* All model cards finalized in docs/

* (Optional) Published Medium or LinkedIn article

# **Master Timeline**

| Phase | Name | Timeline | Key Deliverable |
| :---- | :---- | :---- | :---- |
| 1 | Data Acquisition | Weeks 1–2 | NCES \+ IPEDS \+ synthetic \+ corpus all in repo |
| 2 | Demographic Forecasting | Weeks 2–4 | Prophet/SARIMA model, 2024–2035 projections |
| 3 | Lead \+ Yield Prediction | Weeks 4–7 | Two trained models, SHAP, bias audit |
| 4 | RAG Chatbot | Weeks 6–9 | Working Groq \+ ChromaDB chatbot |
| 5 | Unified Dashboard | Weeks 8–11 | Live multi-page Streamlit app |
| 6 | Portfolio Packaging | Weeks 11–12 | Polished repo, README, optional article |

*Phases overlap intentionally. Phase 3 starts before Phase 2 finishes; Phase 4 starts before Phase 3 finishes; Phase 5 dashboard scaffolding begins as soon as Phase 2 produces its first output. This compresses the total timeline meaningfully without rushing any individual module.*

# **Risk Register**

| Risk | Likelihood | Mitigation |
| :---- | :---- | :---- |
| Synthetic applicant data not realistic enough to produce credible model results | Medium | Calibrate yield rates to published IPEDS averages; document calibration in notebook; have a reviewer sanity-check distributions |
| Forecasting MAPE too high to be portfolio-credible | Low | Hold-out validation on 2018–2023 known data first; Prophet should achieve \<8% MAPE on national-level data |
| Groq free tier rate-limited mid-demo | Low | Cache common queries; fallback to local Ollama if needed; documented in README |
| Chatbot hallucinates outside corpus | Medium | Strict system prompt, refusal test cases in evaluation set, source citation requirement |
| Bias audit reveals significant disparity in lead/yield models | Medium | Apply mitigation, document before/after — finding and correcting bias is stronger evidence than no audit |
| Streamlit Cloud cannot persist ChromaDB across deployments | Medium | Fallback to FAISS pickle loaded at app startup; documented fallback path |
| Three-module scope causes timeline slippage | High | Cut from Phase 6 polish, never from model rigor; Phase 4 chatbot can ship as MVP if needed |

