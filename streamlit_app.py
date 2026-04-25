"""Enrollment Decline & Demographic Shift AI System — entrypoint.

Page 1: Executive Overview. Other pages live in pages/.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils import OUTPUT_DIR

st.set_page_config(
    page_title="Enrollment AI System",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Enrollment Decline & Demographic Shift AI System")
st.caption(
    "Portfolio POC — three integrated AI modules for higher education enrollment management"
)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def _load_all():
    fc = lp = yp = ev = None
    try:
        fc = pd.read_csv(OUTPUT_DIR / "forecasts.csv")
    except Exception:
        pass
    try:
        lp = pd.read_csv(OUTPUT_DIR / "lead_predictions.csv")
    except Exception:
        pass
    try:
        yp = pd.read_csv(OUTPUT_DIR / "yield_predictions.csv")
    except Exception:
        pass
    try:
        ev = pd.read_csv(OUTPUT_DIR / "rag_eval_raw.csv")
    except Exception:
        pass
    return fc, lp, yp, ev


forecasts, lead_pred, yield_pred, rag_eval = _load_all()

# ── KPI row ───────────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

# KPI 1 — Enrollment forecast 2030 vs last confirmed historical actual (≤2023)
if forecasts is not None:
    enroll = forecasts[forecasts["series"] == "total_enrollment"]
    sarima_2030 = enroll[enroll["year"] == 2030]["sarima_yhat"].values
    hist = enroll[(enroll["year"] <= 2023) & enroll["actual"].notna()]
    last_val = hist["actual"].iloc[-1] if len(hist) else None
    last_yr = int(hist["year"].iloc[-1]) if len(hist) else None

    if last_val and sarima_2030.size:
        pct = (sarima_2030[0] - last_val) / last_val * 100
        col1.metric(
            "US Enrollment Forecast 2030",
            f"{sarima_2030[0] / 1e6:.1f}M",
            delta=f"{pct:+.1f}% vs {last_yr}",
        )
    else:
        col1.metric("US Enrollment Forecast 2030", "—")
else:
    col1.metric("US Enrollment Forecast 2030", "—")

# KPI 2 — High-priority leads (top quartile ≈ score ≥ 50)
if lead_pred is not None:
    high_pri = (lead_pred["lead_score"] >= 50).sum()
    total = len(lead_pred)
    col2.metric(
        "High-Priority Leads (score ≥ 50)",
        f"{high_pri:,}",
        delta=f"{high_pri / total:.1%} of {total:,} leads",
    )
else:
    col2.metric("High-Priority Leads", "—")

# KPI 3 — Predicted enrolling class
if yield_pred is not None:
    pred_class = (yield_pred["yield_probability"] >= 60).sum()
    admits = len(yield_pred)
    col3.metric(
        "Predicted Enrollments (prob ≥ 60%)",
        f"{pred_class:,}",
        delta=f"{pred_class / admits:.1%} of {admits:,} admits",
    )
else:
    col3.metric("Predicted Enrolling Class", "—")

# KPI 4 — Chatbot deflection rate
if rag_eval is not None:
    handled = (rag_eval["auto_refused"] == False).sum()
    total_q = len(rag_eval)
    col4.metric(
        "Chatbot Deflection Rate",
        f"{handled / total_q:.0%}",
        delta=f"{handled}/{total_q} queries handled",
    )
else:
    col4.metric("Chatbot Deflection Rate", "—")

st.divider()

# ── Main charts ───────────────────────────────────────────────────────────────

left, right = st.columns([1, 1], gap="large")

# Left — Enrollment funnel
with left:
    st.subheader("Enrollment Funnel")
    st.caption("Synthetic 50k-applicant cohort — all stages")

    if lead_pred is not None and yield_pred is not None:
        n_leads = len(lead_pred)
        n_applied = int(lead_pred["inquired_to_applied"].sum())
        n_admits = len(yield_pred)
        n_enrolled = int(yield_pred["admit_to_enroll"].sum())

        funnel_fig = go.Figure(go.Funnel(
            y=["Leads", "Applicants", "Admits", "Enrolled"],
            x=[n_leads, n_applied, n_admits, n_enrolled],
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=["#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa"]),
            connector=dict(line=dict(color="#cbd5e1", width=2)),
        ))
        funnel_fig.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13),
        )
        st.plotly_chart(funnel_fig, use_container_width=True)

        fcols = st.columns(3)
        fcols[0].metric("Inquiry→Apply rate", f"{n_applied / n_leads:.1%}")
        fcols[1].metric("Apply→Admit rate", f"{n_admits / n_applied:.1%}")
        fcols[2].metric("Admit→Enroll rate", f"{n_enrolled / n_admits:.1%}")
    else:
        st.info("Run notebooks 03 and 04 to populate funnel data.")

# Right — Demographic shift
with right:
    st.subheader("Demographic Enrollment Shift")
    st.caption("SARIMA projected change: last actual → 2035 (major groups)")

    if forecasts is not None:
        race = forecasts[forecasts["series"] == "race_enrollment"].copy()
        GROUPS = ["White", "Black", "Hispanic", "Asian", "TwoOrMore"]
        race = race[race["segment"].isin(GROUPS)]

        last_actual_yr = int(race.dropna(subset=["actual"])["year"].max())
        proj_yr = 2035

        base = (
            race[race["year"] == last_actual_yr]
            .set_index("segment")["actual"]
        )
        proj = (
            race[race["year"] == proj_yr]
            .set_index("segment")["sarima_yhat"]
        )
        shift = ((proj - base) / base * 100).dropna().reindex(GROUPS).dropna()
        shift_df = shift.reset_index()
        shift_df.columns = ["segment", "pct_change"]
        shift_df = shift_df.sort_values("pct_change", ascending=True)

        colors = ["#dc2626" if v < 0 else "#16a34a" for v in shift_df["pct_change"]]
        shift_fig = go.Figure(go.Bar(
            x=shift_df["pct_change"],
            y=shift_df["segment"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in shift_df["pct_change"]],
            textposition="outside",
        ))
        shift_fig.update_layout(
            height=340,
            margin=dict(l=10, r=60, t=10, b=30),
            xaxis=dict(title=f"% change {last_actual_yr}→{proj_yr}", zeroline=True),
            yaxis=dict(title=""),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13),
        )
        st.plotly_chart(shift_fig, use_container_width=True)
        st.caption(
            f"White enrollment projected to decline; Hispanic, Asian, and "
            f"Two-or-More race groups projected to grow. "
            f"Base year: {last_actual_yr}."
        )
    else:
        st.info("Run notebook 02 to populate forecast data.")

st.divider()

# ── Module quick links ────────────────────────────────────────────────────────

st.subheader("Modules")
m1, m2, m3, m4, m5 = st.columns(5)

def _module_card(col, icon, title, description, page_label):
    col.markdown(
        f"""
**{icon} {title}**

{description}

*→ [{page_label}]({page_label})*
""",
        unsafe_allow_html=False,
    )

with m1:
    st.markdown("**📈 Forecasting**")
    st.markdown(
        "SARIMA + Prophet projections of US enrollment to 2035. "
        "Race/ethnicity breakdowns, confidence intervals."
    )
    st.page_link("pages/1_Forecasting.py", label="Open Forecasting →")

with m2:
    st.markdown("**🎯 Lead Scoring**")
    st.markdown(
        "XGBoost classifier ranking 50k leads by inquiry-to-application "
        "probability. Top-3 SHAP per lead."
    )
    st.page_link("pages/2_Lead_Scoring.py", label="Open Lead Scoring →")

with m3:
    st.markdown("**🎓 Yield Prediction**")
    st.markdown(
        "XGBoost yield model on 8.9k admits. SMOTE + Platt calibration. "
        "Class size forecasting."
    )
    st.page_link("pages/3_Yield_Prediction.py", label="Open Yield Prediction →")

with m4:
    st.markdown("**💬 Chatbot**")
    st.markdown(
        "RAG chatbot over public admissions corpus. Groq + ChromaDB. "
        "5/5 trick question refusal."
    )
    st.page_link("pages/4_Chatbot.py", label="Open Chatbot →")

with m5:
    st.markdown("**⚖️ Model Performance**")
    st.markdown(
        "ROC-AUC, F1, calibration curves. Demographic parity + disparate "
        "impact audit across 4 protected groups."
    )
    st.page_link("pages/5_Model_Performance.py", label="Open Performance →")

st.divider()

# ── System summary ────────────────────────────────────────────────────────────

with st.expander("System overview", expanded=False):
    st.markdown("""
| Module | Model | Key metric |
|--------|-------|-----------|
| Demographic Forecasting | SARIMA | MAPE = 3.9% on 2018–2023 hold-out |
| Lead Scoring | XGBoost + LR | ROC-AUC = 0.665 (LR), F1 = 0.239 |
| Yield Prediction | XGBoost + LR | ROC-AUC = 0.713 (LR), F1 = 0.466 |
| RAG Chatbot | Groq llama-3.1-8b | Trick refusal 5/5 · mean latency 5.2s |

**Data:** NCES Digest tables + IPEDS 2018–2022 + 50k synthetic applicants + public admissions corpus (250 pages, 3040 chunks).

**Bias audit:** Lead model `first_gen_flag` DI = 0.74 → threshold adjusted 0.50 → 0.47 for first-gen applicants (parity gap 2.6pp → 0.1pp). Yield model `income_band` flagged as causal (not proxy bias) — no mitigation applied.

**Stack:** Python · Streamlit · XGBoost · sentence-transformers · ChromaDB · Groq · Prophet · SARIMA
""")
