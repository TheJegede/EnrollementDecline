"""Yield prediction dashboard page — Phase 3."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils import OUTPUT_DIR

st.set_page_config(page_title="Yield Prediction", page_icon=":dart:", layout="wide")
st.title("Yield Prediction")
st.caption("XGBoost · admit_to_enroll · top-3 SHAP per admitted student")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    path = OUTPUT_DIR / "yield_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


df = load_predictions()

if df.empty:
    st.warning(
        "yield_predictions.csv not found. "
        "Run `notebooks/04_yield_prediction.ipynb` first."
    )
    st.stop()

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("Filters")
segments = ["All"] + sorted(df["institution_segment"].unique())
sel_segment = st.sidebar.selectbox("Institution segment", segments)

min_prob = st.sidebar.slider("Minimum yield probability", 0, 100, 0, step=5)

outreach_threshold = st.sidebar.slider(
    "Outreach priority threshold (%)",
    10, 90, 60, step=5,
    help="Students above this threshold are predicted to enroll; below → at-risk.",
)

# Apply filters
view = df.copy()
if sel_segment != "All":
    view = view[view["institution_segment"] == sel_segment]
view = view[view["yield_probability"] >= min_prob]
view_sorted = view.sort_values("yield_probability", ascending=False)

# ── KPI row ──────────────────────────────────────────────────────────────────
total_admits = len(df)
filtered_admits = len(view)
predicted_enroll = (view["yield_probability"] >= outreach_threshold).sum()
predicted_class_size = predicted_enroll
actual_yield = view["admit_to_enroll"].mean() if "admit_to_enroll" in view.columns else None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total admitted", f"{total_admits:,}")
col2.metric("Filtered admits", f"{filtered_admits:,}")
col3.metric(f"Predicted enrolls (prob ≥ {outreach_threshold}%)", f"{predicted_class_size:,}")
if actual_yield is not None:
    col4.metric("Actual yield (filtered)", f"{actual_yield:.1%}")

# ── Probability distribution ──────────────────────────────────────────────────
st.subheader("Yield Probability Distribution")
col_a, col_b = st.columns([2, 1])

with col_a:
    fig = px.histogram(
        view,
        x="yield_probability",
        nbins=40,
        color_discrete_sequence=["#dc2626"],
        labels={"yield_probability": "Yield Probability (%)", "count": "Count"},
        title="Yield Probability Distribution (filtered view)",
    )
    fig.add_vline(
        x=outreach_threshold,
        line_dash="dash",
        line_color="#2563eb",
        annotation_text=f"Outreach threshold ({outreach_threshold}%)",
        annotation_position="top right",
        annotation_font_size=10,
    )
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("**Probability bands**")
    bands = pd.cut(view["yield_probability"],
                   bins=[0, 25, 50, 75, 100],
                   labels=["Low (0–25%)", "Moderate (25–50%)", "High (50–75%)", "Very High (75–100%)"])
    band_counts = bands.value_counts().sort_index()
    for band, count in band_counts.items():
        pct = count / len(view) * 100 if len(view) > 0 else 0
        st.write(f"**{band}:** {count:,} ({pct:.0f}%)")

# ── Predicted yield by segment ────────────────────────────────────────────────
st.subheader("Predicted vs Actual Yield by Segment")
seg_agg = df.groupby("institution_segment").agg(
    predicted_yield=("yield_score_raw", "mean"),
    actual_yield=("admit_to_enroll", "mean"),
    admit_count=("admit_to_enroll", "count"),
).reset_index()
seg_agg["predicted_pct"] = seg_agg["predicted_yield"] * 100
seg_agg["actual_pct"]    = seg_agg["actual_yield"] * 100

fig2 = go.Figure()
x = seg_agg["institution_segment"]
fig2.add_trace(go.Bar(name="Predicted yield", x=x, y=seg_agg["predicted_pct"],
                      marker_color="#dc2626"))
fig2.add_trace(go.Bar(name="Actual yield", x=x, y=seg_agg["actual_pct"],
                      marker_color="#2563eb"))
fig2.update_layout(
    barmode="group",
    title="Predicted vs Actual Yield by Institution Segment",
    yaxis_title="Yield (%)",
    height=350,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig2, use_container_width=True)

# ── Predicted class composition ───────────────────────────────────────────────
st.subheader("Predicted Enrolling Class Composition")
enrolling = view[view["yield_probability"] >= outreach_threshold]

if len(enrolling) > 0:
    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        if "institution_segment" in enrolling.columns:
            seg_comp = enrolling["institution_segment"].value_counts().reset_index()
            seg_comp.columns = ["segment", "count"]
            fig3 = px.pie(seg_comp, names="segment", values="count",
                          title="By Institution Segment", hole=0.4)
            fig3.update_layout(height=300)
            st.plotly_chart(fig3, use_container_width=True)

    with comp_col2:
        if "income_band" in enrolling.columns:
            inc_comp = enrolling["income_band"].value_counts().reset_index()
            inc_comp.columns = ["income_band", "count"]
            fig4 = px.pie(inc_comp, names="income_band", values="count",
                          title="By Income Band", hole=0.4,
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig4.update_layout(height=300)
            st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("No applicants above the outreach threshold. Lower the slider.")

# ── Admitted students table ───────────────────────────────────────────────────
st.subheader("Admitted Students — Yield Predictions")

display_cols = ["applicant_id", "institution_segment", "yield_probability",
                "shap_feature_1", "shap_value_1", "shap_feature_2", "shap_value_2",
                "shap_feature_3", "shap_value_3"]
if "admit_to_enroll" in view_sorted.columns:
    display_cols.insert(2, "admit_to_enroll")

table_df = view_sorted.head(200)[
    [c for c in display_cols if c in view_sorted.columns]
].reset_index(drop=True)

st.dataframe(
    table_df.style.background_gradient(subset=["yield_probability"], cmap="Reds"),
    use_container_width=True,
    height=400,
)

# ── Per-student SHAP detail ───────────────────────────────────────────────────
st.subheader("Per-Student SHAP Detail")
st.caption("Select a student to inspect their top-3 yield drivers.")

student_ids = view_sorted.head(200)["applicant_id"].tolist()
sel_id = st.selectbox("Applicant ID", student_ids, key="yield_sel")

if sel_id:
    row = df[df["applicant_id"] == sel_id].iloc[0]
    prob = row["yield_probability"]
    label = "Likely to enroll" if prob >= outreach_threshold else "At-risk — consider outreach"
    st.write(f"**Yield probability:** {prob:.1f}%  —  {label}")

    shap_data = []
    for i in range(1, 4):
        feat = row.get(f"shap_feature_{i}")
        val  = row.get(f"shap_value_{i}")
        if pd.notna(feat) and pd.notna(val):
            shap_data.append({"Feature": feat, "SHAP Value": float(val)})

    if shap_data:
        shap_df = pd.DataFrame(shap_data)
        fig5 = go.Figure(go.Bar(
            x=shap_df["SHAP Value"],
            y=shap_df["Feature"],
            orientation="h",
            marker_color=["#16a34a" if v > 0 else "#dc2626" for v in shap_df["SHAP Value"]],
        ))
        fig5.update_layout(
            title=f"Top-3 SHAP — Applicant {sel_id} (yield prob={prob:.1f}%)",
            xaxis_title="SHAP Value (impact on log-odds)",
            height=250,
        )
        st.plotly_chart(fig5, use_container_width=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.download_button(
    "Download filtered yield predictions (CSV)",
    data=view_sorted.to_csv(index=False),
    file_name="yield_predictions_filtered.csv",
    mime="text/csv",
)
