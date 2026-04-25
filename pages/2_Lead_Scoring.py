"""Lead scoring dashboard page — Phase 3."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils import OUTPUT_DIR

st.set_page_config(page_title="Lead Scoring", page_icon=":mag:", layout="wide")
st.title("Lead Scoring")
st.caption("XGBoost · inquired_to_applied · top-3 SHAP per lead")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    path = OUTPUT_DIR / "lead_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


df = load_predictions()

if df.empty:
    st.warning(
        "lead_predictions.csv not found. "
        "Run `notebooks/03_lead_scoring.ipynb` first."
    )
    st.stop()

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("Filters")
segments = ["All"] + sorted(df["institution_segment"].unique())
sel_segment = st.sidebar.selectbox("Institution segment", segments)

min_score = st.sidebar.slider("Minimum lead score", 0, 100, 0, step=5)

top_n = st.sidebar.number_input("Show top N leads", min_value=10, max_value=5000,
                                  value=100, step=10)

# Apply filters
view = df.copy()
if sel_segment != "All":
    view = view[view["institution_segment"] == sel_segment]
view = view[view["lead_score"] >= min_score]
view_sorted = view.sort_values("lead_score", ascending=False)

# ── KPI row ──────────────────────────────────────────────────────────────────
total_leads = len(df)
filtered_leads = len(view)
high_priority = (view["lead_score"] >= 70).sum()
actual_conversion = view["inquired_to_applied"].mean() if "inquired_to_applied" in view.columns else None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total leads", f"{total_leads:,}")
col2.metric("Filtered leads", f"{filtered_leads:,}")
col3.metric("High priority (score ≥ 70)", f"{high_priority:,}")
if actual_conversion is not None:
    col4.metric("Actual conversion rate", f"{actual_conversion:.1%}")

# ── Score distribution ────────────────────────────────────────────────────────
st.subheader("Score Distribution")
col_a, col_b = st.columns([2, 1])

with col_a:
    fig = px.histogram(
        view,
        x="lead_score",
        nbins=40,
        color_discrete_sequence=["#2563eb"],
        labels={"lead_score": "Lead Score", "count": "Count"},
        title="Lead Score Distribution (filtered view)",
    )
    fig.add_vline(x=70, line_dash="dash", line_color="#dc2626",
                  annotation_text="Priority threshold (70)", annotation_position="top right",
                  annotation_font_size=10)
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("**Score bands**")
    bands = pd.cut(view["lead_score"], bins=[0, 30, 50, 70, 100],
                   labels=["Low (0–30)", "Medium (30–50)", "High (50–70)", "Priority (70–100)"])
    band_counts = bands.value_counts().sort_index()
    for band, count in band_counts.items():
        pct = count / len(view) * 100 if len(view) > 0 else 0
        st.write(f"**{band}:** {count:,} ({pct:.0f}%)")

# ── Score by segment ──────────────────────────────────────────────────────────
st.subheader("Average Score by Segment")
seg_avg = df.groupby("institution_segment")["lead_score"].mean().sort_values(ascending=False).reset_index()
fig2 = px.bar(
    seg_avg,
    x="institution_segment",
    y="lead_score",
    color="lead_score",
    color_continuous_scale="Blues",
    labels={"institution_segment": "Segment", "lead_score": "Avg Lead Score"},
    title="Average Lead Score by Institution Segment",
)
fig2.update_layout(height=300, coloraxis_showscale=False)
st.plotly_chart(fig2, use_container_width=True)

# ── Lead table ────────────────────────────────────────────────────────────────
st.subheader(f"Top {min(top_n, len(view_sorted))} Leads")

display_cols = ["applicant_id", "institution_segment", "lead_score",
                "shap_feature_1", "shap_value_1", "shap_feature_2", "shap_value_2",
                "shap_feature_3", "shap_value_3"]
if "inquired_to_applied" in view_sorted.columns:
    display_cols.insert(2, "inquired_to_applied")

table_df = view_sorted.head(int(top_n))[
    [c for c in display_cols if c in view_sorted.columns]
].reset_index(drop=True)

st.dataframe(
    table_df.style.background_gradient(subset=["lead_score"], cmap="Blues"),
    use_container_width=True,
    height=400,
)

# ── Per-lead SHAP detail ──────────────────────────────────────────────────────
st.subheader("Per-Lead SHAP Detail")
st.caption("Select an applicant ID to inspect their top-3 SHAP drivers.")

applicant_ids = view_sorted.head(int(top_n))["applicant_id"].tolist()
sel_id = st.selectbox("Applicant ID", applicant_ids)

if sel_id:
    row = df[df["applicant_id"] == sel_id].iloc[0]
    score = row["lead_score"]
    st.write(f"**Lead score:** {score:.1f} / 100")

    shap_data = []
    for i in range(1, 4):
        feat = row.get(f"shap_feature_{i}")
        val  = row.get(f"shap_value_{i}")
        if pd.notna(feat) and pd.notna(val):
            shap_data.append({"Feature": feat, "SHAP Value": float(val),
                               "Direction": "Positive" if float(val) > 0 else "Negative"})

    if shap_data:
        shap_df = pd.DataFrame(shap_data)
        fig3 = go.Figure(go.Bar(
            x=shap_df["SHAP Value"],
            y=shap_df["Feature"],
            orientation="h",
            marker_color=["#16a34a" if v > 0 else "#dc2626" for v in shap_df["SHAP Value"]],
        ))
        fig3.update_layout(
            title=f"Top-3 SHAP — Applicant {sel_id} (score={score:.1f})",
            xaxis_title="SHAP Value (impact on log-odds)",
            height=250,
        )
        st.plotly_chart(fig3, use_container_width=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.download_button(
    "Download filtered leads (CSV)",
    data=view_sorted.to_csv(index=False),
    file_name="lead_predictions_filtered.csv",
    mime="text/csv",
)
