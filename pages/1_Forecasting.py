"""Demographic forecasting dashboard page — Phase 2."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils import OUTPUT_DIR

st.set_page_config(
    page_title="Forecasting",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)
st.title("Demographic Forecasting")
st.caption("Prophet vs SARIMA — US enrollment & HS graduate projections to 2035")


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    path = OUTPUT_DIR / "forecasts.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


df = load_forecasts()

if df.empty:
    st.warning("forecasts.csv not found. Run `notebooks/02_demographic_forecasting.ipynb` first.")
    st.stop()

# ── Sidebar controls ────────────────────────────────────────────────────────
st.sidebar.header("Controls")
series_options = {
    "Total Enrollment": "total_enrollment",
    "High School Graduates": "hs_graduates",
    "Race/Ethnicity Breakdown": "race_enrollment",
}
selected_label = st.sidebar.selectbox("Series", list(series_options.keys()))
selected_series = series_options[selected_label]

ci_level = st.sidebar.radio("Confidence Interval", ["80%", "95%", "Both", "None"], index=0)
model_choice = st.sidebar.radio("Model", ["Prophet", "SARIMA", "Both"], index=2)

train_end = 2017
test_end = 2023

# ── Main chart ───────────────────────────────────────────────────────────────
if selected_series in ("total_enrollment", "hs_graduates"):
    sub = df[df["series"] == selected_series].copy()
    unit = 1e6
    ylabel = "Millions"
    title_map = {
        "total_enrollment": "Total Fall Enrollment in Degree-Granting Institutions",
        "hs_graduates": "US High School Graduates",
    }
    title = title_map[selected_series]

    fig = go.Figure()

    ci_suffix = {"80%": "_80", "95%": "_95", "Both": "_80", "None": None}

    for model, color in [("prophet", "#2563eb"), ("sarima", "#dc2626")]:
        if model_choice == "Both" or model_choice.lower() == model:
            # Add CI bands
            for ci, opacity in (
                [("_80", 0.18), ("_95", 0.10)] if ci_level == "Both" else
                [(ci_suffix[ci_level], 0.22)] if ci_level != "None" else []
            ):
                lower_col = f"{model}_lower{ci}"
                upper_col = f"{model}_upper{ci}"
                if lower_col in sub.columns and sub[lower_col].notna().any():
                    fig.add_trace(go.Scatter(
                        x=pd.concat([sub["year"], sub["year"][::-1]]),
                        y=pd.concat([sub[upper_col] / unit, sub[lower_col][::-1] / unit]),
                        fill="toself",
                        fillcolor=color,
                        opacity=opacity,
                        line=dict(width=0),
                        name=f"{model.capitalize()} {ci.strip('_')} CI",
                        showlegend=True,
                    ))

            # Forecast line
            yhat_col = f"{model}_yhat"
            if yhat_col in sub.columns:
                fig.add_trace(go.Scatter(
                    x=sub["year"],
                    y=sub[yhat_col] / unit,
                    mode="lines",
                    name=f"{model.capitalize()} forecast",
                    line=dict(color=color, width=2, dash="dash" if model == "sarima" else "solid"),
                ))

    # Actual values
    actual_sub = sub.dropna(subset=["actual"])
    fig.add_trace(go.Scatter(
        x=actual_sub["year"],
        y=actual_sub["actual"] / unit,
        mode="lines+markers",
        name="Actual",
        line=dict(color="black", width=2),
        marker=dict(size=5),
    ))

    # Vertical reference lines
    for x_val, label, dash in [
        (train_end, "Train/Test split", "dot"),
        (test_end, "Forecast start", "dash"),
    ]:
        fig.add_vline(x=x_val, line_dash=dash, line_color="gray", opacity=0.6,
                      annotation_text=label, annotation_position="top right",
                      annotation_font_size=10)

    fig.update_layout(
        title=f"{title} — 1980–2035",
        xaxis_title="Year",
        yaxis_title=ylabel,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Key metrics ─────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    last_actual = sub.dropna(subset=["actual"])["actual"].iloc[-1]
    last_year = int(sub.dropna(subset=["actual"])["year"].iloc[-1])

    sarima_2035 = sub[sub["year"] == 2035]["sarima_yhat"].values
    prophet_2035 = sub[sub["year"] == 2035]["prophet_yhat"].values

    col1.metric(f"Last Actual ({last_year})", f"{last_actual/1e6:.2f}M")
    if len(sarima_2035):
        col2.metric("SARIMA 2035 Forecast", f"{sarima_2035[0]/1e6:.2f}M")
    if len(prophet_2035):
        col3.metric("Prophet 2035 Forecast", f"{prophet_2035[0]/1e6:.2f}M")

    with st.expander("Model performance on 2018–2023 hold-out set"):
        st.caption(
            "Both models trained on 1980–2017. SARIMA typically achieves lower MAPE on "
            "this series due to recent-lag dynamics. Prophet extrapolates the long-run "
            "growth trend and over-predicts during the enrollment stagnation period."
        )

# ── Race/Ethnicity view ─────────────────────────────────────────────────────
else:
    race_sub = df[df["series"] == "race_enrollment"].copy()
    race_groups = sorted(race_sub["segment"].unique())

    selected_groups = st.sidebar.multiselect(
        "Race/Ethnicity Groups",
        race_groups,
        default=["White", "Hispanic", "Black", "Asian"],
    )

    if not selected_groups:
        st.info("Select at least one group in the sidebar.")
        st.stop()

    fig = go.Figure()
    colors = [
        "#2563eb", "#dc2626", "#16a34a", "#d97706",
        "#7c3aed", "#db2777", "#0891b2", "#65a30d", "#9a3412",
    ]

    for i, group in enumerate(selected_groups):
        g = race_sub[race_sub["segment"] == group]
        color = colors[i % len(colors)]

        # Actual points
        actuals_g = g.dropna(subset=["actual"])
        fig.add_trace(go.Scatter(
            x=actuals_g["year"],
            y=actuals_g["actual"] / 1e6,
            mode="markers",
            name=f"{group} (actual)",
            marker=dict(color=color, size=8),
        ))

        # Linear trend projection (stored in sarima_yhat for race segments)
        fig.add_trace(go.Scatter(
            x=g["year"],
            y=g["sarima_yhat"] / 1e6,
            mode="lines",
            name=f"{group} (trend)",
            line=dict(color=color, width=1.5, dash="dash"),
            opacity=0.7,
            showlegend=False,
        ))

    fig.add_vline(x=2022, line_dash="dot", line_color="gray", opacity=0.5,
                  annotation_text="Last IPEDS data", annotation_font_size=9)
    fig.update_layout(
        title="Enrollment by Race/Ethnicity — IPEDS EFFY 2018–2022 + Linear Projections",
        xaxis_title="Year",
        yaxis_title="Enrollment (millions)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Race projections use linear trend extrapolation from 5 data points (2018–2022). "
        "Not suitable for precise forecasting — treat as directional signal only. "
        "For deeper demographic projections, integrate Census Bureau population projections."
    )

    # Composition table 2018 vs 2022
    st.subheader("Enrollment Composition: 2018 vs 2022")
    r18 = race_sub[race_sub["year"] == 2018].set_index("segment")["actual"]
    r22 = race_sub[race_sub["year"] == 2022].set_index("segment")["actual"]
    comp = pd.DataFrame({"2018": r18, "2022": r22}).dropna()
    comp["Change"] = comp["2022"] - comp["2018"]
    comp["Change %"] = ((comp["2022"] - comp["2018"]) / comp["2018"] * 100).round(1)
    comp = comp.sort_values("Change", ascending=False)
    comp = comp.applymap(lambda x: f"{x:,.0f}" if isinstance(x, float) and abs(x) >= 1 else x)
    st.dataframe(comp, use_container_width=True)
