# Model Card — Demographic Forecasting

## Model Details

| Attribute | Value |
|-----------|-------|
| Task | Time-series forecasting of US college enrollment and high school graduate population |
| Primary algorithm | SARIMA (statsmodels) |
| Benchmark | Prophet (Meta) |
| Training period | 1980–2017 (NCES Digest tables 219.10, 303.10, 302.10) |
| Hold-out | 2018–2023 |
| Forecast horizon | 2024–2035 with 80% and 95% confidence intervals |
| Artifact | `data/output/forecasts.csv` |

## Segments Modeled

- Total US college enrollment
- High school graduates (national)
- Enrollment by race/ethnicity: White, Black, Hispanic, Asian, AmIndian, PacIsl, TwoOrMore, Unknown, NonresAlien
- (Age band and census region data acquired but not separately forecasted in final model)

## Metrics (hold-out 2018–2023)

| Model | MAPE | Notes |
|-------|------|-------|
| SARIMA | **3.9%** | Winner — correctly captures post-2010 enrollment plateau |
| Prophet | ~12% | Over-predicts — extrapolates 1980–2017 upward growth trend past the demographic cliff |

## Key Findings

- White enrollment projected −38% from 2023 to 2035
- Hispanic enrollment projected +8%, Asian +16%, TwoOrMore +23%
- Total US enrollment SARIMA projection: ~22.5M by 2030 (+13% vs 2023 actual of ~19.9M)
- Structural break risk: post-pandemic enrollment patterns may not follow pre-2020 SARIMA seasonality

## Known Limitations

- NCES projections table (303.10) is included in the "actual" column for 2024–2030 — SARIMA trained only on 1980–2017 historical data, not on NCES projections
- National-level model only; state-level granularity requires additional data
- SARIMA does not model birth-cohort dynamics directly — demographic cliff effects are captured via trend extrapolation, not causal modeling
