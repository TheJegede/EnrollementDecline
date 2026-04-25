"""Demographic forecasting — Phase 2 (Prophet primary, SARIMA benchmark).

Train on 1980-2017, test on 2018-2023, project to 2035 with 80% and 95% CIs.
"""
from __future__ import annotations

import logging
import re
import warnings
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import IPEDS_DIR, NCES_DIR, OUTPUT_DIR, set_seeds

log = logging.getLogger(__name__)

TRAIN_END = 2017
TEST_START = 2018
TEST_END = 2023
HORIZON_END = 2035

# IPEDS EFFY column names for race/ethnicity totals
RACE_COLS: dict[str, str] = {
    "White": "EFYWHITT",
    "Black": "EFYBKAAT",
    "Hispanic": "EFYHISPT",
    "Asian": "EFYASIAT",
    "AmIndian": "EFYAIANT",
    "PacIsl": "EFYNHPIT",
    "TwoOrMore": "EFY2MORT",
    "Unknown": "EFYUNKNT",
    "NonresAlien": "EFYNRALT",
}

# Matches NCES footnote markers like \1\ or \12\
_FOOTNOTE_PAT = re.compile(r'\\\d+\\')


def _clean_year(val) -> Optional[int]:
    """Strip NCES footnote markers and return integer year; None for non-year cells."""
    s = _FOOTNOTE_PAT.sub("", str(val)).strip()
    # School-year format: "1979-80" or "2030-31" → take the ending year
    m = re.match(r"^(\d{4})-(\d{2,4})$", s)
    if m:
        start = int(m.group(1))
        end_s = m.group(2)
        if len(end_s) == 2:
            century = (start // 100) * 100
            end = century + int(end_s)
            if end < start:
                end += 100
        else:
            end = int(end_s)
        return end
    if re.match(r"^\d{4}$", s):
        return int(s)
    return None


def _parse_nces_table(
    path: Path, year_col: int, value_col: int, value_name: str
) -> pd.DataFrame:
    """Parse an NCES XLS table. Returns DataFrame with [year, value_name]."""
    raw = pd.read_excel(path, header=None)
    records = []
    for _, row in raw.iterrows():
        yr = _clean_year(row[year_col])
        if yr is None:
            continue
        try:
            val = float(str(row[value_col]).replace(",", "").strip())
        except (ValueError, TypeError):
            continue
        if np.isnan(val) or val <= 0:
            continue
        records.append({"year": yr, value_name: val})

    return (
        pd.DataFrame(records)
        .drop_duplicates("year")
        .sort_values("year")
        .reset_index(drop=True)
    )


def load_total_enrollment(start_year: int = 1980) -> pd.DataFrame:
    """Parse NCES 303.10 → annual total degree-granting enrollment.

    Columns: year (int), enrollment (float), is_projected (bool).
    Gaps in the year range are filled by linear interpolation.
    """
    df = _parse_nces_table(NCES_DIR / "tabn303.10.xls", 0, 1, "enrollment")
    df = df[df["year"] >= start_year].copy()
    # Ensure continuous year range (interpolate any gaps)
    full = pd.DataFrame({"year": range(int(df["year"].min()), int(df["year"].max()) + 1)})
    df = full.merge(df, on="year", how="left")
    df["enrollment"] = df["enrollment"].interpolate(method="linear")
    df["is_projected"] = df["year"] >= 2021
    return df.reset_index(drop=True)


def load_hs_graduates(start_year: int = 1986) -> pd.DataFrame:
    """Parse NCES 219.10 → annual total high school graduates.

    Columns: year (int), hs_graduates (float), is_projected (bool).
    """
    df = _parse_nces_table(NCES_DIR / "tabn219.10.xls", 0, 1, "hs_graduates")
    df = df[df["year"] >= start_year].copy()
    full = pd.DataFrame({"year": range(int(df["year"].min()), int(df["year"].max()) + 1)})
    df = full.merge(df, on="year", how="left")
    df["hs_graduates"] = df["hs_graduates"].interpolate(method="linear")
    df["is_projected"] = df["year"] >= 2022
    return df.reset_index(drop=True)


def load_ipeds_race_enrollment() -> pd.DataFrame:
    """Aggregate IPEDS EFFY 2018-2022 to national enrollment by race/ethnicity.

    Returns DataFrame with columns: year, total, White, Black, Hispanic, Asian, etc.
    """
    records = []
    for year in range(2018, 2023):
        zip_path = IPEDS_DIR / f"EFFY{year}.zip"
        if not zip_path.exists():
            log.warning("missing %s", zip_path.name)
            continue
        with zipfile.ZipFile(zip_path) as zf:
            csv_name = next(
                n for n in zf.namelist()
                if n.lower().endswith(".csv") and "rv" not in n.lower()
            )
            df = pd.read_csv(zf.open(csv_name), encoding="latin-1", low_memory=False)

        # EFFYLEV == 1 = "All students" (consistent across 2018-2022)
        df = df[df["EFFYLEV"] == 1].copy()

        row: dict[str, object] = {"year": year}
        for col_name, ipeds_col in [("total", "EFYTOTLT")] + list(RACE_COLS.items()):
            if ipeds_col in df.columns:
                row[col_name] = pd.to_numeric(df[ipeds_col], errors="coerce").sum()
            else:
                row[col_name] = np.nan
        records.append(row)

    return pd.DataFrame(records).sort_values("year").reset_index(drop=True)


def _to_prophet_df(df: pd.DataFrame, year_col: str, value_col: str) -> pd.DataFrame:
    """Convert year + value DataFrame to Prophet format (ds, y)."""
    return pd.DataFrame({
        "ds": pd.to_datetime(df[year_col].astype(str) + "-12-31"),
        "y": df[value_col].astype(float),
    })


def _silence_prophet() -> None:
    """Suppress noisy Prophet / CmdStan logging."""
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    try:
        import cmdstanpy
        cmdstanpy.utils.get_logger().setLevel(logging.ERROR)
    except Exception:
        pass


def train_prophet(
    train_df: pd.DataFrame,
    *,
    changepoint_prior_scale: float = 0.05,
    interval_width: float = 0.80,
) -> object:
    """Fit Prophet on annual training data. train_df must have (ds, y) columns."""
    from prophet import Prophet  # noqa: PLC0415 — lazy import

    _silence_prophet()
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=interval_width,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(train_df)
    return m


def train_sarima(
    train_series: pd.Series,
    order: tuple = (1, 1, 1),
) -> object:
    """Fit SARIMAX with given order on annual data. Returns fitted result."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SARIMAX(train_series, order=order, trend="c").fit(disp=False)
    return result


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _direction_acc(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))))


def evaluate_models(
    actuals: pd.DataFrame,
    prophet_model,
    sarima_result,
    year_col: str,
    value_col: str,
) -> dict:
    """Compute MAPE, RMSE, direction accuracy on the 2018-2023 hold-out set."""
    test = actuals[
        (actuals[year_col] >= TEST_START) & (actuals[year_col] <= TEST_END)
    ].dropna(subset=[value_col]).copy()
    if test.empty:
        return {}

    years = test[year_col].values
    actual = test[value_col].values.astype(float)

    future = pd.DataFrame({"ds": pd.to_datetime([f"{y}-12-31" for y in years])})
    prophet_pred = prophet_model.predict(future)["yhat"].values.astype(float)

    train_last = int(actuals[actuals[year_col] <= TRAIN_END][year_col].max())
    n_forecast = TEST_END - train_last
    sarima_full = np.array(sarima_result.forecast(steps=n_forecast), dtype=float)
    offset = int(years[0]) - (train_last + 1)
    sarima_pred = sarima_full[offset: offset + len(years)]

    out: dict = {
        "prophet_mape": _mape(actual, prophet_pred),
        "prophet_rmse": _rmse(actual, prophet_pred),
        "sarima_mape": _mape(actual, sarima_pred),
        "sarima_rmse": _rmse(actual, sarima_pred),
    }
    if len(actual) > 1:
        out["prophet_direction_acc"] = _direction_acc(actual, prophet_pred)
        out["sarima_direction_acc"] = _direction_acc(actual, sarima_pred)
    return out


def generate_forecast(
    all_data: pd.DataFrame,
    prophet_80,
    prophet_95,
    sarima_result,
    year_col: str,
    value_col: str,
    series_name: str,
    segment: str = "total",
    start_year: int = 1980,
    horizon_end: int = HORIZON_END,
) -> pd.DataFrame:
    """Build combined forecast table for all years from start_year to horizon_end.

    Includes 80% and 95% CIs from both Prophet and SARIMA.
    """
    all_years = list(range(start_year, horizon_end + 1))

    future = pd.DataFrame({"ds": pd.to_datetime([f"{y}-12-31" for y in all_years])})
    p80 = prophet_80.predict(future)
    p95 = prophet_95.predict(future)

    train_last = int(all_data[all_data[year_col] <= TRAIN_END][year_col].max())
    n_sarima = horizon_end - train_last

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sf_80 = sarima_result.get_forecast(steps=n_sarima).summary_frame(alpha=0.20)
        sf_95 = sarima_result.get_forecast(steps=n_sarima).summary_frame(alpha=0.05)

    sarima_years = list(range(train_last + 1, horizon_end + 1))
    sarima_out = pd.DataFrame({
        "year": sarima_years,
        "sarima_yhat": sf_80["mean"].values,
        "sarima_lower_80": sf_80["mean_ci_lower"].values,
        "sarima_upper_80": sf_80["mean_ci_upper"].values,
        "sarima_lower_95": sf_95["mean_ci_lower"].values,
        "sarima_upper_95": sf_95["mean_ci_upper"].values,
    })

    result = pd.DataFrame({"year": all_years})
    result = result.merge(
        all_data[[year_col, value_col]].rename(
            columns={year_col: "year", value_col: "actual"}
        ),
        on="year",
        how="left",
    )
    result["prophet_yhat"] = p80["yhat"].values
    result["prophet_lower_80"] = p80["yhat_lower"].values
    result["prophet_upper_80"] = p80["yhat_upper"].values
    result["prophet_lower_95"] = p95["yhat_lower"].values
    result["prophet_upper_95"] = p95["yhat_upper"].values
    result = result.merge(sarima_out, on="year", how="left")
    result["series"] = series_name
    result["segment"] = segment

    return result[[
        "year", "series", "segment", "actual",
        "prophet_yhat", "prophet_lower_80", "prophet_upper_80",
        "prophet_lower_95", "prophet_upper_95",
        "sarima_yhat", "sarima_lower_80", "sarima_upper_80",
        "sarima_lower_95", "sarima_upper_95",
    ]]


def run_phase2(save: bool = True) -> tuple[pd.DataFrame, dict]:
    """Full Phase 2 pipeline: load → train → evaluate → forecast → save.

    Returns (forecasts_df, metrics_dict).
    """
    set_seeds()

    enroll_df = load_total_enrollment()
    hs_df = load_hs_graduates()
    race_df = load_ipeds_race_enrollment()

    all_forecasts: list[pd.DataFrame] = []
    metrics_all: dict = {}

    for series_name, df, value_col, start_yr in [
        ("total_enrollment", enroll_df, "enrollment", 1980),
        ("hs_graduates", hs_df, "hs_graduates", 1986),
    ]:
        log.info("Fitting models for: %s", series_name)
        train = df[df["year"] <= TRAIN_END].dropna(subset=[value_col]).copy()
        train_prophet_df = _to_prophet_df(train, "year", value_col)

        m_80 = train_prophet(train_prophet_df, interval_width=0.80)
        m_95 = train_prophet(train_prophet_df, interval_width=0.95)

        train_series = pd.Series(
            train[value_col].values,
            name=value_col,
        )
        sarima_res = train_sarima(train_series)

        metrics = evaluate_models(df, m_80, sarima_res, "year", value_col)
        metrics_all[series_name] = metrics
        log.info(
            "%s  prophet MAPE=%.2f%%  SARIMA MAPE=%.2f%%",
            series_name,
            metrics.get("prophet_mape", float("nan")),
            metrics.get("sarima_mape", float("nan")),
        )

        fcast = generate_forecast(
            df, m_80, m_95, sarima_res,
            "year", value_col, series_name,
            start_year=start_yr,
        )
        all_forecasts.append(fcast)

    # Race/ethnicity: linear trend projection (only 2018-2022 data available)
    for race_group in RACE_COLS:
        if race_group not in race_df.columns:
            continue
        sub = race_df[["year", race_group]].dropna()
        if len(sub) < 2:
            continue
        x = sub["year"].values.astype(float)
        y_vals = sub[race_group].values.astype(float)
        slope, intercept = np.polyfit(x, y_vals, 1)

        all_years = list(range(int(sub["year"].min()), HORIZON_END + 1))
        proj = slope * np.array(all_years, dtype=float) + intercept
        actuals_map = dict(zip(sub["year"], sub[race_group]))

        fcast = pd.DataFrame({
            "year": all_years,
            "series": "race_enrollment",
            "segment": race_group,
            "actual": [actuals_map.get(y, np.nan) for y in all_years],
            "prophet_yhat": np.nan,
            "prophet_lower_80": np.nan,
            "prophet_upper_80": np.nan,
            "prophet_lower_95": np.nan,
            "prophet_upper_95": np.nan,
            "sarima_yhat": proj,
            "sarima_lower_80": np.nan,
            "sarima_upper_80": np.nan,
            "sarima_lower_95": np.nan,
            "sarima_upper_95": np.nan,
        })
        all_forecasts.append(fcast)

    combined = pd.concat(all_forecasts, ignore_index=True)

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(OUTPUT_DIR / "forecasts.csv", index=False)
        log.info("Saved forecasts.csv (%d rows)", len(combined))

    return combined, metrics_all
