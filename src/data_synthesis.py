"""Synthetic applicant generator — Phase 1.

Produces a 50k-row inquiry-level dataset with three sequential outcomes:
inquired_to_applied → admitted → admit_to_enroll. Yield rates per institution
segment are calibrated to published IPEDS averages within ±2pp.

Calibration targets:
- R1: ~33% yield
- regional_state: ~22% yield
- private_lac: ~28% yield
- community_college: ~50% yield (broader inquiry pool, high enrollment among admits)
- online: ~40% yield

Features include bias-audit-relevant attributes (race, gender, first-gen, income).
The lead logit carries a small first-gen penalty so the bias audit has something
real to find — finding and mitigating bias is stronger evidence than no audit.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

SEGMENTS = ["R1", "regional_state", "private_lac", "community_college", "online"]
SEGMENT_PROBS = [0.25, 0.30, 0.15, 0.20, 0.10]

RACE_ETHNICITY = ["White", "Hispanic", "Black", "Asian", "Other"]
RACE_PROBS = [0.50, 0.20, 0.13, 0.07, 0.10]

GENDER = ["F", "M"]
GENDER_PROBS = [0.56, 0.44]

INCOME_BANDS = ["low", "middle", "high"]
INCOME_PROBS = [0.35, 0.45, 0.20]

REGIONS = ["Northeast", "Midwest", "South", "West"]
REGION_PROBS = [0.18, 0.21, 0.39, 0.22]

SOURCE_CHANNELS = [
    "organic_search",
    "paid_ad",
    "fair",
    "referral",
    "highschool_visit",
    "email_campaign",
]
SOURCE_PROBS = [0.30, 0.20, 0.10, 0.15, 0.15, 0.10]

INTENDED_MAJORS = [
    "Business",
    "Engineering",
    "CS",
    "Nursing",
    "Education",
    "Liberal Arts",
    "Sciences",
    "Undecided",
]
MAJOR_PROBS = [0.20, 0.15, 0.13, 0.12, 0.08, 0.12, 0.12, 0.08]

GPA_MEAN_BY_SEGMENT = {
    "R1": 3.60,
    "regional_state": 3.20,
    "private_lac": 3.50,
    "community_college": 2.90,
    "online": 3.00,
}

ADMIT_RATE_BY_SEGMENT = {
    "R1": 0.25,
    "regional_state": 0.65,
    "private_lac": 0.55,
    "community_college": 0.95,
    "online": 0.85,
}

YIELD_TARGET_BY_SEGMENT = {
    "R1": 0.33,
    "regional_state": 0.22,
    "private_lac": 0.28,
    "community_college": 0.50,
    "online": 0.40,
}

AID_BASE_BY_SEGMENT = {
    "R1": 8_000,
    "regional_state": 5_000,
    "private_lac": 18_000,
    "community_college": 3_000,
    "online": 4_000,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrated_intercept(z: np.ndarray, target: float) -> float:
    """Bisection on logit intercept so sigmoid(z + intercept).mean() == target."""
    lo, hi = -10.0, 10.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _sigmoid(z + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_applicants(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    applicant_id = np.arange(n)
    segment = rng.choice(SEGMENTS, size=n, p=SEGMENT_PROBS)
    race = rng.choice(RACE_ETHNICITY, size=n, p=RACE_PROBS)
    gender = rng.choice(GENDER, size=n, p=GENDER_PROBS)
    first_gen = rng.binomial(1, 0.30, n)
    income = rng.choice(INCOME_BANDS, size=n, p=INCOME_PROBS)
    region = rng.choice(REGIONS, size=n, p=REGION_PROBS)

    gpa = np.zeros(n)
    for seg, mu in GPA_MEAN_BY_SEGMENT.items():
        mask = segment == seg
        m = mu / 4.0
        concentration = 30.0
        a = m * concentration
        b = (1.0 - m) * concentration
        gpa[mask] = rng.beta(a, b, mask.sum()) * 4.0
    gpa = np.clip(gpa, 0.0, 4.0).round(2)

    sat = 1100 + (gpa - 3.0) * 200 + rng.normal(0, 80, n)
    sat = np.clip(sat, 400, 1600).round(0).astype(int)

    distance_miles = rng.lognormal(mean=4.0, sigma=1.2, size=n)
    distance_miles = np.clip(distance_miles, 1, 3000).round(1)

    campus_visit_flag = rng.binomial(1, 0.25, n)
    email_engagement_score = np.clip(rng.normal(50, 20, n), 0, 100).round(1)
    financial_aid_inquiry_flag = rng.binomial(1, 0.55, n)
    days_since_first_inquiry = rng.integers(1, 365, n)

    intended_major = rng.choice(INTENDED_MAJORS, size=n, p=MAJOR_PROBS)
    application_date_relative_to_deadline = rng.normal(-21, 30, n).round(0).astype(int)
    source_channel = rng.choice(SOURCE_CHANNELS, size=n, p=SOURCE_PROBS)

    # Lead label — calibrated near 30% positive
    high_intent_source = np.isin(source_channel, ["highschool_visit", "referral"]).astype(float)
    z_lead = (
        0.8 * campus_visit_flag
        + 0.5 * financial_aid_inquiry_flag
        + 0.015 * email_engagement_score
        - 0.0008 * distance_miles
        + 0.6 * (gpa - 3.0)
        + 0.4 * high_intent_source
        - 0.15 * first_gen
    )
    intercept_lead = _calibrated_intercept(z_lead, 0.30)
    p_lead = _sigmoid(z_lead + intercept_lead)
    inquired_to_applied = (rng.uniform(0, 1, n) < p_lead).astype(int)

    # Admit — selectivity by segment via score quantile
    admitted = np.zeros(n, dtype=int)
    for seg, admit_rate in ADMIT_RATE_BY_SEGMENT.items():
        mask = (segment == seg) & (inquired_to_applied == 1)
        ns = int(mask.sum())
        if ns == 0:
            continue
        score = (
            (gpa[mask] - 3.0) * 1.5
            + (sat[mask] - 1200) / 200.0
            + rng.normal(0, 0.3, ns)
        )
        thr = np.quantile(score, 1.0 - admit_rate)
        admitted[np.where(mask)[0]] = (score >= thr).astype(int)

    # Yield — calibrated per segment
    aid_package_amount = np.zeros(n)
    scholarship_offer_flag = np.zeros(n, dtype=int)
    days_to_deposit_deadline = np.zeros(n, dtype=int)
    peer_admit_count = np.zeros(n, dtype=int)
    admit_to_enroll = np.zeros(n, dtype=int)

    for seg, target in YIELD_TARGET_BY_SEGMENT.items():
        mask = (segment == seg) & (admitted == 1)
        ns = int(mask.sum())
        if ns == 0:
            continue
        seg_idx = np.where(mask)[0]

        seg_income = income[mask]
        income_factor = np.where(
            seg_income == "low", 1.5, np.where(seg_income == "middle", 1.0, 0.5)
        )
        base_aid = AID_BASE_BY_SEGMENT[seg]
        aid = rng.normal(base_aid, base_aid * 0.3, ns) * income_factor
        aid = np.clip(aid, 0, 60_000).round(0)
        scholarship = rng.binomial(1, 0.35, ns)
        days_deadline = rng.integers(1, 90, ns)
        peers = rng.poisson(3, ns)

        aid_package_amount[seg_idx] = aid
        scholarship_offer_flag[seg_idx] = scholarship
        days_to_deposit_deadline[seg_idx] = days_deadline
        peer_admit_count[seg_idx] = peers

        z_yield = (
            0.6 * campus_visit_flag[mask]
            + 0.5 * financial_aid_inquiry_flag[mask]
            + (aid / 10_000.0)
            + 0.4 * scholarship
            - 0.0004 * distance_miles[mask]
            + 0.3 * (gpa[mask] - 3.2)
            + rng.normal(0, 0.5, ns)
        )
        intercept_yield = _calibrated_intercept(z_yield, target)
        p_yield = _sigmoid(z_yield + intercept_yield)
        admit_to_enroll[seg_idx] = (rng.uniform(0, 1, ns) < p_yield).astype(int)

    df = pd.DataFrame(
        {
            "applicant_id": applicant_id,
            "institution_segment": segment,
            "race_ethnicity": race,
            "gender": gender,
            "first_gen_flag": first_gen,
            "income_band": income,
            "region": region,
            "hs_gpa": gpa,
            "sat_score": sat,
            "distance_miles": distance_miles,
            "campus_visit_flag": campus_visit_flag,
            "email_engagement_score": email_engagement_score,
            "financial_aid_inquiry_flag": financial_aid_inquiry_flag,
            "days_since_first_inquiry": days_since_first_inquiry,
            "intended_major": intended_major,
            "application_date_relative_to_deadline": application_date_relative_to_deadline,
            "source_channel": source_channel,
            "aid_package_amount": aid_package_amount,
            "scholarship_offer_flag": scholarship_offer_flag,
            "days_to_deposit_deadline": days_to_deposit_deadline,
            "peer_admit_count": peer_admit_count,
            "inquired_to_applied": inquired_to_applied,
            "admitted": admitted,
            "admit_to_enroll": admit_to_enroll,
        }
    )
    return df


def yield_rate_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Yield = admit_to_enroll among admitted, by institution segment."""
    admits = df[df["admitted"] == 1]
    return (
        admits.groupby("institution_segment")["admit_to_enroll"]
        .mean()
        .rename("observed_yield")
        .reset_index()
        .assign(target_yield=lambda d: d["institution_segment"].map(YIELD_TARGET_BY_SEGMENT))
        .assign(delta_pp=lambda d: (d["observed_yield"] - d["target_yield"]) * 100)
    )


def main(out_path: Optional[str] = None) -> pd.DataFrame:
    from src.utils import SYNTHETIC_DIR

    df = generate_applicants()
    target = out_path or (SYNTHETIC_DIR / "applicants.csv")
    df.to_csv(target, index=False)
    return df


if __name__ == "__main__":
    main()
