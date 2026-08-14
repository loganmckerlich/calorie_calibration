"""
Run from project root: python src/app/build.py
Reads  data/processed/combined_daily_summary.csv
Writes src/app/data.json
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.helpers.imputing import impute_calories, impute_weight
from src.helpers.fetch import load

# ── helpers ────────────────────────────────────────────────────────────────

def to_list(series, decimals=2):
    """Convert a pandas series to a JSON-safe list (NaN → null)."""
    return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), decimals) for v in series]


# ── analysis ───────────────────────────────────────────────────────────────

def build():
    df = load()
    df = impute_calories(df)
    df = impute_weight(df)

    # net calories per day: surplus = positive, deficit = negative
    df["net_calories"] = df["consumedKilocaloriesImputed"] - df["totalKilocalories"]

    # cumulative predicted weight change (lbs) from day 0
    df["predicted_delta_lb"] = df["net_calories"] / 3500.0
    df["cumulative_predicted_lb"] = df["predicted_delta_lb"].cumsum()

    # cumulative actual weight change
    start_weight = df["weightImputed"].iloc[0]
    df["cumulative_actual_lb"] = df["weightImputed"] - start_weight

    # error: how much actual diverges from predicted
    df["error_lb"]   = df["cumulative_actual_lb"] - df["cumulative_predicted_lb"]
    df["error_kcal"] = df["error_lb"] * 3500

    # Rolling error rate: how many kcal/day is the bias in each 28-day window?
    # diff(28) = how much cumulative error grew over the past 28 days → divide for per-day rate.
    # A flat line = consistent bias; slope up/down = bias is changing.
    df["error_rate_kcal_per_day"] = (df["error_lb"].diff(28) / 28 * 3500).rolling(7, min_periods=3).mean()

    # per-weigh-in-window stats
    weigh_in_dates = df.index[df["weight"].notna()]
    windows = []
    for i in range(1, len(weigh_in_dates)):
        s, e = weigh_in_dates[i - 1], weigh_in_dates[i]
        predicted_change = df.loc[s:e, "predicted_delta_lb"].sum()
        actual_change    = float(df.loc[e, "weight"]) - float(df.loc[s, "weight"])
        days             = int((e - s).days)
        pct_error        = round((predicted_change - actual_change) / abs(actual_change) * 100, 1) if actual_change != 0 else None
        windows.append({
            "start": s.strftime("%Y-%m-%d"),
            "end":   e.strftime("%Y-%m-%d"),
            "predicted_change_lb": round(predicted_change, 2),
            "actual_change_lb":    round(actual_change, 2),
            "days": days,
            "pct_error": pct_error,
        })

    # ── Key calibration metrics ────────────────────────────────────────────
    # Use first→last weigh-in window for most grounded estimate
    first_wi, last_wi = weigh_in_dates[0], weigh_in_dates[-1]
    span_days = (last_wi - first_wi).days
    total_error_kcal = float(df.loc[last_wi, "error_kcal"])
    daily_kcal_error = round(total_error_kcal / span_days) if span_days > 0 else 0

    avg_daily_net = float(df.loc[first_wi:last_wi, "net_calories"].mean())
    avg_pct_error = round(daily_kcal_error / abs(avg_daily_net) * 100, 1) if avg_daily_net != 0 else 0.0

    # window pct errors (finite windows only)
    valid_pcts = [w["pct_error"] for w in windows if w["pct_error"] is not None]
    median_window_pct_error = round(float(np.median(valid_pcts)), 1) if valid_pcts else 0.0

    logged_days = int(df["consumedKilocalories"].notna().sum())
    total_days  = len(df)

    # 7-day rolling averages for health charts (smooths noise)
    df["hr_resting_smooth"]  = df["restingHeartRate"].rolling(7, min_periods=3).mean()
    df["hr_max_smooth"]      = df["maxHeartRate"].rolling(7, min_periods=3).mean()
    df["calories_burned_smooth"]   = df["totalKilocalories"].rolling(7, min_periods=3).mean()
    df["calories_consumed_smooth"] = df["consumedKilocaloriesImputed"].rolling(7, min_periods=3).mean()

    out = {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        # health page
        "hr_resting":         to_list(df["restingHeartRate"], 0),
        "hr_max":             to_list(df["maxHeartRate"], 0),
        "hr_resting_smooth":  to_list(df["hr_resting_smooth"], 1),
        "hr_max_smooth":      to_list(df["hr_max_smooth"], 1),
        "calories_burned":    to_list(df["totalKilocalories"], 0),
        "calories_consumed":  to_list(df["consumedKilocaloriesImputed"], 0),
        "calories_burned_smooth":   to_list(df["calories_burned_smooth"], 0),
        "calories_consumed_smooth": to_list(df["calories_consumed_smooth"], 0),
        "weight":             to_list(df["weight"], 1),
        # calibration page
        "error_kcal":              to_list(df["error_kcal"], 0),
        "error_rate_kcal_per_day": to_list(df["error_rate_kcal_per_day"], 0),
        "weigh_in_windows": windows,
        "calibration": {
            "daily_kcal_error":        daily_kcal_error,
            "avg_pct_error":           avg_pct_error,
            "median_window_pct_error": median_window_pct_error,
            "span_days":               span_days,
        },
        "summary": {
            "start_weight_lb":           round(start_weight, 1),
            "total_days":                total_days,
            "logged_days":               logged_days,
            "coverage_pct":              round(100 * logged_days / total_days, 1),
            "total_predicted_change_lb": round(float(df["cumulative_predicted_lb"].iloc[-1]), 1),
            "total_actual_change_lb":    round(float(df["cumulative_actual_lb"].iloc[-1]), 1),
            "final_error_lb":            round(float(df["error_lb"].iloc[-1]), 1),
        },
    }

    with open("src/app/data.json", "w") as f:
        json.dump(out, f)

    s = out["summary"]
    c = out["calibration"]
    print(f"✓  Wrote src/app/data.json")
    print(f"   Period:     {out['dates'][0]} → {out['dates'][-1]}  ({total_days} days, {logged_days} logged)")
    print(f"   Predicted:  {s['total_predicted_change_lb']:+.1f} lb")
    print(f"   Actual:     {s['total_actual_change_lb']:+.1f} lb")
    print(f"   Daily kcal error: {c['daily_kcal_error']:+d} kcal/day  ({c['avg_pct_error']:+.1f}%)")


if __name__ == "__main__":
    build()
