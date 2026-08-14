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

    # 28-day rolling mean of error_kcal (≥7 days to show)
    df["rolling_error_kcal"] = df["error_kcal"].rolling(28, min_periods=7).mean()

    # per-weigh-in-window regression points
    weigh_in_dates = df.index[df["weight"].notna()]
    windows = []
    for i in range(1, len(weigh_in_dates)):
        s, e = weigh_in_dates[i - 1], weigh_in_dates[i]
        predicted_change = df.loc[s:e, "predicted_delta_lb"].sum()
        actual_change    = float(df.loc[e, "weight"]) - float(df.loc[s, "weight"])
        windows.append({
            "start": s.strftime("%Y-%m-%d"),
            "end":   e.strftime("%Y-%m-%d"),
            "predicted_change_lb": round(predicted_change, 2),
            "actual_change_lb":    round(actual_change, 2),
            "days": int((e - s).days),
        })

    # linear regression over windows
    if len(windows) >= 3:
        x = np.array([w["predicted_change_lb"] for w in windows])
        y = np.array([w["actual_change_lb"]    for w in windows])
        slope, intercept = np.polyfit(x, y, 1)
        ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = round(1 - ss_res / ss_tot, 3) if ss_tot > 0 else 0.0
        # regression line for the scatter plot
        x_line = [float(x.min()), float(x.max())]
        y_line = [round(slope * v + intercept, 2) for v in x_line]
    else:
        slope, intercept, r2 = 1.0, 0.0, 0.0
        x_line, y_line = [], []

    logged_days = int(df["consumedKilocalories"].notna().sum())
    total_days  = len(df)

    out = {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "cumulative_predicted_lb": to_list(df["cumulative_predicted_lb"]),
        "cumulative_actual_lb":    to_list(df["cumulative_actual_lb"]),
        "error_kcal":              to_list(df["error_kcal"], 0),
        "rolling_error_kcal":      to_list(df["rolling_error_kcal"], 0),
        "weigh_in_windows": windows,
        "regression_line": {"x": x_line, "y": y_line},
        "regression": {
            "slope":     round(float(slope), 3),
            "intercept": round(float(intercept), 3),
            "r2":        round(float(r2), 3),
        },
        "summary": {
            "start_weight_lb":          round(start_weight, 1),
            "total_days":               total_days,
            "logged_days":              logged_days,
            "coverage_pct":             round(100 * logged_days / total_days, 1),
            "total_predicted_change_lb": round(float(df["cumulative_predicted_lb"].iloc[-1]), 1),
            "total_actual_change_lb":    round(float(df["cumulative_actual_lb"].iloc[-1]), 1),
            "final_error_lb":            round(float(df["error_lb"].iloc[-1]), 1),
        },
    }

    with open("src/app/data.json", "w") as f:
        json.dump(out, f)

    s = out["summary"]
    r = out["regression"]
    print(f"✓  Wrote src/app/data.json")
    print(f"   Period:     {out['dates'][0]} → {out['dates'][-1]}  ({total_days} days, {logged_days} logged)")
    print(f"   Predicted:  {s['total_predicted_change_lb']:+.1f} lb")
    print(f"   Actual:     {s['total_actual_change_lb']:+.1f} lb")
    print(f"   Regression: slope={r['slope']:.3f}  intercept={r['intercept']:.3f}  R²={r['r2']:.3f}")


if __name__ == "__main__":
    build()
