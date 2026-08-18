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
    # (reindex not needed — dataframe already has a complete daily index)
    df = impute_calories(df)
    df = impute_weight(df)

    # ── Weekly analysis ────────────────────────────────────────────────────
    # Each Mon-Sun window is independent: tracked deficit vs implied deficit
    # inferred from imputed weight change. Error is spread evenly over the week.
    wk_burned   = df["totalKilocalories"].resample("W-MON").sum()
    wk_consumed = df["consumedKilocaloriesImputed"].resample("W-MON").sum()
    wk_w_first  = df["weightImputed"].resample("W-MON").first()
    wk_w_last   = df["weightImputed"].resample("W-MON").last()
    wk_ndays    = df["totalKilocalories"].resample("W-MON").count()

    wk = pd.DataFrame({
        "burned":       wk_burned,
        "consumed":     wk_consumed,
        "weight_first": wk_w_first,
        "weight_last":  wk_w_last,
        "n_days":       wk_ndays,
    }).dropna()
    wk = wk[wk["n_days"] >= 5]

    # Restrict to weeks anchored by real weigh-ins on both ends.
    # After the last weigh-in, weightImputed is ffill'd flat → actual_deficit = 0.
    first_weighin = df.index[df["weight"].notna()].min()
    last_weighin  = df.index[df["weight"].notna()].max()
    wk = wk[(wk.index >= first_weighin) & (wk.index <= last_weighin)]

    wk["tracked_deficit_kcal"] = (wk["burned"] - wk["consumed"]).round(0)
    wk["actual_loss_lb"]       = wk["weight_first"] - wk["weight_last"]
    wk["actual_deficit_kcal"]  = (wk["actual_loss_lb"] * 3500).round(0)
    # Drop flat-weight weeks — both real weigh-ins at period edges were identical
    # → interpolated flat → actual = 0 → no calibration signal.
    wk = wk[wk["actual_deficit_kcal"] != 0]

    # error > 0: tracked more deficit than scale shows (over-estimated deficit)
    wk["error_kcal"]           = wk["tracked_deficit_kcal"] - wk["actual_deficit_kcal"]
    wk["daily_error_kcal"]     = (wk["error_kcal"] / wk["n_days"]).round(0)

    wk["avg_daily_burned"]   = (wk["burned"]   / wk["n_days"]).round(0)
    wk["avg_daily_consumed"] = (wk["consumed"] / wk["n_days"]).round(0)

    avg_daily_error    = round(float(wk["daily_error_kcal"].mean()), 0)
    median_daily_error = round(float(wk["daily_error_kcal"].median()), 0)
    pct_weeks_over     = round(float((wk["error_kcal"] > 0).mean() * 100), 1)

    corr_burn     = round(float(wk["avg_daily_burned"].corr(wk["daily_error_kcal"])), 2)
    corr_consumed = round(float(wk["avg_daily_consumed"].corr(wk["daily_error_kcal"])), 2)

    wk_labels = [d.strftime("%b %d") for d in wk.index]

    # ── Overall cumulative totals (for summary cards) ──────────────────────
    start_weight = df["weightImputed"].iloc[0]
    end_weight   = df["weightImputed"].iloc[-1]
    actual_total_loss_lb    = round(start_weight - end_weight, 1)
    tracked_total_deficit   = round(float((df["totalKilocalories"] - df["consumedKilocaloriesImputed"]).sum()), 0)
    predicted_total_loss_lb = round(tracked_total_deficit / 3500, 1)

    logged_days = int(df["consumedKilocalories"].notna().sum())
    total_days  = len(df)

    # ── Health page rolling averages ───────────────────────────────────────
    df["hr_resting_smooth"]         = df["restingHeartRate"].rolling(7, min_periods=3).mean()
    df["hr_max_smooth"]             = df["maxHeartRate"].rolling(7, min_periods=3).mean()
    df["calories_burned_smooth"]    = df["totalKilocalories"].rolling(7, min_periods=3).mean()
    df["calories_consumed_smooth"]  = df["consumedKilocaloriesImputed"].rolling(7, min_periods=3).mean()

    out = {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        # health page
        "hr_resting":               to_list(df["restingHeartRate"], 0),
        "hr_max":                   to_list(df["maxHeartRate"], 0),
        "hr_resting_smooth":        to_list(df["hr_resting_smooth"], 1),
        "hr_max_smooth":            to_list(df["hr_max_smooth"], 1),
        "calories_burned":          to_list(df["totalKilocalories"], 0),
        "calories_consumed":        to_list(df["consumedKilocaloriesImputed"], 0),
        "calories_burned_smooth":   to_list(df["calories_burned_smooth"], 0),
        "calories_consumed_smooth": to_list(df["calories_consumed_smooth"], 0),
        "weight":                   to_list(df["weight"], 1),
        "weight_imputed":            to_list(df["weightImputed"], 1),
        # calibration page — weekly independent windows
        "weeks": {
            "labels":               wk_labels,
            "tracked_deficit_kcal": [int(v) for v in wk["tracked_deficit_kcal"]],
            "actual_deficit_kcal":  [int(v) for v in wk["actual_deficit_kcal"]],
            "error_kcal":           [int(v) for v in wk["error_kcal"]],
            "daily_error_kcal":     [int(v) for v in wk["daily_error_kcal"]],
            "avg_daily_burned":      [int(v) for v in wk["avg_daily_burned"]],
            "avg_daily_consumed":    [int(v) for v in wk["avg_daily_consumed"]],
            "n_days":               [int(v) for v in wk["n_days"]],
            "summary": {
                "avg_daily_error_kcal":    int(avg_daily_error),
                "median_daily_error_kcal": int(median_daily_error),
                "pct_weeks_over_tracked":  pct_weeks_over,
                "n_weeks":                 len(wk),
                "corr_burn_vs_error":      corr_burn,
                "corr_consumed_vs_error":  corr_consumed,
            },
        },
        "summary": {
            "start_weight_lb":          round(start_weight, 1),
            "total_days":               total_days,
            "logged_days":              logged_days,
            "coverage_pct":             round(100 * logged_days / total_days, 1),
            "tracked_total_deficit_kcal": int(tracked_total_deficit),
            "predicted_total_loss_lb":  predicted_total_loss_lb,
            "actual_total_loss_lb":     actual_total_loss_lb,
        },
    }

    with open("src/app/data.json", "w") as f:
        json.dump(out, f)

    s = out["summary"]
    ws = out["weeks"]["summary"]
    print(f"✓  Wrote src/app/data.json")
    print(f"   Period:      {out['dates'][0]} → {out['dates'][-1]}  ({total_days} days, {logged_days} logged)")
    print(f"   Predicted loss: {s['predicted_total_loss_lb']:+.1f} lb  |  Actual: {s['actual_total_loss_lb']:+.1f} lb")
    print(f"   Avg daily error: {ws['avg_daily_error_kcal']:+d} kcal/day  |  {ws['pct_weeks_over_tracked']}% of weeks over-tracked")


if __name__ == "__main__":
    build()
