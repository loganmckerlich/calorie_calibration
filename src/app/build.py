"""
Run from project root: python src/app/build.py
Reads  data/processed/combined_daily_summary.csv
Writes src/app/data.json
"""
import json
import math
import os
import sys

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

    # ── Load VO2Max + FTP from biometrics JSON ─────────────────────────────
    bio_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "123302043_userBioMetrics.json")
    with open(bio_path) as f:
        bio_raw = json.load(f)
    vo2_by_date, ftp_by_date = {}, {}
    for r in bio_raw:
        date = r["metaData"]["calendarDate"][:10]
        if r.get("vo2MaxCycling"):
            vo2_by_date[date] = r["vo2MaxCycling"]
        if r.get("functionalThresholdPower"):
            ftp_by_date[date] = r["functionalThresholdPower"]
    vo2_sorted = sorted(vo2_by_date.items())
    ftp_sorted = sorted(ftp_by_date.items())

    # ── Weekly analysis ────────────────────────────────────────────────────
    # Each Mon-Sun window is independent: tracked deficit vs implied deficit
    # inferred from imputed weight change. Error is spread evenly over the week.
    wk_burned   = df["totalKilocalories"].resample("W-MON").sum()
    wk_consumed = df["consumedKilocaloriesImputed"].resample("W-MON").sum()
    wk_w_first  = df["weightImputed"].resample("W-MON").first()
    wk_w_last   = df["weightImputed"].resample("W-MON").last()
    wk_ndays    = df["totalKilocalories"].resample("W-MON").count()
    wk_active   = df["activeKilocalories"].resample("W-MON").sum()

    wk = pd.DataFrame({
        "burned":       wk_burned,
        "consumed":     wk_consumed,
        "weight_first": wk_w_first,
        "weight_last":  wk_w_last,
        "n_days":       wk_ndays,
        "active_kcal":  wk_active,
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

    wk["avg_active_pct"] = ((wk["active_kcal"] / wk["burned"]) * 100).round(1)
    corr_active_vs_error = round(float(wk["avg_active_pct"].corr(wk["daily_error_kcal"])), 2)

    # ── Cumulative predicted vs. actual weight change ──────────────────────
    # Both series start at 0 on the first week's Monday.
    wk["cum_predicted_lb"] = (wk["tracked_deficit_kcal"].cumsum() / 3500).round(2)
    wk["cum_actual_lb"]    = (wk["actual_deficit_kcal"].cumsum()  / 3500).round(2)

    # ── Rolling 4-week average daily error ────────────────────────────────
    wk["rolling_daily_error"] = wk["daily_error_kcal"].rolling(4, min_periods=2).mean().round(0)

    # ── Linear regression: actual_deficit ~ tracked_deficit ───────────────
    # actual = slope * tracked + intercept
    import numpy as np
    x = wk["tracked_deficit_kcal"].values
    y = wk["actual_deficit_kcal"].values
    slope_val, intercept_val = np.polyfit(x, y, 1)
    slope_val, intercept_val = float(slope_val), float(intercept_val)
    # r² of the fit
    y_hat  = slope_val * x + intercept_val
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_val = round(1 - ss_res / ss_tot, 3) if ss_tot else 0.0
    slope_val     = round(slope_val, 3)
    intercept_val = round(intercept_val, 0)
    # trendline endpoints spanning the data range
    x_min, x_max  = int(x.min()), int(x.max())
    trend_x = [x_min, x_max]
    trend_y = [round(slope_val * x_min + intercept_val, 0), round(slope_val * x_max + intercept_val, 0)]

    # ── Calibration factor ────────────────────────────────────────────────
    # What fraction of your tracked deficit actually shows up on the scale?
    total_tracked = float(wk["tracked_deficit_kcal"].sum())
    total_actual  = float(wk["actual_deficit_kcal"].sum())
    calibration_ratio   = round(total_actual / total_tracked, 3) if total_tracked else None
    # Implied daily over/under-log as % of tracked
    bias_pct            = round((1 - calibration_ratio) * 100, 1) if calibration_ratio else None
    # Suggested correction to logged intake (kcal/day)
    implied_correction  = int(avg_daily_error)   # same as avg_daily_error, surfaced differently

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
    df["steps_smooth"]              = df["totalSteps"].rolling(7, min_periods=3).mean()
    df["active_cal_smooth"]         = df["activeKilocalories"].rolling(7, min_periods=3).mean()
    df["bmr_cal_smooth"]            = df["bmrKilocalories"].rolling(7, min_periods=3).mean()

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
        "steps":                    to_list(df["totalSteps"].fillna(0), 0),
        "steps_smooth":             to_list(df["steps_smooth"], 0),
        "active_calories":          to_list(df["activeKilocalories"], 0),
        "bmr_calories":             to_list(df["bmrKilocalories"], 0),
        "active_cal_smooth":        to_list(df["active_cal_smooth"], 0),
        "bmr_cal_smooth":           to_list(df["bmr_cal_smooth"], 0),
        "vo2max": {
            "dates":  [d for d, _ in vo2_sorted],
            "values": [v for _, v in vo2_sorted],
        },
        "ftp": {
            "dates":  [d for d, _ in ftp_sorted],
            "values": [v for _, v in ftp_sorted],
        },
        # calibration page — weekly independent windows
        "weeks": {
            "labels":               wk_labels,
            "tracked_deficit_kcal": [int(v) for v in wk["tracked_deficit_kcal"]],
            "actual_deficit_kcal":  [int(v) for v in wk["actual_deficit_kcal"]],
            "error_kcal":           [int(v) for v in wk["error_kcal"]],
            "daily_error_kcal":     [int(v) for v in wk["daily_error_kcal"]],
            "avg_daily_burned":      [int(v) for v in wk["avg_daily_burned"]],
            "avg_daily_consumed":    [int(v) for v in wk["avg_daily_consumed"]],
            "avg_active_pct":        [float(v) for v in wk["avg_active_pct"]],
            "cum_predicted_lb":      [float(v) for v in wk["cum_predicted_lb"]],
            "cum_actual_lb":         [float(v) for v in wk["cum_actual_lb"]],
            "rolling_daily_error":   [None if (v is None or (isinstance(v, float) and math.isnan(v))) else int(v) for v in wk["rolling_daily_error"]],
            "n_days":               [int(v) for v in wk["n_days"]],
            "regression": {
                "slope":       slope_val,
                "intercept":   int(intercept_val),
                "r2":          r2_val,
                "trend_x":     trend_x,
                "trend_y":     trend_y,
            },
            "summary": {
                "avg_daily_error_kcal":    int(avg_daily_error),
                "median_daily_error_kcal": int(median_daily_error),
                "pct_weeks_over_tracked":  pct_weeks_over,
                "n_weeks":                 len(wk),
                "corr_burn_vs_error":      corr_burn,
                "corr_consumed_vs_error":  corr_consumed,
                "corr_active_pct_vs_error": corr_active_vs_error,
                "calibration_ratio":       calibration_ratio,
                "bias_pct":                bias_pct,
                "implied_correction_kcal": implied_correction,
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
