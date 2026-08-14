# Tracking Calibration Project

## Thesis
The body keeps score: I track calories pretty well, both burn and consume. But at the end of the day my body tracks them flawlessly. The measure of my bodys flawless tracking is my weight. In a way when I track calories I am predicting a weight change that I can compare to the true change in weight.

This. sets itself up well for experimental design.

## Concept

I log calories consumed and calories burned daily, and separately track body
weight. The standard rule of thumb is that a **3,500 kcal deficit ≈ 1 lb of
weight lost** (and vice versa for surplus). If my logging were perfectly
accurate, my predicted weight trend (derived from logged calorie balance)
should track my actual weight trend closely.

It probably doesn't.  Food logging and burn estimates both carry systematic
error. This project treats my own tracking as an instrument to be
calibrated: I compare **predicted** weight change (from logged data) against
**actual** weight change (from the scale) to find out whether I'm
consistently over or under-estimating something, and by how much. The end
goal is a correction factor I can apply going forward so my logged numbers
better reflect reality.

## Data Sources

| Signal | Source | Frequency | Notes |
|---|---|---|---|
| Calories consumed | food logging app export | daily with a few missing | None |
| Calories burned | wearable / app TDEE estimate | daily | bike estimate comes from watts which is quite accurate |
| Body weight | scale | every now and again | I weight myself pretty infrequently |

Keep the raw daily food log and burn log — don't pre-average anything before
export. Smoothing happens in analysis, not collection.

## Method

### 1. Compute predicted weight change
For any period (period based on weigh in frequency):

```
net_calories = sum(calories_in) - sum(calories_out)
predicted_change_lb = net_calories / 3500
```

### 2. Compute actual weight change
Raw daily weight is noisy (water, sodium, glycogen, GI contents — can easily
swing ±1–3 lb day to day). Don't compare raw daily weight to the prediction.

- Apply smoothing to the weight series first.
- Will have to be flexible smoothing due to infrequent and irregular weigh ins
- Compute actual change as `smoothed_weight[end] - smoothed_weight[start]`
  over the same window used for the calorie sum.

### 3. Compare
```
error_lb = actual_change_lb - predicted_change_lb
error_kcal = error_lb * 3500
```

Plot **cumulative predicted vs. cumulative actual** weight change over the
full tracking period, not just week-by-week deltas. A steady divergence
(consistent slope difference) is a systematic bias signal. Scattered,
non-directional error week to week is closer to noise.

### 4. Isolate the bias (as far as possible)
The comparison only tells you "logged data doesn't match outcome" — it
can't automatically tell you whether the error is in intake logging, burn
estimation, or both. Options to narrow it down:
- Look for periods where one side is more trustworthy than usual (e.g., a
  stretch of scale-and-container weighed meals = higher-confidence intake;
  or a stretch of very consistent activity = more reliable burn estimate)
  and see if the error shrinks in that window.
- If available, compare against an independent burn estimate (e.g., a
  different device/algorithm) to see if burn estimates disagree with each
  other by a similar margin to the error you're seeing.

### 5. Derive a calibration factor
Once you have a stable systematic error over several windows, express it
as a percentage or fixed offset:
- e.g., "actual deficit tracks ~12% smaller than logged deficit" →
  suggests under-logging intake or over-estimating burn by that margin.
- Apply this going forward as a correction to either logged intake, burn
  estimate, or the derived prediction — whichever is most likely the
  source (see step 4).

## Known Limitations / Confounders

- **3,500 kcal/lb is an approximation.** Actual energy density of
  gained/lost tissue varies with body composition; treat this as accurate
  to within roughly ±5–10%, not exact.
- **Burn estimates are not ground truth.** Wearable/app TDEE numbers have
  their own error bars, often larger than people assume. This method can't
  fully separate intake error from burn error without an independent check.
- **Lag effects.** Weight doesn't respond instantly to a day's calorie
  balance — use weekly-or-longer windows, not daily comparisons.
- **Non-caloric weight drivers.** Sodium, carb intake (glycogen + water),
  hormonal cycle, hydration, and GI contents at time of weighing all move
  the scale independent of energy balance — this is what the smoothing step
  is meant to average out, but short data windows are still vulnerable to it.
- **Regression toward a moving target.** Metabolic adaptation over long
  deficits/surpluses can shift true TDEE over time, which will look like a
  "calibration drift" even if your logging accuracy hasn't changed.

## Suggested Analysis Tools

- Plot cumulative predicted vs. actual weight change (line chart, two series).
- Linear regression of `actual_change_lb ~ predicted_change_lb`; slope ≠ 1
  or intercept ≠ 0 indicates systematic bias.
- Rolling-window error (e.g., 4-week rolling average of `error_kcal`) to see
  if bias is stable, trending, or noisy.
