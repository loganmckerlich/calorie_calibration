# functions to impute key values
import pandas as pd


def _assign_weekend_blocks(df):
    """Pairs each Saturday with the following Sunday into one block,
    identified by the Saturday's date."""
    dow = df.index.dayofweek
    is_weekend = pd.Series(dow, index=df.index).isin([5, 6])
    block_id = pd.Series(pd.NaT, index=df.index)

    for d in df.index[dow == 5]:
        block_id.loc[d] = d
        sunday = d + pd.Timedelta(days=1)
        if sunday in df.index:
            block_id.loc[sunday] = d

    # lone Sundays at the very start of the dataframe with no matching Saturday
    unresolved = df.index[(dow == 6) & block_id.isna()]
    for d in unresolved:
        block_id.loc[d] = d - pd.Timedelta(days=1)

    return is_weekend, block_id


def _assign_weekday_blocks(df):
    """Groups Mon-Fri into one block per work week, identified by that
    week's Monday date."""
    dow = df.index.dayofweek
    is_weekday = pd.Series(dow, index=df.index).isin([0, 1, 2, 3, 4])
    block_id = pd.Series(pd.NaT, index=df.index)

    for d in df.index[dow == 0]:
        for offset in range(5):
            day = d + pd.Timedelta(days=offset)
            if day in df.index:
                block_id.loc[day] = d

    # weekdays at the start of the dataframe whose Monday isn't in range
    unresolved = df.index[is_weekday & block_id.isna()]
    for d in unresolved:
        monday = d - pd.Timedelta(days=int(dow[df.index.get_loc(d)]))
        block_id.loc[d] = monday

    return is_weekday, block_id


def _impute_by_block(df, col, is_target_day, block_id, n_context_blocks):
    """
    For each block (weekend pair or work week) with missing values,
    find the n_context_blocks nearest complete blocks before and after,
    average their totals, and distribute the shortfall across the
    missing day(s) in the block.
    """
    values = df[col]
    imputed = values.copy()

    frame = pd.DataFrame({"value": values, "block": block_id, "is_target": is_target_day})
    frame = frame[frame["is_target"] & frame["block"].notna()]

    block_order = sorted(frame["block"].unique())

    stats = {}
    for b in block_order:
        rows = frame[frame["block"] == b]
        known = rows["value"].dropna()
        stats[b] = {
            "total_known": known.sum(),
            "n_known": len(known),
            "n_days": len(rows),
            "complete": len(known) == len(rows),
        }

    for i, b in enumerate(block_order):
        s = stats[b]
        n_missing = s["n_days"] - s["n_known"]
        if n_missing == 0:
            continue

        before = [x for x in block_order[:i] if stats[x]["complete"]][-n_context_blocks:]
        after = [x for x in block_order[i + 1:] if stats[x]["complete"]][:n_context_blocks]
        context = before + after
        if not context:
            continue  # no complete blocks nearby to reference, leave as NaN

        avg_total = sum(stats[c]["total_known"] for c in context) / len(context)
        fill_value = (avg_total - s["total_known"]) / n_missing

        target_rows = df.index[(block_id == b) & is_target_day & values.isna()]
        imputed.loc[target_rows] = fill_value

    return imputed


def impute_calories(df, col="consumedKilocalories",
                     weekend_context_blocks=2, weekday_context_blocks=1):
    """
    Imputes missing values in `col` separately for weekends and weekdays,
    matching each incomplete block's total to the average total of the
    nearest complete blocks around it.

    weekend_context_blocks: how many weekend blocks before/after to average
        (default 2, i.e. 2 weekends before + 2 after).
    weekday_context_blocks: how many work-week blocks before/after to average
        (default 1, i.e. the 5 weekdays immediately before + 5 immediately after).
    """
    df = df.copy()
    is_weekend, weekend_block_id = _assign_weekend_blocks(df)
    is_weekday, weekday_block_id = _assign_weekday_blocks(df)

    weekend_imputed = _impute_by_block(df, col, is_weekend, weekend_block_id, weekend_context_blocks)
    weekday_imputed = _impute_by_block(df, col, is_weekday, weekday_block_id, weekday_context_blocks)

    imputed_col = col + "Imputed"
    df[imputed_col] = df[col]
    df.loc[is_weekend, imputed_col] = weekend_imputed[is_weekend]
    df.loc[is_weekday, imputed_col] = weekday_imputed[is_weekday]
    return df

def impute_weight(df, col="weight"):
    df = df.copy()
    new_name = col + "Imputed"
    df[new_name] = df[col].interpolate(method="linear", limit_area="inside")
    df[new_name] = df[new_name].ffill() # assume last weight in to end of time
    df[new_name] = df[new_name].round(1) # assume first weight in to start of time
    return df