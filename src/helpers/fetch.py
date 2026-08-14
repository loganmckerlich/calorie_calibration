import os
import json
import pandas as pd

def init_load():
    all_daily_summary = {}
    biometrics_parsed = {}
    keys_to_keep = ['totalKilocalories','activeKilocalories','bmrKilocalories','consumedKilocalories','remainingKilocalories','wellnessTotalKilocalories','wellnessActiveKilocalories','totalSteps','netCalorieGoal','isVigorousDay','minHeartRate','maxHeartRate','restingHeartRate']
    for file in os.listdir('data'):
        print(f"Loading {file}...")
        if file.endswith(".json") and file.startswith('UDS'):
            daily_summary = json.load(open(f'data/{file}'))
            daily_summary_parsed = {}
            for value in daily_summary:
                daily_summary_parsed[value['calendarDate']] = {k: v for k, v in value.items() if k in keys_to_keep}
            all_daily_summary.update(daily_summary_parsed)
        elif file.endswith("BioMetrics.json"):
            biometrics = json.load(open(f'data/{file}'))
            for value in biometrics:
                if 'weight' in value:
                    biometrics_parsed[value['metaData']['calendarDate'][0:10]] = round(value['weight']['weight'] * 0.00220462,1)


    df = pd.DataFrame.from_dict(all_daily_summary, orient='index').sort_index()
    weight_df = pd.DataFrame.from_dict(biometrics_parsed, orient='index', columns=['weight']).sort_index()
    df = df.merge(weight_df, how='left', left_index=True, right_index=True, validate='1:1', suffixes=('', '_weight'))
    df.index = pd.to_datetime(df.index)

    start_date = pd.to_datetime('2026-03-16') # first day of recent lock in
    end_date = pd.to_datetime('2026-08-13') # yesterday aka last full day

    df = df[df.index >= start_date]
    df = df[df.index <= end_date]
    
    print(f"How populated is calories consumed: {100*round(1-df.consumedKilocalories.isna().mean(), 4)}%")
    print(f"Max Weight: {df.weight.max()} on {df.weight.idxmax().date()}")
    print(f"Min Weight: {df.weight.min()} on {df.weight.idxmin().date()}")
    df.to_csv('data/processed/combined_daily_summary.csv')
    return df

def load():
    if not os.path.exists('data/processed/combined_daily_summary.csv'):
        df = init_load()
    else:
        df = pd.read_csv('data/processed/combined_daily_summary.csv', index_col=0, parse_dates=True)
    return df

if __name__ == "__main__":
    df = load()
    print(df.head())