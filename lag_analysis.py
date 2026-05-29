import pandas as pd
import numpy as np

train_path = r"d:\Hackathon Projects\Gridlock Hackathon 2.0 - Flipkart ( Team Sudarshan-IN )\dataset\train.csv"
train = pd.read_csv(train_path)

df_48 = train[train['day'] == 48][['geohash', 'timestamp', 'demand']].rename(columns={'demand': 'demand_lag_1d'})
df_49 = train[train['day'] == 49][['geohash', 'timestamp', 'demand']]

merged = pd.merge(df_49, df_48, on=['geohash', 'timestamp'], how='inner')
print(f"Merged size: {len(merged)}")
correlation = merged['demand'].corr(merged['demand_lag_1d'])
print(f"Correlation between demand and demand_lag_1d on day 49 (minutes 0-120): {correlation:.4f}")


geohash_mean_48 = train[train['day'] == 48].groupby('geohash')['demand'].mean().rename('geohash_mean_48')
merged = pd.merge(merged, geohash_mean_48, on='geohash', how='left')
print(f"Correlation with geohash_mean_48: {merged['demand'].corr(merged['geohash_mean_48']):.4f}")


def get_minutes(ts):
    h, m = map(int, ts.split(':'))
    return h * 60 + m

train['minutes'] = train['timestamp'].apply(get_minutes)

start_48 = train[(train['day'] == 48) & (train['minutes'] <= 120)].groupby('geohash')['demand'].mean().rename('start_mean_48')
start_49 = train[(train['day'] == 49) & (train['minutes'] <= 120)].groupby('geohash')['demand'].mean().rename('start_mean_49')

start_merged = pd.merge(start_48, start_49, on='geohash', how='inner')
print(f"Correlation of start_mean between day 48 and day 49: {start_merged['start_mean_48'].corr(start_merged['start_mean_49']):.4f}")
print("Mean start demand day 48:", start_merged['start_mean_48'].mean())
print("Mean start demand day 49:", start_merged['start_mean_49'].mean())
