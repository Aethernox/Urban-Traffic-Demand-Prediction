import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score

# Load datasets
train_path = r"d:\Hackathon Projects\Gridlock Hackathon 2.0 - Flipkart ( Team Sudarshan-IN )\dataset\train.csv"
train = pd.read_csv(train_path)

def get_minutes(ts):
    h, m = map(int, ts.split(':'))
    return h * 60 + m

train['minutes'] = train['timestamp'].apply(get_minutes)
train['hour'] = train['minutes'] // 60

# Decoded geohash coords
BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
BASE32_MAP = {c: i for i, c in enumerate(BASE32)}

def decode_geohash(geohash):
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    is_even = True
    for chars in geohash:
        if chars not in BASE32_MAP:
            return None, None
        val = BASE32_MAP[chars]
        for mask in [16, 8, 4, 2, 1]:
            if is_even: # longitude
                mid = (lon_interval[0] + lon_interval[1]) / 2.0
                if val & mask:
                    lon_interval = (mid, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], mid)
            else: # latitude
                mid = (lat_interval[0] + lat_interval[1]) / 2.0
                if val & mask:
                    lat_interval = (mid, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], mid)
            is_even = not is_even
    lat = (lat_interval[0] + lat_interval[1]) / 2.0
    lon = (lon_interval[0] + lon_interval[1]) / 2.0
    return lat, lon

train['lat'], train['lon'] = zip(*train['geohash'].apply(decode_geohash))
train['geohash_pref5'] = train['geohash'].str[:5]
train['geohash_pref4'] = train['geohash'].str[:4]

# Fill missing values
train['RoadType'] = train['RoadType'].fillna('Missing')
train['Weather'] = train['Weather'].fillna('Missing')
train['Temperature'] = train['Temperature'].fillna(train['Temperature'].median())

# Encode categoricals to category dtype for LightGBM
cat_cols = ['geohash', 'geohash_pref5', 'geohash_pref4', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
for c in cat_cols:
    train[c] = train[c].astype('category')

# Create day 48 features for reference
df_48 = train[train['day'] == 48][['geohash', 'timestamp', 'demand']].rename(columns={'demand': 'demand_lag_1d'})
geohash_mean_48 = train[train['day'] == 48].groupby('geohash', observed=False)['demand'].mean().rename('geohash_mean_48')
start_48 = train[(train['day'] == 48) & (train['minutes'] <= 120)].groupby('geohash', observed=False)['demand'].mean().rename('start_mean_48')

# Now let's build the validation set using day 49
df_49 = train[train['day'] == 49].copy()
df_49 = pd.merge(df_49, df_48, on=['geohash', 'timestamp'], how='left')
df_49 = pd.merge(df_49, geohash_mean_48, on='geohash', how='left')
df_49 = pd.merge(df_49, start_48, on='geohash', how='left')

# Start mean day 49 (from minutes <= 120)
start_49 = train[(train['day'] == 49) & (train['minutes'] <= 120)].groupby('geohash', observed=False)['demand'].mean().rename('start_mean_49')
df_49 = pd.merge(df_49, start_49, on='geohash', how='left')

# Start ratio
df_49['start_ratio'] = df_49['start_mean_49'] / (df_49['start_mean_48'] + 1e-5)

# For validation on day 49:
# Train on day 49 minutes <= 60, validate on day 49 minutes > 60
val_train = df_49[df_49['minutes'] <= 60]
val_test = df_49[df_49['minutes'] > 60]

print(f"Validation train size: {len(val_train)}")
print(f"Validation test size: {len(val_test)}")

# 1. Model using Lags
features_lag = [
    'geohash', 'geohash_pref5', 'geohash_pref4', 'lat', 'lon',
    'minutes', 'hour', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Temperature', 'Weather',
    'demand_lag_1d', 'geohash_mean_48', 'start_mean_48', 'start_mean_49', 'start_ratio'
]

X_tr_lag = val_train[features_lag]
y_tr_lag = val_train['demand']
X_va_lag = val_test[features_lag]
y_va_lag = val_test['demand']

model_lag = lgb.LGBMRegressor(random_state=42, n_estimators=100, verbose=-1)
model_lag.fit(X_tr_lag, y_tr_lag)
preds_lag = model_lag.predict(X_va_lag)
r2_lag = r2_score(y_va_lag, preds_lag)
print(f"R2 score of Lag model on validation set: {r2_lag * 100:.4f}")

# 2. General Model (without day-level lags, trained on day 48 + day 49 start)
features_gen = [
    'geohash', 'geohash_pref5', 'geohash_pref4', 'lat', 'lon',
    'minutes', 'hour', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Temperature', 'Weather'
]

df_train_all = pd.concat([train[train['day'] == 48], df_49[df_49['minutes'] <= 60]])
X_tr_gen = df_train_all[features_gen]
y_tr_gen = df_train_all['demand']
X_va_gen = val_test[features_gen]
y_va_gen = val_test['demand']

model_gen = lgb.LGBMRegressor(random_state=42, n_estimators=100, verbose=-1)
model_gen.fit(X_tr_gen, y_tr_gen)
preds_gen = model_gen.predict(X_va_gen)
r2_gen = r2_score(y_va_lag, preds_gen)
print(f"R2 score of General model on validation set: {r2_gen * 100:.4f}")

# 3. Simple Lag Baseline (predicting demand_lag_1d scaled by global or local start ratio)
preds_baseline_scaled = val_test['demand_lag_1d'] * val_test['start_ratio']
# fill NaNs in baseline with overall mean or mean lag
preds_baseline_scaled = preds_baseline_scaled.fillna(val_test['demand_lag_1d'])
preds_baseline_scaled = preds_baseline_scaled.fillna(y_tr_lag.mean())
r2_baseline = r2_score(y_va_lag, preds_baseline_scaled)
print(f"R2 score of Scaled Lag Baseline: {r2_baseline * 100:.4f}")
