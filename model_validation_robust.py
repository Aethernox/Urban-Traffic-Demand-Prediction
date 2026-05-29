import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
BASE32_MAP = {c: i for i, c in enumerate(BASE32)}

train_path = r"d:\Hackathon Projects\Gridlock Hackathon 2.0 - Flipkart ( Team Sudarshan-IN )\dataset\train.csv"
train = pd.read_csv(train_path)

def get_minutes(ts):
    h, m = map(int, ts.split(':'))
    return h * 60 + m

train['minutes'] = train['timestamp'].apply(get_minutes)
train['hour'] = train['minutes'] // 60
train['geohash_pref5'] = train['geohash'].str[:5]
train['geohash_pref4'] = train['geohash'].str[:4]

train['RoadType'] = train['RoadType'].fillna('Missing')
train['Weather'] = train['Weather'].fillna('Missing')
train['Temperature'] = train['Temperature'].fillna(train['Temperature'].median())

df_48_full = train[train['day'] == 48].copy()
df_49_full = train[train['day'] == 49].copy()


lag_gh = df_48_full[['geohash', 'timestamp', 'demand']].rename(columns={'demand': 'demand_lag_gh'})

lag_pref5 = df_48_full.groupby(['geohash_pref5', 'timestamp'])['demand'].mean().reset_index().rename(columns={'demand': 'demand_lag_pref5'})

lag_pref4 = df_48_full.groupby(['geohash_pref4', 'timestamp'])['demand'].mean().reset_index().rename(columns={'demand': 'demand_lag_pref4'})

lag_global = df_48_full.groupby('timestamp')['demand'].mean().reset_index().rename(columns={'demand': 'demand_lag_global'})


start_48_gh = df_48_full[df_48_full['minutes'] <= 120].groupby('geohash')['demand'].mean().rename('start_mean_gh_48')
start_48_pref5 = df_48_full[df_48_full['minutes'] <= 120].groupby('geohash_pref5')['demand'].mean().rename('start_mean_pref5_48')
start_48_pref4 = df_48_full[df_48_full['minutes'] <= 120].groupby('geohash_pref4')['demand'].mean().rename('start_mean_pref4_48')
start_48_global = df_48_full[df_48_full['minutes'] <= 120]['demand'].mean()

mean_48_gh = df_48_full.groupby('geohash')['demand'].mean().rename('mean_gh_48')
mean_48_pref5 = df_48_full.groupby('geohash_pref5')['demand'].mean().rename('mean_pref5_48')
mean_48_pref4 = df_48_full.groupby('geohash_pref4')['demand'].mean().rename('mean_pref4_48')
mean_48_global = df_48_full['demand'].mean()

start_49_gh = df_49_full[df_49_full['minutes'] <= 120].groupby('geohash')['demand'].mean().rename('start_mean_gh_49')
start_49_pref5 = df_49_full[df_49_full['minutes'] <= 120].groupby('geohash_pref5')['demand'].mean().rename('start_mean_pref5_49')
start_49_pref4 = df_49_full[df_49_full['minutes'] <= 120].groupby('geohash_pref4')['demand'].mean().rename('start_mean_pref4_49')
start_49_global = df_49_full[df_49_full['minutes'] <= 120]['demand'].mean()

def apply_hierarchical_features(df):
    out = df.copy()
    
    out = pd.merge(out, lag_gh, on=['geohash', 'timestamp'], how='left')
    out = pd.merge(out, lag_pref5, on=['geohash_pref5', 'timestamp'], how='left')
    out = pd.merge(out, lag_pref4, on=['geohash_pref4', 'timestamp'], how='left')
    out = pd.merge(out, lag_global, on='timestamp', how='left')

    out['demand_lag_1d'] = out['demand_lag_gh']
    out['demand_lag_1d'] = out['demand_lag_1d'].fillna(out['demand_lag_pref5'])
    out['demand_lag_1d'] = out['demand_lag_1d'].fillna(out['demand_lag_pref4'])
    out['demand_lag_1d'] = out['demand_lag_1d'].fillna(out['demand_lag_global'])
    
    out.drop(columns=['demand_lag_gh', 'demand_lag_pref5', 'demand_lag_pref4', 'demand_lag_global'], inplace=True)
    
    out = pd.merge(out, mean_48_gh, on='geohash', how='left')
    out = pd.merge(out, mean_48_pref5, on='geohash_pref5', how='left')
    out = pd.merge(out, mean_48_pref4, on='geohash_pref4', how='left')
    
    out['geohash_mean_48'] = out['mean_gh_48']
    out['geohash_mean_48'] = out['geohash_mean_48'].fillna(out['mean_pref5_48'])
    out['geohash_mean_48'] = out['geohash_mean_48'].fillna(out['mean_pref4_48'])
    out['geohash_mean_48'] = out['geohash_mean_48'].fillna(mean_48_global)
    
    out.drop(columns=['mean_gh_48', 'mean_pref5_48', 'mean_pref4_48'], inplace=True)
    
    out = pd.merge(out, start_48_gh, on='geohash', how='left')
    out = pd.merge(out, start_48_pref5, on='geohash_pref5', how='left')
    out = pd.merge(out, start_48_pref4, on='geohash_pref4', how='left')
    
    out['start_mean_48'] = out['start_mean_gh_48']
    out['start_mean_48'] = out['start_mean_48'].fillna(out['start_mean_pref5_48'])
    out['start_mean_48'] = out['start_mean_48'].fillna(out['start_mean_pref4_48'])
    out['start_mean_48'] = out['start_mean_48'].fillna(start_48_global)
    
    out.drop(columns=['start_mean_gh_48', 'start_mean_pref5_48', 'start_mean_pref4_48'], inplace=True)
    
    out = pd.merge(out, start_49_gh, on='geohash', how='left')
    out = pd.merge(out, start_49_pref5, on='geohash_pref5', how='left')
    out = pd.merge(out, start_49_pref4, on='geohash_pref4', how='left')
    
    out['start_mean_49'] = out['start_mean_gh_49']
    out['start_mean_49'] = out['start_mean_49'].fillna(out['start_mean_pref5_49'])
    out['start_mean_49'] = out['start_mean_49'].fillna(out['start_mean_pref4_49'])
    out['start_mean_49'] = out['start_mean_49'].fillna(start_49_global)
    
    out.drop(columns=['start_mean_gh_49', 'start_mean_pref5_49', 'start_mean_pref4_49'], inplace=True)
    
    out['start_ratio'] = out['start_mean_49'] / (out['start_mean_48'] + 1e-5)
    
    return out

df_49_processed = apply_hierarchical_features(df_49_full)

val_train = df_49_processed[df_49_processed['minutes'] <= 60]
val_test = df_49_processed[df_49_processed['minutes'] > 60]

print(f"Validation train size: {len(val_train)}")
print(f"Validation test size: {len(val_test)}")
print("Missing values in validation test:")
print(val_test[['demand_lag_1d', 'geohash_mean_48', 'start_mean_48', 'start_mean_49', 'start_ratio']].isnull().sum())

def decode_geohash(geohash):
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    is_even = True
    for chars in geohash:
        if chars not in BASE32_MAP:
            return None, None
        val = BASE32_MAP[chars]
        for mask in [16, 8, 4, 2, 1]:
            if is_even:
                mid = (lon_interval[0] + lon_interval[1]) / 2.0
                if val & mask:
                    lon_interval = (mid, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], mid)
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2.0
                if val & mask:
                    lat_interval = (mid, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], mid)
            is_even = not is_even
    lat = (lat_interval[0] + lat_interval[1]) / 2.0
    lon = (lon_interval[0] + lon_interval[1]) / 2.0
    return lat, lon

for df in [val_train, val_test]:
    df['lat'], df['lon'] = zip(*df['geohash'].apply(decode_geohash))

cat_cols = ['geohash', 'geohash_pref5', 'geohash_pref4', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
for c in cat_cols:
    val_train[c] = val_train[c].astype('category')
    val_test[c] = val_test[c].astype('category')

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
print(f"\nR2 score of robust lag model on validation set: {r2_lag * 100:.4f}")


# FINAL TEST PREDICTION
print("\nGenerating final test predictions...")

test_path = r"d:\Hackathon Projects\Gridlock Hackathon 2.0 - Flipkart ( Team Sudarshan-IN )\dataset\test.csv"
test = pd.read_csv(test_path)


test['minutes'] = test['timestamp'].apply(get_minutes)
test['hour'] = test['minutes'] // 60

test['geohash_pref5'] = test['geohash'].str[:5]
test['geohash_pref4'] = test['geohash'].str[:4]

test['RoadType'] = test['RoadType'].fillna('Missing')
test['Weather'] = test['Weather'].fillna('Missing')
test['Temperature'] = test['Temperature'].fillna(train['Temperature'].median())

test_processed = apply_hierarchical_features(test)

test_processed['lat'], test_processed['lon'] = zip(*test_processed['geohash'].apply(decode_geohash))

for c in cat_cols:
    test_processed[c] = test_processed[c].astype('category')

final_train = df_49_processed.copy()

final_train['lat'], final_train['lon'] = zip(
    *final_train['geohash'].apply(decode_geohash)
)

for c in cat_cols:
    final_train[c] = final_train[c].astype('category')

final_train = df_49_processed.copy()


final_train['lat'], final_train['lon'] = zip(
    *final_train['geohash'].apply(decode_geohash)
)


for c in cat_cols:
    final_train[c] = final_train[c].astype('category')


X_final = final_train[features_lag]
y_final = final_train['demand']

X_test_final = test_processed[features_lag]


final_model = lgb.LGBMRegressor(
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1
)

final_model.fit(X_final, y_final)


final_preds = final_model.predict(X_test_final)

final_preds = np.maximum(final_preds, 0)

submission = pd.DataFrame({
    'Index': test['Index'],
    'demand': final_preds
})

# Save CSV
submission.to_csv("submission.csv", index=False)

print("\nsubmission.csv created successfully!")
print(submission.head())

print("\nSubmission shape:")
print(submission.shape)

print("\nTrain Demand Statistics:")
print(train['demand'].describe())

print("\nPrediction Statistics:")
print(submission['demand'].describe())