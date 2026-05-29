import pandas as pd
import numpy as np

train_path = r"d:\Hackathon Projects\Gridlock Hackathon 2.0 - Flipkart ( Team Sudarshan-IN )\dataset\train.csv"
test_path = r"d:\Hackathon Projects\Gridlock Hackathon 2.0 - Flipkart ( Team Sudarshan-IN )\dataset\test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

print("\n--- Train Info ---")
print(train.info())

print("\n--- Test Info ---")
print(test.info())

print("\n--- Missing Values in Train ---")
print(train.isnull().sum())

print("\n--- Missing Values in Test ---")
print(test.isnull().sum())

print("\n--- Unique Days ---")
print("Train days:", sorted(train['day'].unique()))
print("Test days:", sorted(test['day'].unique()))

print("\n--- Geohash Count ---")
print("Unique geohashes in train:", train['geohash'].nunique())
print("Unique geohashes in test:", test['geohash'].nunique())
overlap = len(set(train['geohash']).intersection(set(test['geohash'])))
print("Overlapping geohashes:", overlap)
print("Geohashes only in test:", len(set(test['geohash']) - set(train['geohash'])))

print("\n--- Timestamp Sample ---")
print("Train timestamps sample:", train['timestamp'].unique()[:10])
print("Train unique timestamps count:", train['timestamp'].nunique())
print("Test unique timestamps count:", test['timestamp'].nunique())

print("\n--- Categorical Feature Value Counts ---")
for col in ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']:
    print(f"\nValue counts for {col} in Train:")
    print(train[col].value_counts(dropna=False))
    print(f"Value counts for {col} in Test:")
    print(test[col].value_counts(dropna=False))

print("\n--- Target Demand Stats ---")
print(train['demand'].describe())
