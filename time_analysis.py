import pandas as pd
import numpy as np

train_path = r"d:\Hackathon Projects\Gridlock Hackathon 2.0 - Flipkart ( Team Sudarshan-IN )\dataset\train.csv"
test_path = r"d:\Hackathon Projects\Gridlock Hackathon 2.0 - Flipkart ( Team Sudarshan-IN )\dataset\test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


print("Train day 48 rows:", len(train[train['day'] == 48]))
print("Train day 49 rows:", len(train[train['day'] == 49]))
print("Test day 49 rows:", len(test[test['day'] == 49]))
def timestamp_to_minutes(ts):
    h, m = map(int, ts.split(':'))
    return h * 60 + m

train['minutes'] = train['timestamp'].apply(timestamp_to_minutes)
test['minutes'] = test['timestamp'].apply(timestamp_to_minutes)

train_48_min = train[train['day'] == 48]['minutes'].unique()
train_49_min = train[train['day'] == 49]['minutes'].unique()
test_49_min = test['minutes'].unique()

print("\nTrain day 48 timestamps:")
print(f"Count: {len(train_48_min)}")
print(f"Min minutes: {min(train_48_min)} ({min(train_48_min)//60}:{min(train_48_min)%60})")
print(f"Max minutes: {max(train_48_min)} ({max(train_48_min)//60}:{max(train_48_min)%60})")

print("\nTrain day 49 timestamps:")
print(f"Count: {len(train_49_min)}")
print(f"Min minutes: {min(train_49_min)} ({min(train_49_min)//60}:{min(train_49_min)%60})")
print(f"Max minutes: {max(train_49_min)} ({max(train_49_min)//60}:{max(train_49_min)%60})")

print("\nTest day 49 timestamps:")
print(f"Count: {len(test_49_min)}")
print(f"Min minutes: {min(test_49_min)} ({min(test_49_min)//60}:{min(test_49_min)%60})")
print(f"Max minutes: {max(test_49_min)} ({max(test_49_min)//60}:{max(test_49_min)%60})")

overlap_min = set(train_49_min).intersection(set(test_49_min))
print("\nOverlap timestamps between train day 49 and test day 49:", len(overlap_min))
print("Only in train day 49 minutes:", sorted(list(set(train_49_min) - set(test_49_min))))
print("Only in test day 49 minutes:", sorted(list(set(test_49_min) - set(train_49_min))))
