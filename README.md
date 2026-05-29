# 🚦 Gridlock Hackathon 2.0 - Flipkart

**Team:** Sudarshan-IN

**Objective:** Predict traffic demand across various geographical zones (geohashes) at different times of the day using advanced spatiotemporal machine learning techniques.

---

## 🛠️ Tools Used

This project was built utilizing standard and highly-optimized data science libraries:
* **Python 3.10+**: The core programming language.
* **Pandas**: Used heavily for dataset manipulation, temporal pivoting, aggregations, and dataframe merges.
* **NumPy**: Leveraged for fast mathematical operations, array broadcasting, and predictions clipping.
* **LightGBM (LGBMRegressor)**: The primary gradient boosting framework chosen for its extreme speed, memory efficiency, and native handling of categorical features.
* **Scikit-Learn**: Used primarily for calculating offline validation metrics (e.g., $R^2$ Score).
* **Jupyter Notebook**: Used for creating a clean, step-by-step reproducible pipeline cell-by-cell.

---

## 🎯 My Approach

My strategy revolves around modeling traffic as a highly periodic spatiotemporal system. Since traffic behaves rhythmically (recurring daily patterns), I identified that **historical lag from the previous day** at the exact same location and time is the strongest predictor of current demand. 

My pipeline follows a 3-step approach:
1. **Data Profiling & Spatial Mapping:** I first identified that the test set contained new geohashes (geographical locations) that were not present in the historical training data. This posed a severe "cold-start" problem.
2. **Robust Validation Loop:** I designed an offline validation sandbox mimicking the competition's exact time-split (training on the first hour of Day 49, validating on the remaining hours of Day 49).
3. **Hierarchical Modeling:** Instead of training multiple models, I engineered a single, robust LightGBM Regressor that relies on a progressive fallback mechanism, allowing it to predict traffic seamlessly even for locations it has never seen before.

---

## 🧬 Feature Engineering Details

Feature engineering was the most critical component of my success. I engineered three core groups of features:

### 1. Geohash Coordinate Decoding
Geohashes are Base-32 encoded strings representing grid boxes on the earth. By mathematically decoding these strings into continuous **`latitude`** and **`longitude`** coordinates, I allowed the LightGBM decision trees to infer spatial proximity, distance, and neighborhood clustering naturally.

### 2. Hierarchical Spatial Fallbacks (Lags)
To counter the "cold-start" problem (missing data for new test geohashes), I computed historical lag features (Day 48 demand) at progressively broader spatial granularities. If a highly specific feature is missing, the model seamlessly falls back to a broader average:
* **Level 1 (Highest Precision):** Exact Geohash match (`demand_lag_gh`).
* **Level 2 (Neighborhood):** 5-Character Geohash Prefix (`demand_lag_pref5`).
* **Level 3 (City Region):** 4-Character Geohash Prefix (`demand_lag_pref4`).
* **Level 4 (Global Trend):** Global Average across all locations for that specific timestamp (`demand_lag_global`).

### 3. Dynamic Temporal Scaling (`start_ratio`)
Because traffic baseline volume fluctuates day-to-day (due to weather, holidays, etc.), relying purely on yesterday's demand is suboptimal. I engineered a **`start_ratio`** feature:
* I calculate the average traffic demand in the *first hour* of Day 48 and compare it to the *first hour* of Day 49.
* This ratio (`start_mean_49 / start_mean_48`) acts as a real-time boundary condition, teaching the model to dynamically scale yesterday's historical lag up or down based on today's early morning macro-conditions.

---

## 📂 Repository Structure

| File / Folder | Purpose |
| :--- | :--- |
| 📁 `dataset/` | Contains the raw `train.csv`, `test.csv`, and `sample_submission.csv` datasets. |
| 🐍 `eda.py` | Exploratory Data Analysis profiling shapes, missing values, and geohash overlap. |
| 🐍 `time_analysis.py` | Temporal analysis script to convert standard `"HH:MM"` timestamps to elapsed daily minutes. |
| 🐍 `lag_analysis.py` | Spatiotemporal feature analysis script proving high demand correlation across days. |
| 🐍 `test_geohash.py` | The core mathematical decoder converting Base-32 strings into continuous lat/lon coordinates. |
| 🐍 `model_validation.py` | Offline sandbox mimicking the competition's exact train/validation temporal split. |
| 🐍 **`model_validation_robust.py`** | **The Production Pipeline.** Incorporates hierarchical fallbacks, trains the model, and predicts. |
| 📓 **`traffic_prediction.ipynb`** | A single Jupyter Notebook consolidating the entire robust pipeline. |

---

## 🚀 How to Run the Project

### 1. Prerequisites
Ensure you have Python installed, then install the required dependencies:
```bash
pip install pandas numpy lightgbm scikit-learn
```

### 2. Verify Dataset Location
Ensure your datasets are located in the `dataset/` directory at the project root.

### 3. Generate Predictions
**Option A: Command Line**
Run the robust validation script to train the model and generate the output:
```bash
python model_validation_robust.py
```
This will automatically generate a `submission.csv` file in the root directory.

**Option B: Jupyter Notebook**
Open `traffic_prediction.ipynb` in your preferred environment and execute the cells sequentially.
