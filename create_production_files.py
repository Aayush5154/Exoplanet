import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import pickle

print("--- Starting FINAL Production File Creation ---")

# --- 1. Load Data ---
try:
    # Ensure this path is correct relative to where you run the script
    csv_file_path = r"data/KOI_cumulative_2025.09.21_07.29.50.csv"
    data = pd.read_csv(csv_file_path, skiprows=53)
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print(f"❌ ERROR: CSV file not found at '{csv_file_path}'. Please check the path.")
    exit()

# --- 2. Preprocessing Pipeline ---
target = "koi_disposition"
drop_cols = ["kepid", "kepoi_name", "kepler_name", "koi_pdisposition"]
data_clean = data.drop(columns=drop_cols, errors='ignore')
threshold = 0.3 * len(data_clean)
data_clean = data_clean.dropna(axis=1, thresh=threshold)
data_clean[target] = data_clean[target].map({
    "CONFIRMED": 1, "FALSE POSITIVE": 0, "CANDIDATE": 0
}).fillna(0)

X = data_clean.drop(columns=[target])
y = data_clean[target]

# --- 3. Save Medians and CREATE CATEGORY MAPPINGS ---
print("Processing and saving medians...")
numeric_cols = X.select_dtypes(include=np.number).columns
medians_dict = X[numeric_cols].median().to_dict()
joblib.dump(medians_dict, "medians.pkl")
print("✅ medians.pkl saved.")

# Fill missing numeric values
for col, median_val in medians_dict.items():
    X[col] = X[col].fillna(median_val)

print("Processing and saving category mappings...")
# THIS IS THE NEW, MORE RELIABLE METHOD
category_mappings = {}
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = X[col].astype(str)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    # Create a simple mapping dict { 'category_A': 0, 'category_B': 1 }
    mapping = {category: int(index) for index, category in enumerate(le.classes_)}
    category_mappings[col] = mapping

joblib.dump(category_mappings, "category_mappings.pkl") # New file name
print("✅ category_mappings.pkl saved.")

# --- 4. Train and Save the Model ---
print("Training XGBoost model...")
xgb = XGBClassifier(
    objective="binary:logistic", eval_metric="logloss",
    use_label_encoder=False, random_state=42,
    scale_pos_weight=len(y[y==0]) / len(y[y==1])
)
xgb.fit(X, y)
with open("model.pkl", "wb") as file:
    pickle.dump(xgb, file)
print("✅ model.pkl saved.")
print("\n--- All production files created successfully! ---")