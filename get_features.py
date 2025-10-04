import joblib
import pickle

# Load model and medians
model = pickle.load(open('model.pkl', 'rb'))
medians = joblib.load('medians.pkl')

print("All 42 model features with median values:")
print("=" * 50)

for i, feature in enumerate(model.get_booster().feature_names, 1):
    median_val = medians.get(feature, 0.0)
    print(f"{i:2d}. {feature}: {median_val}")

print(f"\nTotal features: {len(model.get_booster().feature_names)}")

