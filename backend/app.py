from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import joblib

app = Flask(__name__, template_folder='../templates')
CORS(app)

# --- Load All Your Production Files ---
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")

    medians = joblib.load("medians.pkl")
    print("Medians loaded successfully!")

    # Load the new, simpler mapping file
    category_mappings = joblib.load("category_mappings.pkl")
    print("Category mappings loaded successfully!")

    MODEL_FEATURE_NAMES = model.get_booster().feature_names

except FileNotFoundError as e:
    print(f"Error loading production files: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

def preprocess_data(df):
    """
    FINAL, most robust preprocessing pipeline.
    """
    id_cols_to_drop = ["kepid", "kepoi_name", "kepler_name", "koi_pdisposition", "koi_disposition"]
    df = df.drop(columns=[col for col in id_cols_to_drop if col in df.columns], errors='ignore')

    # Convert all numeric columns to proper numeric types
    for col in df.columns:
        if col in medians:  # This is a numeric column
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(medians[col])
        elif col in category_mappings:  # This is a categorical column
            df[col] = df[col].astype(str)
            # Map known categories, fill unknown with 0
            df[col] = df[col].map(category_mappings[col]).fillna(0).astype(int)

    # Fill any remaining missing values with medians
    for col, median_val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)

    # Ensure all model features are present and numeric
    for col in MODEL_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = medians.get(col, 0)
        else:
            # Ensure numeric columns are properly typed
            if col in medians:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(medians[col])

    df = df[MODEL_FEATURE_NAMES]
    
    # Final check: ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    if model is None:
        return jsonify({'error': 'Model or preprocessing files are not loaded. Check server logs.'}), 500
    
    try:
        # Get data from request
        data = request.get_json()
        
        # Create DataFrame from the input data
        input_df = pd.DataFrame([data])
        
        # Preprocess the data
        processed_df = preprocess_data(input_df)
        
        # Make prediction
        prediction = model.predict(processed_df)[0]
        probability = model.predict_proba(processed_df)[0][1]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'is_exoplanet': bool(prediction == 1)
        })
        
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if model is None:
        return jsonify({'error': 'Model or preprocessing files are not loaded. Check server logs.'}), 500
    
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400

    try:
        df_raw = pd.read_csv(file, skiprows=53)
        df_for_plot = df_raw.copy()
        df_processed = preprocess_data(df_raw)
        
        predictions = model.predict(df_processed)
        num_exoplanets = np.sum(predictions)
        total_stars = len(predictions)

        # Handle NaN values by filtering them out for the chart
        # Keep only rows where both x and y values are not NaN
        valid_data = df_for_plot[['koi_period', 'koi_prad']].dropna()
        x_data = valid_data['koi_period'].tolist()
        y_data = valid_data['koi_prad'].tolist()
        
        chart_data = {
            'x': x_data,
            'y': y_data,
            'mode': 'markers', 
            'type': 'scatter'
        }
        chart_layout = {
            'title': 'Planetary Radius vs. Orbital Period',
            'xaxis': {'title': 'Orbital Period [days]'},
            'yaxis': {'title': 'Planetary Radius [Earth radii]', 'type': 'log'}
        }
        result = {
            'summary': f"Found {num_exoplanets} CONFIRMED exoplanet(s) out of {total_stars} objects analyzed.",
            'num_exoplanets': int(num_exoplanets),
            'total_stars': int(total_stars),
            'success_rate': f"{num_exoplanets/total_stars:.1%}" if total_stars > 0 else "0%",
            'chart_data': chart_data,
            'chart_layout': chart_layout
        }
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)