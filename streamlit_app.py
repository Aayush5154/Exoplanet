import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Exoplanet Prediction Model",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and preprocessing files
@st.cache_resource
def load_model():
    try:
        with open("backend/model.pkl", "rb") as f:
            model = pickle.load(f)
        medians = joblib.load("backend/medians.pkl")
        category_mappings = joblib.load("backend/category_mappings.pkl")
        return model, medians, category_mappings
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

# Load the model
model, medians, category_mappings = load_model()

if model is None:
    st.error("Model files not found. Please ensure the backend files are in the correct location.")
    st.stop()

# Navigation
st.sidebar.title("🚀 Exoplanet Prediction")
page = st.sidebar.selectbox(
    "Choose Input Method",
    ["Manual Input", "CSV Upload", "About"]
)

# Suggested nearest values based on your images
suggested_values = {
    "koi_score": 0.33,
    "koi_fpflag_nt": 0.00,
    "koi_fpflag_ss": 0.00,
    "koi_fpflag_co": 0.00,
    "koi_fpflag_ec": 0.00,
    "koi_period": 10.01,
    "koi_period_err1": 0.00,
    "koi_period_err2": -0.00,
    "koi_time0bk": 137.52,
    "koi_time0bk_err1": 0.00,
    "koi_time0bk_err2": -0.00,
    "koi_impact": 0.54,
    "koi_impact_err1": 0.19,
    "koi_impact_err2": -0.21,
    "koi_duration": 3.79,
    "koi_duration_err1": 0.14,
    "koi_duration_err2": -0.14,
    "koi_depth": 421.10,
    "koi_depth_err1": 20.75,
    "koi_depth_err2": -20.75,
    "koi_prad": 2.39,
    "koi_prad_err1": 0.52,
    "koi_prad_err2": -0.30
}

# Feature display names
feature_display_names = {
    "koi_score": "Disposition Score",
    "koi_fpflag_nt": "Not Transit-Like False Positive Flag",
    "koi_fpflag_ss": "Stellar Eclipse False Positive Flag", 
    "koi_fpflag_co": "Centroid Offset False Positive Flag",
    "koi_fpflag_ec": "Ephemeris Contamination False Positive Flag",
    "koi_period": "Orbital Period [days]",
    "koi_period_err1": "Orbital Period Upper Unc. [days]",
    "koi_period_err2": "Orbital Period Lower Unc. [days]",
    "koi_time0bk": "Transit Epoch [BKJD]",
    "koi_time0bk_err1": "Transit Epoch Upper Unc. [BKJD]",
    "koi_time0bk_err2": "Transit Epoch Lower Unc. [BKJD]",
    "koi_impact": "Impact Parameter",
    "koi_impact_err1": "Impact Parameter Upper Unc.",
    "koi_impact_err2": "Impact Parameter Lower Unc.",
    "koi_duration": "Transit Duration [hrs]",
    "koi_duration_err1": "Transit Duration Upper Unc. [hrs]",
    "koi_duration_err2": "Transit Duration Lower Unc. [hrs]",
    "koi_depth": "Transit Depth [ppm]",
    "koi_depth_err1": "Transit Depth Upper Unc. [ppm]",
    "koi_depth_err2": "Transit Depth Lower Unc. [ppm]",
    "koi_prad": "Planetary Radius [Earth radii]",
    "koi_prad_err1": "Planetary Radius Upper Unc. [Earth radii]",
    "koi_prad_err2": "Planetary Radius Lower Unc. [Earth radii]"
}

def preprocess_data(df):
    """Preprocess the data using the same pipeline as the backend"""
    # Drop ID columns if they exist
    id_cols_to_drop = ["kepid", "kepoi_name", "kepler_name", "koi_pdisposition", "koi_disposition"]
    df = df.drop(columns=[col for col in id_cols_to_drop if col in df.columns], errors='ignore')
    
    # Fill missing values with medians
    for col, median_val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
    
    # Encode categorical features
    for col, mapping in category_mappings.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    
    # Ensure all model features are present
    for col in model.get_booster().feature_names:
        if col not in df.columns:
            df[col] = medians.get(col, 0)
    
    return df[model.get_booster().feature_names]

if page == "Manual Input":
    st.title("🚀 Exoplanet Prediction Model")
    st.write("Input the KOI features to predict if it is a confirmed exoplanet.")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    user_input = {}
    
    with col1:
        st.subheader("📊 Basic Parameters")
        for feature in list(suggested_values.keys())[:len(suggested_values)//2]:
            display_name = feature_display_names.get(feature, feature)
            placeholder_value = suggested_values.get(feature, 0.0)
            user_input[feature] = st.number_input(
                display_name,
                value=placeholder_value,
                format="%.2f",
                help=f"Suggested value: {placeholder_value}"
            )
    
    with col2:
        st.subheader("📈 Advanced Parameters")
        for feature in list(suggested_values.keys())[len(suggested_values)//2:]:
            display_name = feature_display_names.get(feature, feature)
            placeholder_value = suggested_values.get(feature, 0.0)
            user_input[feature] = st.number_input(
                display_name,
                value=placeholder_value,
                format="%.2f",
                help=f"Suggested value: {placeholder_value}"
            )
    
    # Add any missing features that the model expects
    model_features = model.get_booster().feature_names
    for feature in model_features:
        if feature not in user_input:
            user_input[feature] = medians.get(feature, 0.0)
    
    # Prediction button
    if st.button("🔮 Predict Exoplanet", type="primary"):
        try:
            # Create DataFrame from user input
            input_df = pd.DataFrame([user_input])
            
            # Preprocess the data
            processed_df = preprocess_data(input_df)
            
            # Make prediction
            prediction = model.predict(processed_df)[0]
            probability = model.predict_proba(processed_df)[0][1]
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            if prediction == 1:
                st.success(f"🌟 **EXOPLANET DETECTED!** 🌟")
                st.metric("Confidence Level", f"{probability:.2%}")
                st.balloons()
            else:
                st.warning("❌ **Not an Exoplanet**")
                st.metric("Confidence Level", f"{1-probability:.2%}")
            
            # Show detailed probability
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Exoplanet Probability", f"{probability:.2%}")
            with col2:
                st.metric("Non-Exoplanet Probability", f"{1-probability:.2%}")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

elif page == "CSV Upload":
    st.title("📁 CSV File Upload")
    st.write("Upload a CSV file containing KOI data for batch prediction.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with KOI data. The file should contain the required features."
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file, skiprows=53)  # Skip header rows like in backend
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} records.")
            
            if st.button("🔮 Analyze Data", type="primary"):
                with st.spinner("Processing data and making predictions..."):
                    # Preprocess the data
                    processed_df = preprocess_data(df.copy())
                    
                    # Make predictions
                    predictions = model.predict(processed_df)
                    probabilities = model.predict_proba(processed_df)[:, 1]
                    
                    # Calculate results
                    num_exoplanets = np.sum(predictions)
                    total_stars = len(predictions)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Objects", total_stars)
                    with col2:
                        st.metric("Confirmed Exoplanets", num_exoplanets)
                    with col3:
                        st.metric("Success Rate", f"{num_exoplanets/total_stars:.1%}")
                    
                    # Show detailed results
                    results_df = pd.DataFrame({
                        'Index': range(len(df)),
                        'Exoplanet Prediction': ['Yes' if p == 1 else 'No' for p in predictions],
                        'Confidence': [f"{prob:.2%}" for prob in probabilities]
                    })
                    
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="exoplanet_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

elif page == "About":
    st.title("ℹ️ About Exoplanet Prediction Model")
    
    st.markdown("""
    ## 🚀 Exoplanet Prediction Model
    
    This application uses machine learning to predict whether a Kepler Object of Interest (KOI) 
    is a confirmed exoplanet based on various stellar and planetary parameters.
    
    ### 🔬 Features Used
    - **Orbital Parameters**: Period, epoch, impact parameter
    - **Transit Properties**: Duration, depth, radius
    - **Stellar Properties**: Temperature, surface gravity, radius
    - **Quality Flags**: Various false positive indicators
    
    ### 🎯 Model Information
    - **Algorithm**: XGBoost Classifier
    - **Training Data**: Kepler mission data
    - **Features**: 20+ stellar and planetary parameters
    - **Accuracy**: High precision exoplanet detection
    
    ### 📊 How to Use
    1. **Manual Input**: Enter parameters manually with suggested values
    2. **CSV Upload**: Upload batch data for multiple predictions
    3. **Results**: Get confidence scores and detailed analysis
    
    ### 🌟 About Kepler Mission
    The Kepler space telescope discovered thousands of exoplanets by monitoring 
    the brightness of stars for periodic dimming caused by planetary transits.
    """)
    
    st.markdown("---")
    st.markdown("**Developed with ❤️ for exoplanet discovery**")


