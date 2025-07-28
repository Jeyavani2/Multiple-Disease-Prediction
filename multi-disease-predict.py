#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# multi.py
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np # Needed for numerical types and fillna later

# --- Configuration for Feature Definitions and Model Paths ---
MODELS_DIR = 'models'

# A dictionary to easily map disease to its model path and feature configuration
# The 'all_cols_path' is CRUCIAL here. It points to the saved X_train.columns list.
DISEASE_CONFIGS = {
    "Kidney Disease": {
        "model_path": os.path.join(MODELS_DIR, 'kidney_pipeline.joblib'),
        "le_target_path": os.path.join(MODELS_DIR, 'le_kidney_target.joblib'), # Assuming you save kidney target encoder
        "X_train_cols_path": os.path.join(MODELS_DIR, 'kidney_X_train_cols.joblib'),
        # These lists help in creating the input form fields
        "input_features": {
            # These lists should ideally be dynamically extracted from the loaded X_train_cols,
            # or pre-defined based on your understanding of the dataset.
            # Example (adjust heavily based on actual cleaned kidney_disease.csv columns):
            'numerical': ['Age', 'BloodPressure', 'SpecificGravity', 'Albumin', 'Sugar', 
                          'BloodGlucoseRandom', 'BloodUrea', 'SerumCreatinine', 'Sodium', 
                          'Potassium', 'Hemoglobin', 'PackedCellVolume', 'WhiteBloodCellCount', 
                          'RedBloodCellCount'],
            'categorical': ['RedBloodCells', 'PusCell', 'PusCellClumps', 'Bacteria', 
                            'Hypertension', 'DiabetesMellitus', 'CoronaryArteryDisease', 
                            'Appetite', 'PedalEdema', 'Anemia']
        },
        "categorical_options": { # Define options for categorical inputs
            'RedBloodCells': ['normal', 'abnormal'],
            'PusCell': ['normal', 'abnormal'],
            'PusCellClumps': ['present', 'notpresent'],
            'Bacteria': ['present', 'notpresent'],
            'Hypertension': ['yes', 'no'],
            'DiabetesMellitus': ['yes', 'no'],
            'CoronaryArteryDisease': ['yes', 'no'],
            'Appetite': ['good', 'poor'],
            'PedalEdema': ['yes', 'no'],
            'Anemia': ['yes', 'no']
        }
    },
    "Parkinson's Disease": {
        "model_path": os.path.join(MODELS_DIR, 'parkinsons_pipeline.joblib'),
        "le_target_path": None, # Parkinsons target usually 0/1 directly
        "X_train_cols_path": os.path.join(MODELS_DIR, 'parkinsons_X_train_cols.joblib'),
        "input_features": {
            # Example (adjust heavily based on actual cleaned parkinsons.csv columns):
            'numerical': [
                'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
                'spread2', 'D2', 'PPE'
            ],
            'categorical': []
        },
        "categorical_options": {}
    },
    "Liver Disease": {
        "model_path": os.path.join(MODELS_DIR, 'liver_pipeline.joblib'),
        "le_target_path": None, # Liver target usually 0/1 directly (after remapping 1->1, 2->0)
        "le_gender_path": os.path.join(MODELS_DIR, 'le_liver_gender.joblib'), # Specific for Liver Gender
        "X_train_cols_path": os.path.join(MODELS_DIR, 'liver_X_train_cols.joblib'),
        "input_features": {
            # Example (adjust heavily based on actual cleaned indian_liver_patient.csv columns):
            'numerical': [
                'Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
                'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'
            ],
            'categorical': ['Gender'] # 'Gender' will be a string 'Male'/'Female' for input
        },
        "categorical_options": {
            'Gender': ['Male', 'Female']
        }
    }
}

# --- Utility Functions ---
@st.cache_resource # Cache models and other heavy objects
def load_and_cache_model(model_path):
    """Loads a pre-trained model pipeline and caches it."""
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at '{model_path}'. Please ensure it's in the '{MODELS_DIR}' directory.")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from '{model_path}': {e}")
        return None

@st.cache_data # Cache small data objects like column lists
def load_and_cache_cols(cols_path):
    """Loads and caches the list of expected columns."""
    if not os.path.exists(cols_path):
        st.error(f"Error: Column list file not found at '{cols_path}'. Please ensure it's in the '{MODELS_DIR}' directory.")
        return None
    try:
        cols = joblib.load(cols_path)
        return cols
    except Exception as e:
        st.error(f"Error loading column list from '{cols_path}': {e}")
        return None

# --- Load all models and column lists on app start ---
models_loaded = {}
expected_cols_loaded = {}
label_encoders_loaded = {}

for disease, config in DISEASE_CONFIGS.items():
    model = load_and_cache_model(config["model_path"])
    cols = load_and_cache_cols(config["X_train_cols_path"])
    
    models_loaded[disease] = model
    expected_cols_loaded[disease] = cols
    
    if "le_target_path" in config and config["le_target_path"] is not None:
        try:
            le_target = joblib.load(config["le_target_path"])
            label_encoders_loaded[disease] = le_target
        except FileNotFoundError:
            st.warning(f"Warning: Target LabelEncoder not found for {disease} at '{config['le_target_path']}'.")
        except Exception as e:
            st.warning(f"Warning: Error loading target LabelEncoder for {disease}: {e}")
    
    if "le_gender_path" in config and config["le_gender_path"] is not None: # For Liver Gender
        try:
            le_gender = joblib.load(config["le_gender_path"])
            label_encoders_loaded['Liver Gender'] = le_gender
        except FileNotFoundError:
            st.warning(f"Warning: Gender LabelEncoder not found for Liver at '{config['le_gender_path']}'.")
        except Exception as e:
            st.warning(f"Warning: Error loading Gender LabelEncoder for Liver: {e}")


# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Multiple Disease Prediction",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("Multiple Disease Prediction")
st.write("Enter the following details to predict the likelihood of a chronic disease.")

# --- Sidebar Model Status ---
st.sidebar.header("Model Status")
for disease, model in models_loaded.items():
    if model:
        st.sidebar.success(f"{disease} model loaded.")
    else:
        st.sidebar.error(f"{disease} model failed to load.")

# --- Sidebar Disease Selection ---
st.sidebar.header("Disease Prediction")
selected_disease = st.sidebar.radio(
    "Select Disease",
    list(DISEASE_CONFIGS.keys()) # Get disease names from config
)

# --- Main Content Area for Inputs ---
st.header("Patient Details & Test Results")

current_config = DISEASE_CONFIGS[selected_disease]
current_model = models_loaded[selected_disease]
current_expected_cols = expected_cols_loaded[selected_disease]

if current_model is None or current_expected_cols is None:
    st.warning(f"Cannot display input form for {selected_disease} as model or column config is missing.")
else:
    user_input_data = {}
    
    with st.form(key='disease_prediction_form'):
        st.subheader("Numerical Inputs")
        for feature in current_config['input_features']['numerical']:
            user_input_data[feature] = st.number_input(
                f"Enter {feature}",
                value=0.0, # Default value, adjust as needed or provide range
                format="%.2f",
                key=f"num_{selected_disease}_{feature}"
            )

        if current_config['input_features']['categorical']:
            st.subheader("Categorical Inputs")
            for feature in current_config['input_features']['categorical']:
                options = current_config['categorical_options'].get(feature, ['Unknown']) # Get options or default
                if len(options) > 2: # Use selectbox for multiple options
                    user_input_data[feature] = st.selectbox(
                        f"Select {feature}",
                        options,
                        key=f"cat_{selected_disease}_{feature}"
                    )
                else: # Use radio for binary options like Gender, Yes/No
                    user_input_data[feature] = st.radio(
                        f"Select {feature}",
                        options,
                        key=f"cat_{selected_disease}_{feature}"
                    )

        submit_button = st.form_submit_button(label='Predict Likelihood')

    # --- Prediction Logic ---
    if submit_button:
        if current_model is None:
            st.error(f"The model for {selected_disease} is not loaded. Please check the model files.")
        else:
            try:
                # Create a DataFrame from user inputs
                input_df = pd.DataFrame([user_input_data])

                # IMPORTANT: Align columns to match the exact order and presence of columns
                # from the training data (before ColumnTransformer in the pipeline).
                # This ensures the pipeline's preprocessor receives data in the expected format.
                final_input_df = pd.DataFrame(columns=current_expected_cols)
                for col in current_expected_cols:
                    if col in input_df.columns:
                        final_input_df[col] = input_df[col]
                    else:
                        # For missing columns (e.g., if a categorical value was not present in training
                        # but your X_train_cols lists pre-OHE column names).
                        # Or if a feature was truly missing from input (shouldn't happen with defined inputs).
                        # For now, fill with a placeholder, but careful handling is needed.
                        # For numeric, 0 or median might be appropriate. For categorical, an empty string or 'NaN'.
                        # The preprocessor in your pipeline should be robust to this (e.g., SimpleImputer for numerics, OneHotEncoder for categoricals).
                        final_input_df[col] = np.nan # Use NaN to let the pipeline's imputer (if any) handle it

                # Ensure correct dtypes for categorical columns (if not handled by NaN initially)
                for feature in current_config['input_features']['categorical']:
                    if feature in final_input_df.columns:
                        final_input_df[feature] = final_input_df[feature].astype(str) # Ensure it's string for OHE

                # Make prediction
                prediction = current_model.predict(final_input_df)[0]
                probability = current_model.predict_proba(final_input_df)[:, 1][0] # Probability of positive class (disease)

                # Get the predicted label (e.g., 'ckd' or 'notckd' for Kidney)
                predicted_label = ""
                if selected_disease in label_encoders_loaded:
                    predicted_label = label_encoders_loaded[selected_disease].inverse_transform([prediction])[0]
                elif selected_disease == "Liver Disease" and 'Liver Gender' in label_encoders_loaded:
                     # For Liver, the target is 0/1 directly in df, so no specific LE for target
                     # Instead, clarify what 0/1 means
                     predicted_label = "Liver Patient" if prediction == 1 else "No Liver Patient"
                elif selected_disease == "Parkinson's Disease":
                    predicted_label = "Has Parkinson's" if prediction == 1 else "No Parkinson's"
                else:
                    predicted_label = "Positive" if prediction == 1 else "Negative" # Default if no specific LE

                st.subheader("Prediction Results")
                st.write(f"The model predicts: **{predicted_label}**")
                st.write(f"Probability of {selected_disease} (Positive): **{probability:.2f}**")

                risk_level = ""
                if probability >= 0.75:
                    risk_level = "High Risk 🔴"
                elif probability >= 0.5:
                    risk_level = "Medium Risk 🟠"
                else:
                    risk_level = "Low Risk 🟢"
                st.write(f"Risk Level: **{risk_level}**")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info("Please ensure your input values are correct and match the expected format for the selected disease.")
                st.info("Also, verify that the saved `_X_train_cols.joblib` files correctly reflect the features and their order from your training data.")

