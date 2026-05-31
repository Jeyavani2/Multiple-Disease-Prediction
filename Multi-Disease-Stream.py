import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np


MODELS_DIR = 'models'


DISEASE_CONFIGS = {
    "Kidney Disease": {
        "model_path": os.path.join(MODELS_DIR, 'kidney_pipeline.joblib'),
        "le_target_path": os.path.join(MODELS_DIR, 'le_kidney_target.joblib'), # Assuming you save kidney target encoder
        "X_train_cols_path": os.path.join(MODELS_DIR, 'kidney_X_train_cols.joblib'),
        "positive_class_label": "ckd", # Explicitly define the positive class string
        "negative_class_label": "no ckd", # Explicitly define the negative class string
        "input_features": {
            'numerical': [
                {'name': 'Age', 'min': 0.0, 'max': 120.0, 'default': 40.0, 'format': "%.0f"},
                {'name': 'BloodPressure', 'min': 60.0, 'max': 200.0, 'default': 80.0, 'format': "%.1f"},
                {'name': 'SpecificGravity', 'min': 1.000, 'max': 1.035, 'default': 1.010, 'format': "%.3f"},
                {'name': 'Albumin', 'min': 0.0, 'max': 5.0, 'default': 0.0, 'format': "%.1f"}, # Assuming 0-5 scale for proteinuria
                {'name': 'Sugar', 'min': 0.0, 'max': 5.0, 'default': 0.0, 'format': "%.1f"}, # Assuming 0-5 scale for glycosuria
                {'name': 'BloodGlucoseRandom', 'min': 50.0, 'max': 600.0, 'default': 100.0, 'format': "%.1f"}, # Assuming mg/dL
                {'name': 'BloodUrea', 'min': 5.0, 'max': 200.0, 'default': 20.0, 'format': "%.1f"}, # Assuming mg/dL
                {'name': 'SerumCreatinine', 'min': 0.5, 'max': 15.0, 'default': 1.0, 'format': "%.1f"}, # Assuming mg/dL
                {'name': 'Sodium', 'min': 120.0, 'max': 160.0, 'default': 140.0, 'format': "%.1f"}, # Assuming mEq/L
                {'name': 'Potassium', 'min': 2.5, 'max': 7.5, 'default': 4.0, 'format': "%.1f"}, # Assuming mEq/L
                {'name': 'Hemoglobin', 'min': 5.0, 'max': 18.0, 'default': 14.0, 'format': "%.1f"}, # Assuming g/dL
                {'name': 'PackedCellVolume', 'min': 15.0, 'max': 60.0, 'default': 40.0, 'format': "%.1f"}, # Assuming %
                {'name': 'WhiteBloodCellCount', 'min': 1000.0, 'max': 20000.0, 'default': 7000.0, 'format': "%.0f"}, # Assuming cells/uL
                {'name': 'RedBloodCellCount', 'min': 2.0, 'max': 7.0, 'default': 4.5, 'format': "%.1f"} # Assuming millions/uL
            ],
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
        "positive_class_label": "Yes Parkinson Disease",
        "negative_class_label": "No Parkinson Disease",
        "input_features": {
            'numerical': [
                {'name': 'MDVP:Fo(Hz)', 'min': 80.0, 'max': 300.0, 'default': 150.0, 'format': "%.2f"},
                {'name': 'MDVP:Fhi(Hz)', 'min': 80.0, 'max': 600.0, 'default': 200.0, 'format': "%.2f"},
                {'name': 'MDVP:Flo(Hz)', 'min': 60.0, 'max': 250.0, 'default': 100.0, 'format': "%.2f"},
                {'name': 'MDVP:Jitter(%)', 'min': 0.0, 'max': 0.05, 'default': 0.005, 'format': "%.4f"},
                {'name': 'MDVP:Jitter(Abs)', 'min': 0.0, 'max': 0.001, 'default': 0.00005, 'format': "%.5f"},
                {'name': 'MDVP:RAP', 'min': 0.0, 'max': 0.03, 'default': 0.002, 'format': "%.4f"},
                {'name': 'MDVP:PPQ', 'min': 0.0, 'max': 0.03, 'default': 0.002, 'format': "%.4f"},
                {'name': 'Jitter:DDP', 'min': 0.0, 'max': 0.09, 'default': 0.006, 'format': "%.4f"},
                {'name': 'MDVP:Shimmer', 'min': 0.0, 'max': 0.1, 'default': 0.03, 'format': "%.4f"},
                {'name': 'MDVP:Shimmer(dB)', 'min': 0.0, 'max': 1.0, 'default': 0.25, 'format': "%.2f"},
                {'name': 'Shimmer:APQ3', 'min': 0.0, 'max': 0.06, 'default': 0.015, 'format': "%.4f"},
                {'name': 'Shimmer:APQ5', 'min': 0.0, 'max': 0.09, 'default': 0.02, 'format': "%.4f"},
                {'name': 'MDVP:APQ', 'min': 0.0, 'max': 0.1, 'default': 0.025, 'format': "%.4f"},
                {'name': 'Shimmer:DDA', 'min': 0.0, 'max': 0.15, 'default': 0.04, 'format': "%.4f"},
                {'name': 'NHR', 'min': 0.0, 'max': 0.5, 'default': 0.1, 'format': "%.3f"},
                {'name': 'HNR', 'min': 0.0, 'max': 40.0, 'default': 20.0, 'format': "%.2f"},
                {'name': 'RPDE', 'min': 0.3, 'max': 0.8, 'default': 0.5, 'format': "%.3f"},
                {'name': 'DFA', 'min': 0.5, 'max': 1.0, 'default': 0.7, 'format': "%.3f"},
                {'name': 'spread1', 'min': -10.0, 'max': 0.0, 'default': -5.0, 'format': "%.2f"},
                {'name': 'spread2', 'min': 0.0, 'max': 0.5, 'default': 0.2, 'format': "%.2f"},
                {'name': 'D2', 'min': 1.0, 'max': 4.0, 'default': 2.0, 'format': "%.2f"},
                {'name': 'PPE', 'min': 0.0, 'max': 0.5, 'default': 0.2, 'format': "%.2f"}
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
        "positive_class_label": "Yes Liver Disease",
        "negative_class_label": "No Liver Disease",
        "input_features": { 
            'numerical': [
                {'name': 'Age', 'min': 0.0, 'max': 90.0, 'default': 40.0, 'format': "%.0f"},
                {'name': 'Total_Bilirubin', 'min': 0.0, 'max': 70.0, 'default': 1.0, 'format': "%.1f"},
                {'name': 'Direct_Bilirubin', 'min': 0.0, 'max': 30.0, 'default': 0.5, 'format': "%.1f"},
                {'name': 'Alkaline_Phosphotase', 'min': 50.0, 'max': 2000.0, 'default': 180.0, 'format': "%.1f"},
                {'name': 'Alamine_Aminotransferase', 'min': 5.0, 'max': 500.0, 'default': 40.0, 'format': "%.1f"},
                {'name': 'Aspartate_Aminotransferase', 'min': 5.0, 'max': 500.0, 'default': 40.0, 'format': "%.1f"},
                {'name': 'Total_Protiens', 'min': 4.0, 'max': 10.0, 'default': 7.0, 'format': "%.1f"},
                {'name': 'Albumin', 'min': 1.0, 'max': 6.0, 'default': 3.5, 'format': "%.1f"},
                {'name': 'Albumin_and_Globulin_Ratio', 'min': 0.0, 'max': 2.0, 'default': 1.0, 'format': "%.2f"}
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
        return None # Return None if file not found
    try:
        cols = joblib.load(cols_path)
        return cols
    except Exception as e:
        st.error(f"Error loading column list from '{cols_path}': {e}")
        return None # Return None on error

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
            
    if "le_gender_path" in config and config["le_gender_path"] is not None: # Specific for Liver Gender
        try:
            le_gender = joblib.load(config["le_gender_path"])
            label_encoders_loaded['Liver Gender'] = le_gender # Store specifically for Liver's gender
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



# --- Sidebar Disease Selection ---
st.sidebar.header("Disease Prediction")
selected_disease = st.sidebar.radio(
    "Select Disease",
    list(DISEASE_CONFIGS.keys()) # Get disease names from config
)

# --- Main Content Area for Inputs ---
st.header("Patient Details & Test Results")

current_config = DISEASE_CONFIGS[selected_disease]
current_model = models_loaded.get(selected_disease) # Use .get to avoid KeyError if not loaded
current_expected_cols = expected_cols_loaded.get(selected_disease) # Use .get

if current_model is None or current_expected_cols is None:
    st.warning(f"Cannot display input form for {selected_disease} as model or column config is missing. Please check the 'models' directory.")
else:
    user_input_data = {}
    
    with st.form(key='disease_prediction_form'):
        st.subheader("Numerical Inputs")
        # Loop through numerical features defined in config, using min/max/default values
        for feature_info in current_config['input_features']['numerical']:
            user_input_data[feature_info['name']] = st.number_input(
                f"Enter {feature_info['name']}",
                min_value=float(feature_info['min']),
                max_value=float(feature_info['max']),
                value=float(feature_info['default']),
                format=feature_info['format'],
                key=f"num_{selected_disease}_{feature_info['name']}"
            )

        if current_config['input_features']['categorical']:
            st.subheader("Categorical Inputs")
            for feature_name in current_config['input_features']['categorical']:
                options = current_config['categorical_options'].get(feature_name, ['Unknown']) # Get options or default
                if len(options) > 2: # Use selectbox for multiple options
                    user_input_data[feature_name] = st.selectbox(
                        f"Select {feature_name}",
                        options,
                        key=f"cat_{selected_disease}_{feature_name}"
                    )
                else: # Use radio for binary options like Gender, Yes/No
                    user_input_data[feature_name] = st.radio(
                        f"Select {feature_name}",
                        options,
                        key=f"cat_{selected_disease}_{feature_name}"
                    )

        submit_button = st.form_submit_button(label='Predict Likelihood')

    # --- Prediction Logic ---
    if submit_button:
        if current_model is None:
            st.error(f"The model for {selected_disease} is not loaded. Please check the model files.")
            st.stop() 
        else:
            try:
                # Create a DataFrame from user inputs
                input_df = pd.DataFrame([user_input_data])
                

                # IMPORTANT: Align columns to match the exact order and presence of columns
                # from the training data (before ColumnTransformer in the pipeline).
                final_input_df = pd.DataFrame(columns=current_expected_cols)
               
                for col in current_expected_cols:
                    if col in input_df.columns:
                        #st.write(input_df[col])
                        final_input_df[col] = input_df[col]
                    else:
                        # Fill with NaN for missing columns; the pipeline's imputer should handle this.
                        # This ensures all expected columns are present.
                        #st.write('finalnonr')
                        final_input_df[col] = np.nan 
    
                # Ensure numerical columns are numeric type
                for feature_info in current_config['input_features']['numerical']:
                    feature_name = feature_info['name']
                    if feature_name in final_input_df.columns:
                        final_input_df[feature_name] = pd.to_numeric(final_input_df[feature_name], errors='coerce')


                # --- CRUCIAL FIX FOR LIVER GENDER ENCODING ---
                # Apply LabelEncoder for 'Gender' specifically for Liver Disease before passing to pipeline
                if selected_disease == "Liver Disease" and 'Gender' in final_input_df.columns:
                    if 'Liver Gender' in label_encoders_loaded:
                        le_gender = label_encoders_loaded['Liver Gender']
                        try:
                            # Transform the 'Gender' column from string ('Male'/'Female') to numerical (0/1)
                            final_input_df['Gender'] = le_gender.transform(final_input_df['Gender'])
                        except ValueError as ve:
                            st.error(f"Error encoding Gender for Liver Disease: {ve}. Ensure 'Male' and 'Female' are valid options.")
                            st.info("This might happen if the LabelEncoder was not trained on all possible gender values, or if an unexpected value was entered.")
                            st.stop() 
                    else:
                        st.error("Error: LabelEncoder for Liver Gender not loaded. Cannot process 'Gender' feature.")
                        st.stop() 

                # Ensure other categorical columns are string type, as expected by OneHotEncoder in the pipeline
                for feature_name in current_config['input_features']['categorical']:
                    if feature_name in final_input_df.columns:
                        if selected_disease == "Liver Disease" and feature_name == 'Gender':
                            continue # Gender already handled for Liver
                        final_input_df[feature_name] = final_input_df[feature_name].astype(str)

                # Get the raw numerical prediction (0 or 1)
                #st.write('ppppp')
                #st.write(final_input_df)
                raw_prediction_numeric = current_model.predict(final_input_df)[0]
               # st.write(raw_prediction_numeric)
                #raw_prediction_numeric = current_model.predict(final_input_df)
                
                # Get the predicted label string (e.g., 'ckd' or 'notckd')
                predicted_label_text = ""
                if selected_disease in label_encoders_loaded:
                    le_target = label_encoders_loaded[selected_disease]
                    predicted_label_text = le_target.inverse_transform([raw_prediction_numeric])[0]
                   # st.write(f'predicted label={predicted_label_text}')
                    if predicted_label_text == 'ckd':
                        predicted_label_text='Yes Ckd Disease'
                    else: 
                        predicted_label_text='No Ckd Disease'
                else:
                    # Fallback for models without a specific target encoder (e.g., Parkinson's, Liver if 1 is positive)
                    predicted_label_text = current_config["positive_class_label"] if raw_prediction_numeric == 1 else current_config["negative_class_label"]

                # Get the probabilities for all classes
               # st.write(final_input_df[0])
                probabilities = current_model.predict_proba(final_input_df)[0] 

                # --- CRITICAL FIX: Get the probability of the POSITIVE_CLASS_LABEL ('ckd') ---
                raw_probability_positive_class = 0.0 # Initialize
                if selected_disease in label_encoders_loaded:
                    le_target = label_encoders_loaded[selected_disease]
                    # Get the numerical index for the 'positive_class_label' (e.g., 'ckd')
                    try:
                        positive_class_index = le_target.transform([current_config["positive_class_label"]])[0]
                        raw_probability_positive_class = probabilities[positive_class_index]
                    except ValueError:
                        st.warning(f"Positive class label '{current_config['positive_class_label']}' not found in LabelEncoder classes for {selected_disease}. Falling back to default probability index 1.")
                        raw_probability_positive_class = probabilities[1] # Fallback
                else:
                    # For models without a specific target encoder (e.g., Parkinson's, Liver if 1 is positive)
                    raw_probability_positive_class = probabilities[1] # Assume 1 is positive class probability

                st.subheader("Prediction Results")
              

                risk_level = ""
                
                st.write(f"The model predicts: **{predicted_label_text}**")
                st.write(f"Probability of {selected_disease} (Positive): **{raw_probability_positive_class:.2f}**")
                if raw_probability_positive_class >= 0.75:
                     risk_level = "High Risk 🔴"
                elif raw_probability_positive_class >= 0.5:
                     risk_level = "Medium Risk 🟠"
                else:
                     risk_level = "Low Risk 🟢"
                
                st.write(f"Risk Level: **{risk_level}**")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info("Please ensure your input values are correct and match the expected format for the selected disease.")
                st.info("Also, verify that the saved `_X_train_cols.joblib` files correctly reflect the features and their order from your training data.")
