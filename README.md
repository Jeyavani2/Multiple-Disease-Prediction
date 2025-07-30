

# Project: Multi-Disease Prediction System

This repository contains code for a multi-disease prediction system, likely focusing on diseases such as Kidney Disease, with the potential for expansion to other diseases like Pakinson Disease and Liver Disease. The system utilizes machine learning models, specifically Gradient Boosting Classifier as shown in the Kidney Disease model training.

## Files:

* **`Multi-Disease-Data.py`**: This Jupyter Notebook appears to contain the training and evaluation logic for the machine learning models. It shows an example of training a Kidney Disease model using Logistic Regression, including GridSearchCV for hyperparameter tuning and evaluation metrics such as Accuracy, Precision, Recall, F1-Score, and AUC-ROC. It also outlines numerical and categorical features used for the Kidney Disease model.
* **`Multi-Disease-Stream.py`**: This Python script seems to be a Streamlit application that provides a user interface for predicting diseases. It loads pre-trained models and their associated configurations (e.g., target encoders, feature columns) for different diseases from a `models` directory. The script defines a `DISEASE_CONFIGS` dictionary to manage paths and labels for various disease models, with "Kidney Disease" being an example.

## Setup and Installation:

To run this project, you will need to install the required Python packages. You can do this using `pip` and the `requirements.txt` file.

### `requirements.txt`

streamlit
pandas
joblib
numpy
scikit-learn
matplotlib
seaborn


To install the dependencies, navigate to the project directory in your terminal and run:

```bash
pip install -r requirements.txt
Usage:
Model Training (if applicable):
Execute the Multi-Disease-Data.ipynb notebook to train the machine learning models. Ensure you have the necessary data files for training.

Running the Streamlit Application:
Once the models are trained and saved (or if pre-trained models are provided in the models directory), you can run the Streamlit application using the following command:

Bash

streamlit run Multi-Disease-Stream.py
This will open the application in your web browser, allowing you to interact with the multi-disease prediction system.
