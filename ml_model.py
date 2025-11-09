import os
import joblib
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
# --- CHANGE 1: Import the imblearn Pipeline ---
from imblearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report
# --- CHANGE 2: Import SMOTE ---
from imblearn.over_sampling import SMOTE

# --- File Paths (Unchanged) ---
MODEL_PATH = 'maintenance_model.joblib'
IMPUTATION_PATH = 'imputation_values.joblib'
DATA_FILE = "vehicle_maintenance_data.csv"

# --- Custom Transformer (Unchanged) ---
class DateToDaysTransformer(BaseEstimator, TransformerMixin):
    """Calculates the number of days between a date column and a reference date."""
    def __init__(self, date_col, reference_date='2025-10-15'):
        self.date_col = date_col
        self.reference_date = pd.to_datetime(reference_date) 

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.date_col] = pd.to_datetime(X_copy[self.date_col], errors='coerce', dayfirst=True)
        time_delta = self.reference_date - X_copy[self.date_col]
        
        new_feature_name = self._get_new_feature_name()
        X_copy[new_feature_name] = time_delta.dt.days.fillna(9999) 
        return X_copy.drop(columns=[self.date_col])

    def get_feature_names_out(self, input_features=None):
        return [self._get_new_feature_name()]

    def _get_new_feature_name(self):
        if self.date_col == 'Last_Service_Date':
            return 'Days_Since_Last_Service'
        elif self.date_col == 'Warranty_Expiry_Date':
            return 'Days_Since_Warranty_Expiry'
        else:
            return 'Days_Since_' + self.date_col.replace('Date', '').strip().replace('_', '')

# --- Model Training Function (UPDATED) ---
def train_and_save_model():
    print("Attempting to load vehicle_maintenance_data.csv...")
    if not os.path.exists(DATA_FILE):
        print(f"FATAL ERROR: {DATA_FILE} not found.")
        raise FileNotFoundError(f"{DATA_FILE} not found.")

    df = pd.read_csv(DATA_FILE)
    print("Data loaded successfully.")

    leaky_columns = ['Brake_Condition', 'Battery_Status']
    df_train = df.drop(columns=leaky_columns)
    print(f"Dropped leaky columns for training: {leaky_columns}")

    X = df_train.drop('Need_Maintenance', axis=1)
    y = df_train['Need_Maintenance']

    # --- Feature Lists (Unchanged) ---
    M_HIST_ORDER = ['Poor', 'Average', 'Good']
    OWNER_ORDER = ['Third', 'Second', 'First'] 
    TIRE_ORDER = ['Worn Out', 'Good', 'New'] 

    ordinal_features = ['Maintenance_History', 'Owner_Type', 'Tire_Condition']
    ordinal_categories = [M_HIST_ORDER, OWNER_ORDER, TIRE_ORDER]
    date_features = ['Last_Service_Date', 'Warranty_Expiry_Date']
    numerical_features = [
        'Mileage', 'Reported_Issues', 'Vehicle_Age', 'Engine_Size', 
        'Odometer_Reading', 'Insurance_Premium', 'Service_History', 
        'Accident_History', 'Fuel_Efficiency'
    ]
    nominal_features = ['Vehicle_Model', 'Fuel_Type', 'Transmission_Type']
    
    # --- Preprocessor (Unchanged) ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('date_service', DateToDaysTransformer(date_col='Last_Service_Date'), ['Last_Service_Date']),
            ('date_warranty', DateToDaysTransformer(date_col='Warranty_Expiry_Date'), ['Warranty_Expiry_Date']),
            ('ord', OrdinalEncoder(
                categories=ordinal_categories, 
                handle_unknown='use_encoded_value', 
                unknown_value=-1
            ), ordinal_features),
            ('nom', OneHotEncoder(
                handle_unknown='ignore', 
                sparse_output=False
            ), nominal_features),
            ('scale', StandardScaler(), numerical_features)
        ],
        remainder='drop' 
    )

    # --- CHANGE 3: Add SMOTE to the pipeline ---
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # This step synthetically balances the training data
        ('smote', SMOTE(random_state=42)), 
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            # class_weight is no longer needed as SMOTE handles balancing
            # class_weight='balanced', 
            n_jobs=-1
        ))
    ])

    # --- Split data to get an accuracy score ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Starting model training with SMOTE for evaluation...")
    # When we call fit, SMOTE is ONLY applied to X_train, y_train
    # X_test remains unbalanced, which is what we want for a real test
    pipeline.fit(X_train, y_train)
    
    # --- Evaluate the model ---
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Model Evaluation (on 20% test data) ---")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report (with SMOTE):")
    print(classification_report(y_test, y_pred))
    print("------------------------------------------")

    # --- Retrain on 100% of data for production ---
    print("\nRetraining model on 100% of data with SMOTE for production...")
    pipeline.fit(X, y) # Retrain on all data
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Production model training complete and saved as '{MODEL_PATH}'.")

    # --- Imputation & Feature Importance (Unchanged) ---
    imputation_values = {
        'numerical': {col: X[col].mean() for col in numerical_features},
        'categorical': {col: X[col].mode()[0] for col in nominal_features + ordinal_features + date_features}
    }
    joblib.dump(imputation_values, IMPUTATION_PATH)
    print(f"Imputation values calculated and saved as '{IMPUTATION_PATH}'.")

    try:
        print("\n--- Model Feature Importances (Weights) ---")
        model = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        pd.set_option('display.max_rows', None)
        print(importance_df)
        pd.reset_option('display.max_rows')
    except Exception as e:
        print(f"Could not calculate feature importances: {e}")

    return pipeline, imputation_values

# --- Model Loading Function (Unchanged) ---
def load_model_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(IMPUTATION_PATH):
        print("Model files not found or incomplete. Retraining model...")
        pipeline, imputation_values = train_and_save_model()
    else:
        print("Loading existing model artifacts...")
        pipeline = joblib.load(MODEL_PATH)
        imputation_values = joblib.load(IMPUTATION_PATH)
    
    return pipeline, imputation_values

# --- Prediction Logic Function (Unchanged) ---
def get_prediction(input_data: dict, model, imputation_values: dict):
    data = input_data.copy()
    for col, value in imputation_values['numerical'].items():
        if data[col] is None or data[col] == 0:
            data[col] = value
    for col, value in imputation_values['categorical'].items():
        if data[col] is None or data[col] == "":
            data[col] = value
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0][prediction[0]])
    }

# --- Scheduling Logic Function (Unchanged) ---
def get_schedule_prediction(input_data: dict, model, imputation_values: dict):
    prediction_result = get_prediction(input_data, model, imputation_values)
    prediction = prediction_result['prediction']
    schedule_message = ""
    suggested_date = None
    if prediction == 0:
        schedule_message = "No immediate maintenance required based on current data."
    else:
        schedule_message = "Maintenance is predicted as required."
        try:
            last_service_str = input_data.get('Last_Service_Date')
            if last_service_str:
                last_service_date = date.fromisoformat(last_service_str)
                suggested_service_date = last_service_date + timedelta(days=180)
                schedule_message += f" Last service was {last_service_str}."
                suggested_date = suggested_service_date.isoformat()
            else:
                schedule_message += " Cannot suggest date as Last_Service_Date was not provided."
        except Exception as e:
            schedule_message += f" Error calculating schedule: {e}"
    prediction_result.update({
        "schedule_message": schedule_message,
        "suggested_service_date": suggested_date
    })
    return prediction_result