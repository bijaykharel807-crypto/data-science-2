import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from io import StringIO

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="ML Dashboard", layout="wide")

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Project Details", "Dataset", "EDA", "Prediction"]
)

# ---------- CONSTANTS ----------
MODEL_DIR = "saved_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

ALL_FILES = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
MODEL_FILES = [f for f in ALL_FILES if f != 'meta.pkl']

# ---------- FEATURE NAMES ----------
FEATURE_NAMES = [
    "Age", "SibSp", "FamilySize", "Parch", "Pclass",
    "Embarked", "Sex", "Fare", "IsAlone"
]
CATEGORICAL_FEATURES = ["Embarked", "Sex"]
NUMERIC_FEATURES = [f for f in FEATURE_NAMES if f not in CATEGORICAL_FEATURES]
TARGET_COLUMN = "Survived"

# ---------- HELPER: ensure derived features ----------
def ensure_derived_features(df):
    if 'FamilySize' not in df.columns and 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    if 'IsAlone' not in df.columns and 'FamilySize' in df.columns:
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df

# ---------- TRAINING FUNCTION ----------
def train_models(df, target_col='Survived'):
    """Train a set of classifiers on the given DataFrame and save them."""
    # Prepare features and target
    X = df[FEATURE_NAMES].copy()
    y = df[target_col]

    # Define preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ])

    # Split data (optional, but we keep it for validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }

    # Train and save each as a pipeline (preprocessor + model)
    for name, clf in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
        pipeline.fit(X_train, y_train)
        # Save the pipeline
        filename = f"{name}.pkl"
        joblib.dump(pipeline, os.path.join(MODEL_DIR, filename))

    # Save meta info (e.g., preprocessor, feature names, metrics)
    meta = {
        'preprocessor': preprocessor,
        'feature_names': FEATURE_NAMES,
        'target': target_col,
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, 'meta.pkl'))

    # Clear cached models so they are reloaded
    st.cache_resource.clear()
    st.success(f"✅ {len(models)} models trained and saved to `{MODEL_DIR}`.")

# ---------- MODEL LOADER ----------
@st.cache_resource
def load_model(filename):
    filepath = os.path.join(MODEL_DIR, filename)
    try:
        return joblib.load(filepath)
    except Exception:
        pass
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        pass
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f, encoding='latin1', fix_imports=True)
    except Exception:
        pass
    return None

@st.cache_resource
def load_all_models():
    models = {}
    for fname in MODEL_FILES:
        model = load_model(fname)
        if model is not None:
            models[fname] = model
    return models

@st.cache_resource
def load_meta():
    meta_path = os.path.join(MODEL_DIR, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    try:
        return joblib.load(meta_path)
    except:
        try:
            with open(meta_path, 'rb') as f:
                return pickle.load(f)
        except:
            return None

# ---------- PREPROCESSOR EXTRACTION ----------
def get_preprocessor_from_models(models):
    """Try to extract a preprocessor from saved pipelines."""
    for model in models.values():
        if isinstance(model, Pipeline) and hasattr(model, 'named_steps'):
            for step_name, step_obj in model.named_steps.items():
                if isinstance(step_obj, ColumnTransformer):
                    return step_obj
    return None

def get_preprocessor_only():
    """
    Return a preprocessor from meta.pkl or from a pipeline.
    Does NOT build a new preprocessor from the dataset (no retraining).
    """
    meta = load_meta()
    if meta and 'preprocessor' in meta:
        return meta['preprocessor']

    models = load_all_models()
    preprocessor = get_preprocessor_from_models(models)
    if preprocessor is not None:
        return preprocessor

    return None

# ---------- PAGES ----------

if page == "Project Details":
    st.title("📋 Project Details")
    st.markdown("""
    ### Titanic Survival Prediction

    This app serves pre‑trained classifiers to predict survival.

    - **Features**: Age, SibSp, FamilySize, Parch, Pclass, Embarked, Sex, Fare, IsAlone
      *FamilySize and IsAlone are derived automatically if you provide SibSp and Parch.*
    - **Target**: Survived (0 = No, 1 = Yes)
    - **Models**: Decision Tree, Random Forest, Gradient Boosting, KNN, Logistic Regression
    - **Best model**: `BEST_DecisionTree.pkl`

    **Models are trained on‑the‑fly** from the dataset you load.
    Use the sidebar to navigate.
    """)

elif page == "Dataset":
    st.title("📊 Dataset")
    st.write("Upload your CSV file to load the dataset.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = ensure_derived_features(df)
            st.session_state['df'] = df
            st.success("Dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    # Display dataset if loaded
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.subheader("Data Preview")
        st.dataframe(df.head(10))

        st.subheader("Dataset Info")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Shape**: {df.shape[0]} rows, {df.shape[1]} columns")
            st.write("**Columns**:", list(df.columns))
        with col2:
            st.write("**Missing Values**:")
            st.write(df.isnull().sum())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="dataset.csv", mime="text/csv")

        # ---------- TRAINING BUTTON ----------
        st.subheader("Model Training")
        if st.button("Train Models from this Dataset"):
            if TARGET_COLUMN not in df.columns:
                st.error(f"Target column '{TARGET_COLUMN}' not found in dataset.")
            else:
                # Check required features
                missing = [f for f in FEATURE_NAMES if f not in df.columns]
                if missing:
                    st.warning(f"Missing features: {missing}. Please ensure all required features are present.")
                else:
                    with st.spinner("Training models... This may take a moment."):
                        train_models(df)
                    # Rerun to refresh model list and clear caches
                    st.experimental_rerun()
    else:
        st.info("Please upload a CSV file to get started.")

elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    if 'df' not in st.session_state:
        st.warning("Please load data first (go to Dataset page).")
    else:
        df = st.session_state['df']
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(include='all'))

        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", ax=ax)
            st.pyplot(fig)

        if TARGET_COLUMN in df.columns:
            st.subheader(f"Target Distribution ({TARGET_COLUMN})")
            fig, ax = plt.subplots()
            df[TARGET_COLUMN].value_counts().plot(kind='bar', ax=ax)
            ax.set_xlabel(TARGET_COLUMN)
            ax.set_ylabel("Count")
            st.pyplot(fig)

        if len(df.columns) <= 8 and len(numeric_df.columns) > 1:
            st.subheader("Pairplot (numeric columns)")
            num_cols = numeric_df.columns[:5].tolist()
            if len(num_cols) > 1:
                fig = sns.pairplot(df[num_cols])
                st.pyplot(fig)

else:  # Prediction
    st.title("🎯 Make a Prediction")
    st.write("Enter passenger details and choose a model to predict survival.")

    models = load_all_models()
    if not models:
        st.warning("No models found. Please go to the Dataset page, load a dataset, and click 'Train Models from this Dataset'.")
        st.stop()

    # Load preprocessor from meta or pipeline
    preprocessor = get_preprocessor_only()
    if preprocessor is None:
        st.error("No preprocessor found. Please retrain models (go to Dataset page).")
        st.stop()

    model_choice = st.selectbox("Select a model", list(models.keys()))
    model = models[model_choice]

    st.subheader("Passenger Information")

    input_data = {}
    cols = st.columns(3)
    for i, feat in enumerate(FEATURE_NAMES):
        with cols[i % 3]:
            if feat in CATEGORICAL_FEATURES:
                if feat == "Sex":
                    options = ['male', 'female']
                elif feat == "Embarked":
                    options = ['C', 'Q', 'S']
                else:
                    options = ['Yes', 'No']
                input_data[feat] = st.selectbox(f"{feat}", options)
            else:
                if feat == "Age":
                    input_data[feat] = st.number_input(f"{feat}", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
                elif feat == "SibSp":
                    input_data[feat] = st.number_input(f"{feat}", min_value=0, max_value=10, value=0, step=1)
                elif feat == "Parch":
                    input_data[feat] = st.number_input(f"{feat}", min_value=0, max_value=10, value=0, step=1)
                elif feat == "FamilySize":
                    input_data[feat] = st.number_input(f"{feat}", min_value=0, max_value=15, value=1, step=1)
                elif feat == "Pclass":
                    input_data[feat] = st.selectbox(f"{feat}", [1, 2, 3])
                elif feat == "Fare":
                    input_data[feat] = st.number_input(f"{feat}", min_value=0.0, max_value=600.0, value=30.0, step=1.0)
                elif feat == "IsAlone":
                    input_data[feat] = st.selectbox(f"{feat}", [0, 1])
                else:
                    input_data[feat] = st.number_input(f"{feat}", value=0.0)

    input_df = pd.DataFrame([input_data])[FEATURE_NAMES]

    try:
        input_processed = preprocessor.transform(input_df)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    if st.button("Predict", type="primary"):
        try:
            prediction = model.predict(input_processed)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_processed)
                st.success(f"Prediction: **{prediction[0]}** (0 = No, 1 = Yes)")
                st.write("Class probabilities:", probs[0])
            else:
                st.success(f"Prediction: **{prediction[0]}**")
        except Exception as e:
            st.error(f"Prediction error: {e}")

    with st.expander("Model details"):
        st.write(f"Model: {model_choice}")
        st.write(f"Type: {type(model)}")