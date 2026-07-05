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
    ["Project Details", "Dataset", "EDA", "Prediction"]   # Model Comparison removed
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

# ---------- PREPROCESSOR EXTRACTION (NO RETRAINING) ----------
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

    **Models are pre‑trained** – see the `train/` folder for training scripts.
    Use the sidebar to navigate.
    """)

elif page == "Dataset":
    st.title("📊 Dataset")
    st.write("Load your dataset from a URL or paste CSV content below.")

    # URL Loader
    st.subheader("Load from URL")
    url = st.text_input("Enter the URL of a CSV file (e.g., raw GitHub link):")
    if st.button("Load from URL"):
        if url:
            try:
                df = pd.read_csv(url)
                df = ensure_derived_features(df)
                st.session_state['df'] = df
                st.success("Dataset loaded from URL successfully!")
            except Exception as e:
                st.error(f"Error loading from URL: {e}")

    # Paste CSV data
    st.subheader("Paste CSV Data")
    csv_text = st.text_area("Paste the CSV content here (including header row):", height=200)
    if st.button("Load from pasted CSV"):
        if csv_text.strip():
            try:
                df = pd.read_csv(StringIO(csv_text))
                df = ensure_derived_features(df)
                st.session_state['df'] = df
                st.success("Dataset loaded from pasted CSV successfully!")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
        else:
            st.warning("Please paste some CSV data.")

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
    else:
        st.info("Please load a dataset using one of the methods above.")

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
        st.error("No pre‑trained models found. Please ensure models are in `saved_models/`.")
        st.stop()

    # Load preprocessor from meta or pipeline – no fitting from dataset
    preprocessor = get_preprocessor_only()
    if preprocessor is None:
        st.error("No preprocessor found. Please ensure `meta.pkl` is present or models are saved as pipelines.")
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