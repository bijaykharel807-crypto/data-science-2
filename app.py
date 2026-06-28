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

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="ML Dashboard", layout="wide")

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Project Details", "Dataset", "EDA", "Model Comparison", "Prediction"]
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

# ---------- MODEL LOADER ----------
@st.cache_resource
def load_model(filename):
    filepath = os.path.join(MODEL_DIR, filename)
    # Strategy 1: joblib
    try:
        return joblib.load(filepath)
    except Exception:
        pass
    # Strategy 2: pickle
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        pass
    # Strategy 3: pickle with latin1
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

meta = load_meta()
if meta and "feature_names" in meta:
    FEATURE_NAMES = meta["feature_names"]
    NUMERIC_FEATURES = [f for f in FEATURE_NAMES if f not in CATEGORICAL_FEATURES]
if meta and "target_column" in meta:
    TARGET_COLUMN = meta["target_column"]

# ---------- RETRAINING FUNCTION ----------
def retrain_models(df):
    """
    Train all models using the provided dataset and save them with joblib.
    Also saves a meta.pkl containing the preprocessor, feature names, etc.
    """
    # Prepare features and target
    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN]

    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(drop='first'), CATEGORICAL_FEATURES)
        ])

    # Models to train
    models = {
        'DecisionTree.pkl': DecisionTreeClassifier(random_state=42),
        'RandomForest.pkl': RandomForestClassifier(random_state=42),
        'GradientBoosting.pkl': GradientBoostingClassifier(random_state=42),
        'KNN.pkl': KNeighborsClassifier(),
        'LogisticRegression.pkl': LogisticRegression(random_state=42),
        'BEST_DecisionTree.pkl': DecisionTreeClassifier(random_state=42)  # you can change this to a different best
    }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Train and save each model
    for filename, model in models.items():
        model.fit(X_train_processed, y_train)
        joblib.dump(model, os.path.join(MODEL_DIR, filename))

    # Save meta data (preprocessor, feature names, target column)
    meta_data = {
        'preprocessor': preprocessor,
        'feature_names': FEATURE_NAMES,
        'target_column': TARGET_COLUMN,
        'numeric_features': NUMERIC_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES
    }
    joblib.dump(meta_data, os.path.join(MODEL_DIR, 'meta.pkl'))

    return models, X_test_processed, y_test

# ---------- PAGES ----------

if page == "Project Details":
    st.title("📋 Project Details")
    st.markdown("""
    ### Titanic Survival Prediction

    This app serves multiple trained classifiers to predict survival.

    - **Features**: Age, SibSp, FamilySize, Parch, Pclass, Embarked, Sex, Fare, IsAlone
    - **Target**: Survived (0 = No, 1 = Yes)
    - **Models**: Decision Tree, Random Forest, Gradient Boosting, KNN, Logistic Regression
    - **Best model**: `BEST_DecisionTree.pkl`

    Use the sidebar to navigate.
    """)

elif page == "Dataset":
    st.title("📊 Dataset")
    st.write("Load your dataset from a URL, upload a CSV, or use the built‑in sample.")

    # URL Loader
    st.subheader("Load from URL")
    url = st.text_input("Enter the URL of a CSV file (e.g., raw GitHub link):")
    if st.button("Load from URL"):
        if url:
            try:
                df = pd.read_csv(url)
                st.session_state['df'] = df
                st.success("Dataset loaded from URL successfully!")
            except Exception as e:
                st.error(f"Error loading from URL: {e}")

    # File Uploader
    st.subheader("Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.success("Dataset uploaded successfully!")

    # Sample data (fallback)
    if 'df' not in st.session_state:
        np.random.seed(42)
        n = 200
        data = {
            "Age": np.random.uniform(1, 80, n).round(1),
            "SibSp": np.random.randint(0, 5, n),
            "FamilySize": np.random.randint(1, 6, n),
            "Parch": np.random.randint(0, 4, n),
            "Pclass": np.random.choice([1,2,3], n),
            "Embarked": np.random.choice(['C','Q','S'], n),
            "Sex": np.random.choice(['male','female'], n),
            "Fare": np.random.uniform(5, 100, n).round(2),
            "IsAlone": np.random.choice([0,1], n),
            TARGET_COLUMN: np.random.choice([0,1], n)
        }
        df = pd.DataFrame(data)
        st.session_state['df'] = df
        st.info("Using generated sample data. Load your own CSV or use a URL to replace it.")

    # Display dataset
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

elif page == "Model Comparison":
    st.title("⚖️ Model Comparison")
    st.write("Evaluate all models on a test set (20% holdout).")

    if 'df' not in st.session_state:
        st.warning("Please load a dataset first.")
        st.stop()

    df = st.session_state['df']
    if TARGET_COLUMN not in df.columns:
        st.error(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
        st.stop()

    # Check if all required features exist
    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_features:
        st.error(f"Missing features: {missing_features}. Please load a dataset with all required columns.")
        st.stop()

    # ----- Retraining section -----
    st.subheader("🔄 Retrain Models from Dataset")
    if st.button("Retrain all models"):
        with st.spinner("Training models... This may take a few seconds."):
            models, X_test_processed, y_test = retrain_models(df)
            st.success("✅ Models retrained and saved successfully!")
            # Clear cache so that fresh models are loaded
            st.cache_resource.clear()
            # Reload models
            models = load_all_models()
            if not models:
                st.error("Retrained models could not be loaded. Please check logs.")
                st.stop()
            # Proceed to evaluation
            st.session_state['models'] = models
            st.session_state['X_test_processed'] = X_test_processed
            st.session_state['y_test'] = y_test
            st.rerun()

    # ----- Evaluate models (either from cache or after retraining) -----
    if 'models' in st.session_state and 'X_test_processed' in st.session_state and 'y_test' in st.session_state:
        models = st.session_state['models']
        X_test_processed = st.session_state['X_test_processed']
        y_test = st.session_state['y_test']
    else:
        # Try to load existing models
        models = load_all_models()
        if models:
            # If models exist, we need to evaluate on the current dataset
            X = df[FEATURE_NAMES]
            y = df[TARGET_COLUMN]
            # Load meta to get preprocessor
            meta = load_meta()
            if meta and 'preprocessor' in meta:
                preprocessor = meta['preprocessor']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_test_processed = preprocessor.transform(X_test)
            else:
                # No preprocessor, use raw features (might fail)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_test_processed = X_test
            st.session_state['models'] = models
            st.session_state['X_test_processed'] = X_test_processed
            st.session_state['y_test'] = y_test
        else:
            st.info("No models found. Click 'Retrain all models' to train from your dataset.")
            st.stop()

    # Now evaluate
    results = []
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test_processed)
            if len(np.unique(st.session_state['y_test'])) <= 10:
                if len(np.unique(st.session_state['y_test'])) == 2:
                    acc = accuracy_score(st.session_state['y_test'], y_pred)
                    prec = precision_score(st.session_state['y_test'], y_pred, average='binary')
                    rec = recall_score(st.session_state['y_test'], y_pred, average='binary')
                    f1 = f1_score(st.session_state['y_test'], y_pred, average='binary')
                    results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1})
                else:
                    acc = accuracy_score(st.session_state['y_test'], y_pred)
                    prec = precision_score(st.session_state['y_test'], y_pred, average='weighted')
                    rec = recall_score(st.session_state['y_test'], y_pred, average='weighted')
                    f1 = f1_score(st.session_state['y_test'], y_pred, average='weighted')
                    results.append({"Model": name, "Accuracy": acc, "Precision (w)": prec, "Recall (w)": rec, "F1 (w)": f1})
            else:
                mse = mean_squared_error(st.session_state['y_test'], y_pred)
                r2 = r2_score(st.session_state['y_test'], y_pred)
                results.append({"Model": name, "MSE": mse, "R²": r2})
        except Exception as e:
            st.warning(f"Could not evaluate {name}: {e}")

    if results:
        results_df = pd.DataFrame(results)
        st.subheader("Performance Table")
        metric_cols = results_df.columns[1:]
        st.dataframe(results_df.style.highlight_max(axis=0, subset=metric_cols))

        st.subheader("Performance Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        melted = results_df.melt(id_vars=['Model'], value_vars=metric_cols,
                                 var_name='Metric', value_name='Score')
        sns.barplot(data=melted, x='Model', y='Score', hue='Metric', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No results to display.")

else:  # Prediction
    st.title("🎯 Make a Prediction")
    st.write("Enter passenger details and choose a model to predict survival.")

    # Load models
    models = load_all_models()
    if not models:
        st.error("No models loaded. Please go to Model Comparison and retrain models first.")
        st.stop()

    model_choice = st.selectbox("Select a model", list(models.keys()))
    model = models[model_choice]
    meta = load_meta()

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

    # Apply preprocessor if available
    if meta and 'preprocessor' in meta:
        try:
            input_processed = meta['preprocessor'].transform(input_df)
        except Exception as e:
            st.warning(f"Preprocessing failed: {e}")
            input_processed = input_df
    else:
        input_processed = input_df

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
        if meta:
            st.write("Metadata:", meta)