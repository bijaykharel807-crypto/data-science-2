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
ALL_FILES = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
MODEL_FILES = [f for f in ALL_FILES if f != 'meta.pkl']

FEATURE_NAMES = [
    "Age", "SibSp", "FamilySize", "Parch", "Pclass",
    "Embarked", "Sex", "Fare", "IsAlone"
]
CATEGORICAL_FEATURES = ["Embarked", "Sex"]
TARGET_COLUMN = "Survived"

# ---------- ROBUST MODEL LOADER ----------
@st.cache_resource
def load_model(filename):
    filepath = os.path.join(MODEL_DIR, filename)
    # Strategy 1: joblib
    try:
        return joblib.load(filepath)
    except Exception:
        pass

    # Strategy 2: pickle with default settings
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        pass

    # Strategy 3: pickle with latin1 encoding and fix_imports
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f, encoding='latin1', fix_imports=True)
    except Exception:
        pass

    # Strategy 4: try loading as a text file (unlikely, but just in case)
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # not a model, return None
            return None
    except:
        pass

    # If all fail, log the error (but we can't show it directly without leaking)
    st.error(f"❌ Could not load {filename}. Please re‑save with joblib in the same environment.")
    return None

@st.cache_resource
def load_all_models():
    models = {}
    for fname in MODEL_FILES:
        model = load_model(fname)
        if model is not None:
            models[fname] = model
        else:
            st.warning(f"⛔ Failed to load: {fname}")
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
            try:
                with open(meta_path, 'rb') as f:
                    return pickle.load(f, encoding='latin1', fix_imports=True)
            except:
                return None

meta = load_meta()
if meta and "feature_names" in meta:
    FEATURE_NAMES = meta["feature_names"]
if meta and "target_column" in meta:
    TARGET_COLUMN = meta["target_column"]

# ---------- PAGES ----------
# (everything else stays the same – keep the rest of the code exactly as before)
# I'll include the rest for completeness:

if page == "Project Details":
    st.title("📋 Project Details")
    st.markdown("""
    ### Titanic Survival Prediction

    This app serves multiple trained classifiers to predict whether a passenger survived the Titanic disaster.

    - **Features**: Age, SibSp, FamilySize, Parch, Pclass, Embarked, Sex, Fare, IsAlone
    - **Target**: Survived (0 = No, 1 = Yes)
    - **Models**: Decision Tree, Random Forest, Gradient Boosting, KNN, Logistic Regression
    - **Best model**: `BEST_DecisionTree.pkl`

    Use the sidebar to navigate.
    """)

elif page == "Dataset":
    st.title("📊 Dataset")
    st.write("Upload your CSV or use the built‑in Titanic sample.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.success("Dataset loaded!")
    else:
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
            st.info("Using generated sample data. Upload your own CSV to replace it.")
        else:
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
    else:
        df = st.session_state['df']
        if TARGET_COLUMN not in df.columns:
            st.error(f"Target column '{TARGET_COLUMN}' not found.")
        else:
            for feat in FEATURE_NAMES:
                if feat not in df.columns:
                    df[feat] = 0
            X = df[FEATURE_NAMES]
            y = df[TARGET_COLUMN]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = load_all_models()
            if not models:
                st.error("❌ No models could be loaded. Please re‑save them with joblib (see instructions above).")
            else:
                results = []
                for name, model in models.items():
                    try:
                        y_pred = model.predict(X_test)
                        if len(np.unique(y)) <= 10:
                            if len(np.unique(y)) == 2:
                                acc = accuracy_score(y_test, y_pred)
                                prec = precision_score(y_test, y_pred, average='binary')
                                rec = recall_score(y_test, y_pred, average='binary')
                                f1 = f1_score(y_test, y_pred, average='binary')
                                results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1})
                            else:
                                acc = accuracy_score(y_test, y_pred)
                                prec = precision_score(y_test, y_pred, average='weighted')
                                rec = recall_score(y_test, y_pred, average='weighted')
                                f1 = f1_score(y_test, y_pred, average='weighted')
                                results.append({"Model": name, "Accuracy": acc, "Precision (w)": prec, "Recall (w)": rec, "F1 (w)": f1})
                        else:
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
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
                    st.info("No models evaluated.")

else:  # Prediction
    st.title("🎯 Make a Prediction")
    st.write("Enter passenger details and choose a model to predict survival.")

    if not MODEL_FILES:
        st.error("No model files found in 'saved_models'.")
        st.stop()

    model_choice = st.selectbox("Select a model", MODEL_FILES)
    model = load_model(model_choice)
    if model is None:
        st.error(f"Could not load {model_choice}. Please re‑save with joblib.")
        st.stop()

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

    if meta and "preprocessor" in meta:
        try:
            input_df = meta["preprocessor"].transform(input_df)
        except Exception as e:
            st.warning(f"Preprocessing failed: {e}")

    if st.button("Predict", type="primary"):
        try:
            prediction = model.predict(input_df)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)
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