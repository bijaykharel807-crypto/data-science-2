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
    ["Project Details", "Dataset", "EDA", "Train Models", "Prediction"]
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

# ---------- PREPROCESSOR EXTRACTION ----------
def get_preprocessor_from_models(models):
    for model in models.values():
        if isinstance(model, Pipeline) and hasattr(model, 'named_steps'):
            for step_name, step_obj in model.named_steps.items():
                if isinstance(step_obj, ColumnTransformer):
                    return step_obj
    return None

def get_preprocessor_only():
    meta = load_meta()
    if meta and 'preprocessor' in meta:
        return meta['preprocessor']
    models = load_all_models()
    preprocessor = get_preprocessor_from_models(models)
    if preprocessor is not None:
        return preprocessor
    return None

# ---------- TRAINING FUNCTION ----------
def train_models(df):
    """Train all models on the given dataframe and save to saved_models/"""
    # Ensure required columns exist
    required = FEATURE_NAMES + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ])
    preprocessor.fit(X_train)

    # Models
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'LogisticRegression': LogisticRegression(random_state=42)
    }

    # Save preprocessor meta
    meta = {'preprocessor': preprocessor}
    with open(os.path.join(MODEL_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    # Train and save each model
    for name, clf in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, os.path.join(MODEL_DIR, f'{name}.pkl'))

    # Also save a "best" model (here we pick the best accuracy on test set)
    # For simplicity, we'll just copy the best performer (RandomForest often good)
    # Or we can evaluate all and pick the best
    best_acc = -1
    best_name = None
    for name, clf in models.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        if acc > best_acc:
            best_acc = acc
            best_name = name
    # Save the best as BEST_<name>.pkl
    best_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', models[best_name])
    ])
    best_pipe.fit(X_train, y_train)
    joblib.dump(best_pipe, os.path.join(MODEL_DIR, f'BEST_{best_name}.pkl'))

    return f"✅ Models trained successfully! Best model: {best_name} with accuracy {best_acc:.3f}"

# ---------- PAGES ----------

if page == "Project Details":
    st.title("📋 Project Details")
    st.markdown("""
    ### Titanic Survival Prediction

    This app trains and uses classifiers to predict survival.

    - **Features**: Age, SibSp, FamilySize, Parch, Pclass, Embarked, Sex, Fare, IsAlone
    - **Target**: Survived (0 = No, 1 = Yes)
    - **Models**: Decision Tree, Random Forest, Gradient Boosting, KNN, Logistic Regression
    - **Workflow**:
        1. Load a dataset (or use the default Titanic data).
        2. Explore with EDA.
        3. Train models on the **Train Models** page.
        4. Make predictions on the **Prediction** page.
    """)

elif page == "Dataset":
    st.title("📊 Dataset")
    st.write("Load your dataset by pasting CSV content below, or use the default Titanic dataset.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Default Titanic Dataset"):
            try:
                df = sns.load_dataset('titanic')
                # Keep relevant columns and rename
                df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']].copy()
                df.rename(columns={
                    'survived': 'Survived',
                    'pclass': 'Pclass',
                    'sex': 'Sex',
                    'age': 'Age',
                    'sibsp': 'SibSp',
                    'parch': 'Parch',
                    'fare': 'Fare',
                    'embarked': 'Embarked'
                }, inplace=True)
                df = ensure_derived_features(df)
                # Drop rows with missing values for simplicity
                df = df.dropna()
                st.session_state['df'] = df
                st.success("Default Titanic dataset loaded!")
            except Exception as e:
                st.error(f"Error loading default dataset: {e}")

    with col2:
        st.write("Or paste your own CSV:")
        csv_text = st.text_area("Paste CSV content (header row required):", height=150)
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
        st.info("No dataset loaded. Use one of the options above.")

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

elif page == "Train Models":
    st.title("🧠 Train Models")
    st.write("Train all models on the currently loaded dataset. If no dataset is loaded, the default Titanic dataset will be used.")

    # Check if dataset exists; if not, load default temporarily for training
    df_for_training = None
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df_for_training = st.session_state['df']
        st.info(f"Using dataset with {len(df_for_training)} rows from the Dataset page.")
    else:
        st.warning("No dataset loaded. Will use the default Titanic dataset for training (loaded automatically).")
        try:
            df_temp = sns.load_dataset('titanic')
            df_temp = df_temp[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']].copy()
            df_temp.rename(columns={
                'survived': 'Survived',
                'pclass': 'Pclass',
                'sex': 'Sex',
                'age': 'Age',
                'sibsp': 'SibSp',
                'parch': 'Parch',
                'fare': 'Fare',
                'embarked': 'Embarked'
            }, inplace=True)
            df_temp = ensure_derived_features(df_temp)
            df_temp = df_temp.dropna()
            df_for_training = df_temp
            st.info("Default Titanic dataset loaded for training.")
        except Exception as e:
            st.error(f"Could not load default dataset: {e}")

    if df_for_training is not None:
        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training models... This may take a moment."):
                try:
                    result = train_models(df_for_training)
                    st.success(result)
                    # Clear cache so that models are reloaded next time
                    st.cache_resource.clear()
                    st.info("Models saved to `saved_models/`. You can now go to the Prediction page.")
                except Exception as e:
                    st.error(f"Training failed: {e}")
    else:
        st.error("No dataset available for training. Please load a dataset first.")

    # Show existing models
    st.subheader("Existing Models")
    models = load_all_models()
    if models:
        st.write("Found the following models:")
        for fname in models.keys():
            st.write(f"- {fname}")
    else:
        st.write("No models found. Train them using the button above.")

elif page == "Prediction":
    st.title("🎯 Make a Prediction")
    st.write("Enter passenger details and choose a model to predict survival.")

    models = load_all_models()
    if not models:
        st.error("No pre‑trained models found. Please go to the **Train Models** page and train the models first.")
        st.stop()

    preprocessor = get_preprocessor_only()
    if preprocessor is None:
        st.error("No preprocessor found. Please train models again.")
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