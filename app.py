import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

st.set_page_config(page_title="Titanic Survival Prediction Dashboard", page_icon="🚢", layout="wide")

MODELS_DIR = "saved_models"
TARGET_COL = "survived"

RAW_FEATURE_COLS = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
CATEGORICAL_COLS = ["sex", "embarked", "pclass"]

LABEL_MAPS = {
    "sex": {"male": 1, "female": 0},
    "embarked": {"S": 2, "C": 0, "Q": 1},
}


# ---------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    """Loads the Titanic dataset directly from seaborn's built-in data repo — no CSV file needed."""
    try:
        raw = sns.load_dataset("titanic")
    except Exception:
        return None
    df = raw[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df["fare"] = df["fare"].fillna(df["fare"].median())
    return df


@st.cache_resource
def load_model(model_filename):
    return joblib.load(os.path.join(MODELS_DIR, model_filename))


def list_models():
    if not os.path.isdir(MODELS_DIR):
        return []
    excluded_names = {"scaler.pkl", "encoder.pkl", "preprocessor.pkl"}
    candidates = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl") and f not in excluded_names])

    valid_models = []
    for f in candidates:
        try:
            obj = load_model(f)
            if hasattr(obj, "predict"):
                valid_models.append(f)
        except Exception:
            continue
    return valid_models


def _proba(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1] * 100
    preds = m.predict(X)
    return np.array(preds, dtype=float) * 100


def build_onehot_10col(raw_df):
    """
    Builds the specific 10-column one-hot encoding your saved models expect:
    pclass, sex_female, sex_male, age, sibsp, parch, fare,
    embarked_C, embarked_Q, embarked_S
    (5 numeric passthrough + sex one-hot(2) + embarked one-hot(3), drop_first=False)
    """
    encoded = pd.get_dummies(raw_df, columns=["sex", "embarked"], drop_first=False)
    for col in ["sex_female", "sex_male", "embarked_C", "embarked_Q", "embarked_S"]:
        if col not in encoded.columns:
            encoded[col] = 0
    ordered_cols = ["pclass", "sex_female", "sex_male", "age", "sibsp", "parch",
                     "fare", "embarked_C", "embarked_Q", "embarked_S"]
    return encoded[ordered_cols]


def get_probability(model, X_input):
    """
    Returns survival probability (%) for each row in X_input.
    Tries encodings in order of likelihood, based on the models' expected
    feature count (10 = one-hot encoded sex + embarked, no drop_first).
    """
    # 1. The specific 10-column one-hot encoding (most likely match)
    try:
        return _proba(model, build_onehot_10col(X_input))
    except Exception:
        pass

    # 2. Raw input as-is (works if model is a full Pipeline with its own preprocessing)
    try:
        return _proba(model, X_input)
    except Exception:
        pass

    # 3. One-hot encode, align to model.feature_names_in_ if available
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        encoded = pd.get_dummies(X_input)
        for col in expected:
            if col not in encoded.columns:
                encoded[col] = 0
        encoded = encoded.reindex(columns=expected, fill_value=0)
        try:
            return _proba(model, encoded)
        except Exception:
            pass

    # 4. Simple label encoding (last resort)
    label_df = X_input.copy()
    for col, mapping in LABEL_MAPS.items():
        if col in label_df.columns:
            label_df[col] = label_df[col].map(mapping)
    return _proba(model, label_df)  # let this raise if it still fails


# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
PAGES = ["Project Details", "Dataset", "EDA", "Model Comparison", "Prediction"]

with st.sidebar:
    if HAS_OPTION_MENU:
        page = option_menu(
            menu_title="Navigation",
            options=PAGES,
            icons=["info-circle", "bar-chart", "graph-up", "bullseye", "target"],
            menu_icon="list",
            default_index=1,
            styles={
                "container": {"padding": "10px", "background-color": "#f4f5f7"},
                "icon": {"font-size": "16px"},
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "3px", "border-radius": "8px"},
                "nav-link-selected": {"background-color": "#ffffff", "color": "#e11d48", "font-weight": "600"},
            },
        )
    else:
        st.markdown("### Navigation")
        page = st.radio("Select Page", PAGES, index=1, label_visibility="collapsed")

df = load_dataset()


# ---------------------------------------------------------
# PAGE: Project Details
# ---------------------------------------------------------
if page == "Project Details":
    st.title("🚢 Titanic Survival Prediction Dashboard")
    st.write("An interactive machine learning app to explore Titanic passenger data and predict survival probability.")

    st.subheader("About this project")
    st.markdown("""
    - **Dataset** — preview Titanic passenger data (loaded automatically, no file upload needed)
    - **EDA** — explore survival patterns by class, sex, age, and fare
    - **Model Comparison** — compare saved classification models by accuracy, precision, recall, F1, ROC-AUC
    - **Prediction** — enter passenger details and get a predicted survival **probability (%)**
    """)

    if df is not None:
        st.info(f"Dataset loaded: **{df.shape[0]} rows × {df.shape[1]} columns** (via seaborn's built-in Titanic dataset)")
    else:
        st.warning("Could not load the Titanic dataset. Check your internet connection (it's fetched from seaborn's data repo on first run).")

    st.subheader("Models available")
    models = list_models()
    if models:
        st.write(", ".join(m.replace(".pkl", "") for m in models))
    else:
        st.warning(f"No `.pkl` files found in `{MODELS_DIR}/`.")


# ---------------------------------------------------------
# PAGE: Dataset
# ---------------------------------------------------------
elif page == "Dataset":
    st.title("📊 Dataset")

    if df is None:
        st.error("Could not load the Titanic dataset. Check your internet connection.")
        st.stop()

    st.subheader("Dataset Preview")
    n_rows = st.slider("Select number of rows to view", min_value=5, max_value=min(100, len(df)), value=5)
    st.dataframe(df.head(n_rows), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Shape of Dataset:")
        st.code(str(df.shape))

        st.subheader("Missing Values:")
        st.dataframe(df.isnull().sum().to_frame(name="0"), use_container_width=True)

    with c2:
        st.subheader("Statistical Summary:")
        st.dataframe(df.describe(include="all"), use_container_width=True)


# ---------------------------------------------------------
# PAGE: EDA
# ---------------------------------------------------------
elif page == "EDA":
    st.title("📈 Exploratory Data Analysis")

    if df is None:
        st.error("Could not load the Titanic dataset.")
        st.stop()

    st.subheader("Survival count")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=TARGET_COL, data=df, ax=ax)
    ax.set_xticklabels(["Did not survive", "Survived"])
    st.pyplot(fig)

    st.subheader("Survival rate by category")
    cat_feat = st.selectbox("Choose a feature", ["sex", "pclass", "embarked"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=cat_feat, y=TARGET_COL, data=df, ax=ax)
    ax.set_ylabel("Survival rate")
    st.pyplot(fig)

    st.subheader("Correlation heatmap")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Age distribution by survival")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x="age", hue=TARGET_COL, kde=True, multiple="stack", ax=ax)
    st.pyplot(fig)


# ---------------------------------------------------------
# PAGE: Model Comparison
# ---------------------------------------------------------
elif page == "Model Comparison":
    st.title("🎯 Model Comparison")

    model_files = list_models()
    if not model_files:
        st.error(f"No .pkl files found in '{MODELS_DIR}'.")
        st.stop()
    if df is None:
        st.error("Could not load the Titanic dataset.")
        st.stop()

    X_raw = df[RAW_FEATURE_COLS]
    y = df[TARGET_COL]

    results = []
    for mf in model_files:
        try:
            model = load_model(mf)
            proba = get_probability(model, X_raw) / 100
            preds = (proba >= 0.5).astype(int)
            results.append({
                "Model": mf.replace(".pkl", ""),
                "Accuracy": accuracy_score(y, preds),
                "Precision": precision_score(y, preds, zero_division=0),
                "Recall": recall_score(y, preds, zero_division=0),
                "F1": f1_score(y, preds, zero_division=0),
                "ROC-AUC": roc_auc_score(y, proba),
            })
        except Exception as e:
            results.append({"Model": mf.replace(".pkl", ""), "Accuracy": None, "Precision": None, "Recall": None, "F1": None, "ROC-AUC": None})
            st.error(f"{mf} failed: {e}")

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    st.dataframe(results_df, use_container_width=True)

    if results_df["Accuracy"].notna().any():
        st.subheader("Accuracy comparison")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=results_df, x="Model", y="Accuracy", ax=ax, palette="viridis")
        plt.xticks(rotation=20)
        st.pyplot(fig)
        st.success(f"🏆 Best performing model: **{results_df.iloc[0]['Model']}**")


# ---------------------------------------------------------
# PAGE: Prediction
# ---------------------------------------------------------
elif page == "Prediction":
    st.title("🎯 Prediction")

    model_files = list_models()
    if not model_files:
        st.error(f"No .pkl files found in '{MODELS_DIR}'.")
        st.stop()

    selected_model_file = st.selectbox("Choose a model", model_files)
    model = load_model(selected_model_file)
    st.success(f"Loaded **{selected_model_file}**")

    st.subheader("Enter passenger details")
    cols = st.columns(2)
    input_values = {}

    with cols[0]:
        input_values["pclass"] = st.selectbox("pclass (ticket class)", [1, 2, 3], index=2)
        input_values["sex"] = st.selectbox("sex", ["male", "female"])
        input_values["age"] = st.number_input("age", value=float(df["age"].median()) if df is not None else 28.0, min_value=0.0, max_value=100.0)
        input_values["embarked"] = st.selectbox("embarked", ["S", "C", "Q"])

    with cols[1]:
        input_values["sibsp"] = st.number_input("sibsp (siblings/spouses aboard)", min_value=0, max_value=10, value=0, step=1)
        input_values["parch"] = st.number_input("parch (parents/children aboard)", min_value=0, max_value=10, value=0, step=1)
        input_values["fare"] = st.number_input("fare", value=float(df["fare"].median()) if df is not None else 32.0, min_value=0.0)

    if st.button("Predict Survival Probability"):
        try:
            raw_row = pd.DataFrame([input_values])[RAW_FEATURE_COLS]
            proba = get_probability(model, raw_row)[0]
            st.success(f"🚢 Predicted Survival Probability: **{proba:.2f}%**")
            st.progress(min(max(proba / 100, 0.0), 1.0))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.caption("The model may expect a specific encoding. Check how sex/embarked were "
                       "encoded during training and adjust LABEL_MAPS in app.py if needed.")