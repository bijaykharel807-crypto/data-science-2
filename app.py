import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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
# Hyperparameter tuning config
# ---------------------------------------------------------
MODEL_REGISTRY = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=1000),
        "params": {
            "C": {"type": "float_list", "default": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "penalty": {"type": "cat_list", "options": ["l1", "l2"], "default": ["l2"]},
            "solver": {"type": "fixed", "value": "liblinear"},
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": {"type": "int_list", "default": [100, 200, 300]},
            "max_depth": {"type": "int_list_nullable", "default": [None, 5, 10, 20]},
            "min_samples_split": {"type": "int_list", "default": [2, 5, 10]},
            "min_samples_leaf": {"type": "int_list", "default": [1, 2, 4]},
        },
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": {"type": "int_list", "default": [100, 200]},
            "learning_rate": {"type": "float_list", "default": [0.01, 0.05, 0.1, 0.2]},
            "max_depth": {"type": "int_list", "default": [2, 3, 4]},
        },
    },
    "SVM": {
        "estimator": SVC(probability=True, random_state=42),
        "params": {
            "C": {"type": "float_list", "default": [0.1, 1.0, 10.0]},
            "kernel": {"type": "cat_list", "options": ["linear", "rbf", "poly"], "default": ["rbf"]},
            "gamma": {"type": "cat_list", "options": ["scale", "auto"], "default": ["scale"]},
        },
    },
    "K-Nearest Neighbors": {
        "estimator": KNeighborsClassifier(),
        "params": {
            "n_neighbors": {"type": "int_list", "default": [3, 5, 7, 9, 11]},
            "weights": {"type": "cat_list", "options": ["uniform", "distance"], "default": ["uniform"]},
            "p": {"type": "cat_list", "options": [1, 2], "default": [2]},
        },
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": {"type": "int_list_nullable", "default": [None, 3, 5, 10]},
            "min_samples_split": {"type": "int_list", "default": [2, 5, 10]},
            "criterion": {"type": "cat_list", "options": ["gini", "entropy"], "default": ["gini"]},
        },
    },
}


def render_param_grid_ui(model_name):
    """Renders widgets for each hyperparameter and returns the resulting param grid dict."""
    config = MODEL_REGISTRY[model_name]["params"]
    param_grid = {}

    for param_name, spec in config.items():
        if spec["type"] == "fixed":
            param_grid[param_name] = [spec["value"]]
            continue

        label = f"{param_name}"

        if spec["type"] == "float_list":
            raw = st.text_input(f"{label} (comma-separated)", value=", ".join(str(v) for v in spec["default"]))
            try:
                param_grid[param_name] = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
            except ValueError:
                st.warning(f"Could not parse values for {label}, using default.")
                param_grid[param_name] = spec["default"]

        elif spec["type"] == "int_list":
            raw = st.text_input(f"{label} (comma-separated)", value=", ".join(str(v) for v in spec["default"]))
            try:
                param_grid[param_name] = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
            except ValueError:
                st.warning(f"Could not parse values for {label}, using default.")
                param_grid[param_name] = spec["default"]

        elif spec["type"] == "int_list_nullable":
            display_default = ", ".join("None" if v is None else str(v) for v in spec["default"])
            raw = st.text_input(f"{label} (comma-separated, use 'None' for no limit)", value=display_default)
            values = []
            for x in raw.split(","):
                x = x.strip()
                if x == "" :
                    continue
                values.append(None if x.lower() == "none" else int(x))
            param_grid[param_name] = values

        elif spec["type"] == "cat_list":
            selected = st.multiselect(label, options=spec["options"], default=spec["default"])
            param_grid[param_name] = selected if selected else spec["default"]

    # drop empty lists
    param_grid = {k: v for k, v in param_grid.items() if v}
    return param_grid


# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
PAGES = ["Project Details", "Dataset", "EDA", "Model Comparison", "Hyperparameter Tuning", "Prediction"]

with st.sidebar:
    if HAS_OPTION_MENU:
        page = option_menu(
            menu_title="Navigation",
            options=PAGES,
            icons=["info-circle", "bar-chart", "graph-up", "bullseye", "sliders", "target"],
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
    - **Hyperparameter Tuning** — run Grid Search / Randomized Search on a chosen model and optionally save it
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
# PAGE: Hyperparameter Tuning
# ---------------------------------------------------------
elif page == "Hyperparameter Tuning":
    st.title("🛠️ Hyperparameter Tuning")
    st.write("Pick a model, set the search space, and run Grid Search or Randomized Search with cross-validation.")

    if df is None:
        st.error("Could not load the Titanic dataset.")
        st.stop()

    X = build_onehot_10col(df[RAW_FEATURE_COLS])
    y = df[TARGET_COL]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        model_name = st.selectbox("Model", list(MODEL_REGISTRY.keys()))
    with col_b:
        search_type = st.selectbox("Search strategy", ["Grid Search", "Randomized Search"])
    with col_c:
        cv_folds = st.slider("CV folds", min_value=2, max_value=10, value=5)

    scoring = st.selectbox(
        "Scoring metric",
        ["accuracy", "roc_auc", "precision", "recall", "f1"],
        index=0,
    )

    st.subheader(f"Search space for {model_name}")
    st.caption("Edit the comma-separated values or selected options to control which combinations are tried.")
    param_grid = render_param_grid_ui(model_name)

    n_combinations = 1
    for v in param_grid.values():
        n_combinations *= max(len(v), 1)
    st.info(f"This search space covers **{n_combinations}** parameter combination(s) × {cv_folds}-fold CV "
             f"= **{n_combinations * cv_folds}** model fits.")

    n_iter = None
    if search_type == "Randomized Search":
        n_iter = st.slider("Number of random combinations to try (n_iter)",
                            min_value=1, max_value=max(n_combinations, 1),
                            value=min(10, max(n_combinations, 1)))

    run_col, save_col = st.columns([1, 1])
    run_clicked = run_col.button("🚀 Run Hyperparameter Search", type="primary")

    if "tuning_result" not in st.session_state:
        st.session_state.tuning_result = None

    if run_clicked:
        if not param_grid:
            st.error("Please provide at least one valid parameter value.")
        else:
            estimator = MODEL_REGISTRY[model_name]["estimator"]
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

            with st.spinner(f"Running {search_type} on {model_name}..."):
                start = time.time()
                try:
                    if search_type == "Grid Search":
                        search = GridSearchCV(
                            estimator=estimator,
                            param_grid=param_grid,
                            scoring=scoring,
                            cv=cv,
                            n_jobs=-1,
                            refit=True,
                        )
                    else:
                        search = RandomizedSearchCV(
                            estimator=estimator,
                            param_distributions=param_grid,
                            n_iter=n_iter,
                            scoring=scoring,
                            cv=cv,
                            n_jobs=-1,
                            random_state=42,
                            refit=True,
                        )
                    search.fit(X, y)
                    elapsed = time.time() - start

                    st.session_state.tuning_result = {
                        "model_name": model_name,
                        "best_estimator": search.best_estimator_,
                        "best_params": search.best_params_,
                        "best_score": search.best_score_,
                        "cv_results": pd.DataFrame(search.cv_results_),
                        "scoring": scoring,
                        "elapsed": elapsed,
                    }
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.session_state.tuning_result = None

    result = st.session_state.tuning_result
    if result and result["model_name"] == model_name:
        st.success(f"✅ Search completed in {result['elapsed']:.1f}s")

        m1, m2 = st.columns(2)
        m1.metric(f"Best CV {result['scoring']}", f"{result['best_score']:.4f}")
        m2.write("**Best parameters:**")
        m2.json(result["best_params"])

        st.subheader("Top 10 parameter combinations")
        cv_df = result["cv_results"][["params", "mean_test_score", "std_test_score", "rank_test_score"]]
        cv_df = cv_df.sort_values("rank_test_score").head(10).reset_index(drop=True)
        st.dataframe(cv_df, use_container_width=True)

        st.subheader("Evaluate the tuned model on the full dataset")
        best_model = result["best_estimator"]
        proba = _proba(best_model, X) / 100
        preds = (proba >= 0.5).astype(int)
        eval_cols = st.columns(5)
        eval_cols[0].metric("Accuracy", f"{accuracy_score(y, preds):.3f}")
        eval_cols[1].metric("Precision", f"{precision_score(y, preds, zero_division=0):.3f}")
        eval_cols[2].metric("Recall", f"{recall_score(y, preds, zero_division=0):.3f}")
        eval_cols[3].metric("F1", f"{f1_score(y, preds, zero_division=0):.3f}")
        eval_cols[4].metric("ROC-AUC", f"{roc_auc_score(y, proba):.3f}")

        st.subheader("Save tuned model")
        default_filename = model_name.lower().replace(" ", "_") + "_tuned.pkl"
        save_filename = st.text_input("Filename to save in saved_models/", value=default_filename)
        if st.button("💾 Save this model"):
            if not save_filename.endswith(".pkl"):
                save_filename += ".pkl"
            os.makedirs(MODELS_DIR, exist_ok=True)
            save_path = os.path.join(MODELS_DIR, save_filename)
            joblib.dump(best_model, save_path)
            load_model.clear()  # clear cache_resource so the new file is picked up
            st.success(f"Saved to `{save_path}`. It will now appear in Model Comparison and Prediction pages.")


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