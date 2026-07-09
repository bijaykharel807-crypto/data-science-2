import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Titanic Survivor Prediction",
    page_icon="🚢",
    layout="wide"
)

# Title and description
st.title("🚢 Titanic Survivor Prediction Dashboard")
st.markdown("An interactive machine learning app to explore Titanic data and predict passenger survival.")

# Navigation
st.sidebar.markdown("## 📍 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Dataset", "📈 EDA", "🤖 Model Comparison", "🎯 Prediction"],
    index=0
)

# Load Titanic dataset with only required columns
@st.cache_data
def load_titanic_data():
    try:
        import seaborn as sns
        titanic = sns.load_dataset('titanic')
        # Keep only required columns
        required_cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
        titanic = titanic[required_cols]
        return titanic
    except:
        # If seaborn fails, create sample data
        np.random.seed(42)
        n = 891
        titanic = pd.DataFrame({
            'survived': np.random.choice([0, 1], n, p=[0.6, 0.4]),
            'pclass': np.random.choice([1, 2, 3], n, p=[0.25, 0.25, 0.5]),
            'sex': np.random.choice(['male', 'female'], n, p=[0.6, 0.4]),
            'age': np.random.normal(30, 14, n).clip(1, 80),
            'sibsp': np.random.choice([0, 1, 2, 3, 4], n, p=[0.6, 0.2, 0.1, 0.07, 0.03]),
            'parch': np.random.choice([0, 1, 2, 3, 4], n, p=[0.7, 0.15, 0.08, 0.05, 0.02]),
            'fare': np.random.exponential(32, n).clip(0, 512),
            'embarked': np.random.choice(['C', 'Q', 'S'], n, p=[0.2, 0.1, 0.7])
        })
        # Add some missing values
        titanic.loc[np.random.choice(n, 100), 'age'] = np.nan
        titanic.loc[np.random.choice(n, 20), 'embarked'] = np.nan
        return titanic

df = load_titanic_data()

# Function to create all features needed for the model
def create_all_features(data):
    """Create all features including engineered features"""
    # Create a copy
    X = data.copy()
    
    # Fill missing values
    X['age'] = X['age'].fillna(X['age'].median())
    X['fare'] = X['fare'].fillna(X['fare'].median())
    X['embarked'] = X['embarked'].fillna('S')
    
    # Encode categorical variables
    X['sex'] = X['sex'].map({'male': 0, 'female': 1})
    X['embarked'] = X['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    
    # Create family size feature
    X['FamilySize'] = X['sibsp'] + X['parch'] + 1
    
    # Create IsAlone feature
    X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
    
    # Rename columns to match model expectations
    X = X.rename(columns={
        'pclass': 'Pclass',
        'sex': 'Sex',
        'age': 'Age',
        'sibsp': 'SibSp',
        'parch': 'Parch',
        'fare': 'Fare',
        'embarked': 'Embarked'
    })
    
    # Select final features in correct order
    final_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
    
    # Ensure all required columns exist
    for col in final_features:
        if col not in X.columns:
            X[col] = 0
    
    return X[final_features]

# Function to clean all existing models
def clean_all_models():
    """Remove all existing model files"""
    model_folder = 'saved_models'
    if os.path.exists(model_folder):
        for file in os.listdir(model_folder):
            if file.endswith('.pkl'):
                file_path = os.path.join(model_folder, file)
                try:
                    os.remove(file_path)
                except:
                    pass
        st.info("🗑️ Cleaned all old model files")
    else:
        os.makedirs(model_folder)
        st.info("📁 Created saved_models folder")

# Function to train fresh models
@st.cache_resource
def train_fresh_models():
    """Train fresh models and save them"""
    
    # Clean old models first
    clean_all_models()
    
    # Create all features
    X = create_all_features(df)
    y = df['survived']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models with clean names
    models_dict = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
    }
    
    # Train and save models
    trained_models = {}
    model_folder = 'saved_models'
    
    for name, model in models_dict.items():
        try:
            # For KNN and LogisticRegression, use scaled data
            if name in ['KNN', 'Logistic Regression']:
                model.fit(X_train_scaled, y_train)
                # Test the model
                test_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                # Test the model
                test_pred = model.predict(X_test)
            
            # Calculate test accuracy
            test_accuracy = accuracy_score(y_test, test_pred)
            
            # Save model with clean name
            file_name = f'{name.replace(" ", "")}.pkl'
            file_path = os.path.join(model_folder, file_name)
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            
            trained_models[name] = model
            st.info(f"✅ Trained {name} (Test Accuracy: {test_accuracy:.4f})")
            
        except Exception as e:
            st.warning(f"Could not train {name}: {str(e)}")
    
    # Train and save BEST model
    try:
        best_model = RandomForestClassifier(n_estimators=100, random_state=42)
        best_model.fit(X_train, y_train)
        best_test_pred = best_model.predict(X_test)
        best_accuracy = accuracy_score(y_test, best_test_pred)
        
        best_path = os.path.join(model_folder, 'BestModel.pkl')
        with open(best_path, 'wb') as f:
            pickle.dump(best_model, f)
        trained_models['Best Model'] = best_model
        st.info(f"✅ Trained Best Model (Test Accuracy: {best_accuracy:.4f})")
        
    except Exception as e:
        st.warning(f"Could not train Best Model: {str(e)}")
    
    # Save scaler
    try:
        scaler_path = os.path.join(model_folder, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    except Exception as e:
        st.warning(f"Could not save scaler: {str(e)}")
    
    st.success("✅ All models trained successfully!")
    return trained_models

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_folder = 'saved_models'
    
    # Check if folder exists
    if not os.path.exists(model_folder):
        st.info("📁 No models found. Training fresh models...")
        return train_fresh_models()
    
    # List all model files
    model_files = [f for f in os.listdir(model_folder) 
                   if f.endswith('.pkl') and f not in ['scaler.pkl', 'meta.pkl']]
    
    if not model_files:
        st.info("📁 No model files found. Training fresh models...")
        return train_fresh_models()
    
    # Try to load each model
    loaded_models = {}
    for model_file in model_files:
        file_path = os.path.join(model_folder, model_file)
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
                
            # Verify it's a valid model
            if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                # Test if model works with our data
                try:
                    test_X = create_all_features(df.head(1))
                    test_pred = model.predict(test_X)
                    
                    # Model is valid
                    model_name = model_file.replace('.pkl', '')
                    # Rename for display
                    if model_name == 'BestModel':
                        model_name = 'Best Model'
                    elif model_name == 'LogisticRegression':
                        model_name = 'Logistic Regression'
                    elif model_name == 'DecisionTree':
                        model_name = 'Decision Tree'
                    elif model_name == 'RandomForest':
                        model_name = 'Random Forest'
                    elif model_name == 'GradientBoosting':
                        model_name = 'Gradient Boosting'
                    
                    loaded_models[model_name] = model
                except Exception as e:
                    st.warning(f"Model {model_file} incompatible with current data: {str(e)}")
                    # Remove incompatible model
                    try:
                        os.remove(file_path)
                        st.info(f"🗑️ Removed incompatible model: {model_file}")
                    except:
                        pass
            else:
                st.warning(f"Invalid model: {model_file}")
                # Remove invalid model
                try:
                    os.remove(file_path)
                except:
                    pass
        except Exception as e:
            st.warning(f"Could not load {model_file}: {str(e)}")
            # Remove corrupted model
            try:
                os.remove(file_path)
                st.info(f"🗑️ Removed corrupted model: {model_file}")
            except:
                pass
    
    # If no models loaded, train fresh
    if not loaded_models:
        st.info("🔄 No compatible models found. Training fresh models...")
        return train_fresh_models()
    
    return loaded_models

# Load models
with st.spinner("Loading models..."):
    models = load_models()

# Show model status in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📦 Model Status")
if models:
    st.sidebar.success(f"✅ {len(models)} models loaded successfully")
    with st.sidebar.expander("View Models"):
        for name in models.keys():
            st.sidebar.write(f"- {name}")
else:
    st.sidebar.error("❌ No models loaded!")

# Remove meta model if it exists
if 'meta' in models:
    del models['meta']

# -------------------- PAGE 1: DATASET --------------------
if page == "📊 Dataset":
    st.header("📊 Dataset Preview")
    
    # Select number of rows
    num_rows = st.selectbox(
        "Select number of rows to view",
        options=[5, 10, 20, 50, 100],
        index=0
    )
    
    # Create a copy of the dataframe for display
    display_df = df.head(num_rows).copy()
    
    # Rename columns for better readability
    column_mapping = {
        'survived': 'Survived',
        'pclass': 'Passenger Class',
        'sex': 'Sex',
        'age': 'Age',
        'sibsp': 'Siblings/Spouses',
        'parch': 'Parents/Children',
        'fare': 'Fare',
        'embarked': 'Embarked'
    }
    
    # Only rename columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in display_df.columns:
            display_df = display_df.rename(columns={old_name: new_name})
    
    # Display table
    st.dataframe(display_df, use_container_width=True)
    
    st.write("---")
    
    # Shape of dataset
    st.subheader("Shape of Dataset:")
    st.write(f"({df.shape[0]}, {df.shape[1]})")
    
    st.write("---")
    
    # Statistical Summary
    st.subheader("Statistical Summary:")
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'survived' in numeric_cols:
        numeric_cols.remove('survived')
    
    # Create summary
    if numeric_cols:
        summary_df = df[numeric_cols].describe()
        
        # Rename columns for better display
        summary_columns_mapping = {
            'pclass': 'Passenger Class',
            'age': 'Age',
            'sibsp': 'Siblings/Spouses',
            'parch': 'Parents/Children',
            'fare': 'Fare'
        }
        
        # Rename columns that exist
        for old_name, new_name in summary_columns_mapping.items():
            if old_name in summary_df.columns:
                summary_df = summary_df.rename(columns={old_name: new_name})
        
        # Display summary
        st.dataframe(summary_df.round(2), use_container_width=True)
    else:
        st.info("No numeric columns found for statistical summary")

# -------------------- PAGE 2: EDA --------------------
elif page == "📈 EDA":
    st.header("📈 Exploratory Data Analysis")
    
    # Create tabs for different EDA sections
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Survival Analysis", "Missing Values"])
    
    with tab1:
        st.subheader("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            if 'age' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                df['age'].dropna().hist(bins=30, edgecolor='black', alpha=0.7)
                ax.set_title('Age Distribution')
                ax.set_xlabel('Age')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Age column not found")
        
        with col2:
            # Fare distribution
            if 'fare' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                df['fare'].hist(bins=30, edgecolor='black', alpha=0.7)
                ax.set_title('Fare Distribution')
                ax.set_xlabel('Fare (£)')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Fare column not found")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Passenger Class distribution
            if 'pclass' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                df['pclass'].value_counts().sort_index().plot(kind='bar', ax=ax)
                ax.set_title('Passenger Class Distribution')
                ax.set_xlabel('Passenger Class')
                ax.set_ylabel('Count')
                ax.set_xticklabels(['First', 'Second', 'Third'])
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Passenger Class column not found")
        
        with col4:
            # Sex distribution
            if 'sex' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                df['sex'].value_counts().plot(kind='bar', ax=ax)
                ax.set_title('Sex Distribution')
                ax.set_xlabel('Sex')
                ax.set_ylabel('Count')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Sex column not found")
    
    with tab2:
        st.subheader("Correlation Analysis")
        
        # Correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)
            plt.close()
            
            # Show top correlations with survived
            if 'survived' in corr_matrix.columns:
                st.subheader("Top Correlations with Survival")
                survived_corr = corr_matrix['survived'].drop('survived').sort_values(ascending=False)
                corr_df = pd.DataFrame({
                    'Feature': survived_corr.index,
                    'Correlation': survived_corr.values
                })
                st.dataframe(corr_df, use_container_width=True)
        else:
            st.info("No numeric columns found for correlation analysis")
    
    with tab3:
        st.subheader("Survival Analysis")
        
        if 'survived' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Survival by Passenger Class
                if 'pclass' in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    survival_by_class = df.groupby('pclass')['survived'].mean()
                    survival_by_class.plot(kind='bar', ax=ax)
                    ax.set_title('Survival Rate by Passenger Class')
                    ax.set_xlabel('Passenger Class')
                    ax.set_ylabel('Survival Rate')
                    ax.set_xticklabels(['First', 'Second', 'Third'])
                    ax.set_ylim(0, 1)
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show values
                    st.write("Survival Rates:")
                    survival_df = pd.DataFrame({
                        'Passenger Class': ['First', 'Second', 'Third'],
                        'Survival Rate': survival_by_class.values
                    })
                    survival_df['Survival Rate'] = survival_df['Survival Rate'] * 100
                    survival_df['Survival Rate'] = survival_df['Survival Rate'].round(2).astype(str) + '%'
                    st.dataframe(survival_df, use_container_width=True)
                else:
                    st.info("Passenger Class column not found")
            
            with col2:
                # Survival by Sex
                if 'sex' in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    survival_by_sex = df.groupby('sex')['survived'].mean()
                    survival_by_sex.plot(kind='bar', ax=ax)
                    ax.set_title('Survival Rate by Sex')
                    ax.set_xlabel('Sex')
                    ax.set_ylabel('Survival Rate')
                    ax.set_ylim(0, 1)
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show values
                    st.write("Survival Rates:")
                    survival_df = pd.DataFrame({
                        'Sex': ['Female', 'Male'],
                        'Survival Rate': survival_by_sex.values
                    })
                    survival_df['Survival Rate'] = survival_df['Survival Rate'] * 100
                    survival_df['Survival Rate'] = survival_df['Survival Rate'].round(2).astype(str) + '%'
                    st.dataframe(survival_df, use_container_width=True)
                else:
                    st.info("Sex column not found")
            
            # Age vs Survival
            if 'age' in df.columns:
                st.subheader("Age vs Survival")
                fig, ax = plt.subplots(figsize=(10, 6))
                df['age_bin'] = pd.cut(df['age'], bins=10)
                survival_by_age = df.groupby('age_bin')['survived'].mean()
                survival_by_age.plot(kind='bar', ax=ax)
                ax.set_title('Survival Rate by Age Group')
                ax.set_xlabel('Age Group')
                ax.set_ylabel('Survival Rate')
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                plt.close()
                df = df.drop('age_bin', axis=1)
        else:
            st.info("Survived column not found")
    
    with tab4:
        st.subheader("Missing Values Analysis")
        
        # Calculate missing values
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Values': missing_data.values,
            'Percentage': missing_percentage.values
        })
        
        # Filter columns with missing values
        missing_df = missing_df[missing_df['Missing Values'] > 0]
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
            
            # Visualize missing data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(missing_df['Column'], missing_df['Missing Values'])
            ax.set_title('Missing Values by Column')
            ax.set_xlabel('Column')
            ax.set_ylabel('Missing Values Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        else:
            st.success("✅ No missing values found in the dataset!")

# -------------------- PAGE 3: MODEL COMPARISON --------------------
elif page == "🤖 Model Comparison":
    st.header("🤖 Model Comparison")
    
    if models:
        # Remove any non-model objects
        valid_models = {}
        for name, model in models.items():
            if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                valid_models[name] = model
        
        if not valid_models:
            st.warning("No valid models found. Retraining...")
            with st.spinner("Training models... Please wait."):
                models = train_fresh_models()
                st.success("Models trained successfully!")
                st.rerun()
        else:
            # Prepare data for model evaluation with all features
            def prepare_data():
                # Create all features
                X = create_all_features(df)
                y = df['survived']
                return X, y
            
            X, y = prepare_data()
            
            # Evaluate all models
            results = []
            for name, model in valid_models.items():
                try:
                    # Make predictions
                    predictions = model.predict(X)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y, predictions)
                    precision = precision_score(y, predictions, average='binary')
                    recall = recall_score(y, predictions, average='binary')
                    f1 = f1_score(y, predictions, average='binary')
                    
                    results.append({
                        'Model': name,
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1
                    })
                except Exception as e:
                    st.warning(f"Could not evaluate {name}: {str(e)}")
            
            if results:
                # Convert to DataFrame
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('Accuracy', ascending=False)
                
                # Display like the image
                st.subheader("📊 Model Accuracy Comparison")
                
                # Create the accuracy table
                accuracy_table = results_df[['Model', 'Accuracy']].copy()
                accuracy_table['Accuracy'] = (accuracy_table['Accuracy'] * 100).round(2)
                st.dataframe(accuracy_table, use_container_width=True)
                
                # Create the bar chart
                st.subheader("📈 Model Accuracy Visualization")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get model names and accuracies
                model_names = results_df['Model'].values
                accuracies = (results_df['Accuracy'].values * 100).round(2)
                
                # Create bars with different colors
                colors = ['#4CAF50' if i == 0 else '#2196F3' for i in range(len(model_names))]
                bars = ax.bar(model_names, accuracies, color=colors, width=0.6)
                
                # Customize the chart
                ax.set_xlabel('Models', fontsize=12)
                ax.set_ylabel('Accuracy (%)', fontsize=12)
                ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
                ax.set_ylim(0, 100)
                
                # Add value labels on top of bars
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{acc:.1f}%', ha='center', va='bottom', fontsize=11)
                
                # Add grid for better readability
                ax.grid(axis='y', alpha=0.3)
                ax.set_axisbelow(True)
                
                st.pyplot(fig)
                plt.close()
                
                # Best model highlight
                best_model = results_df.iloc[0]
                st.success(f"🏆 **Best Model:** {best_model['Model']} with accuracy {best_model['Accuracy']*100:.2f}%")
                
                # Show all metrics for the best model
                with st.expander("📋 View All Model Metrics"):
                    # Display all metrics
                    display_df = results_df.copy()
                    for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
                        display_df[col] = (display_df[col] * 100).round(2).astype(str) + '%'
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Confusion Matrix for best model
                    st.subheader("🔍 Best Model - Confusion Matrix")
                    
                    # Predict with best model
                    best_model_obj = valid_models[best_model['Model']]
                    predictions = best_model_obj.predict(X)
                    
                    cm = confusion_matrix(y, predictions)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title(f'Confusion Matrix - {best_model["Model"]}')
                    st.pyplot(fig)
                    plt.close()
                    
                    # Classification Report
                    st.subheader("📋 Detailed Classification Report")
                    report = classification_report(y, predictions, target_names=['Not Survived', 'Survived'], output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    report_df = report_df.round(4)
                    st.dataframe(report_df, use_container_width=True)
            else:
                st.warning("No models could be evaluated")
    else:
        st.warning("No models found! Training fresh models...")
        with st.spinner("Training models... Please wait."):
            models = train_fresh_models()
            st.success("Models trained successfully!")
            st.rerun()

# -------------------- PAGE 4: PREDICTION --------------------
else:
    st.header("🎯 Make Prediction")
    
    # Remove meta model if it exists
    if 'meta' in models:
        del models['meta']
    
    # Filter valid models
    valid_models = {}
    for name, model in models.items():
        if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
            valid_models[name] = model
    
    if valid_models:
        # Model selection
        selected_model_name = st.selectbox(
            "Select Model for Prediction",
            options=list(valid_models.keys()),
            help="Choose the model you want to use for prediction"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            pclass = st.selectbox(
                "Passenger Class",
                options=[1, 2, 3],
                format_func=lambda x: f"{x} - {'First' if x==1 else 'Second' if x==2 else 'Third'} Class"
            )
            
            sex = st.radio(
                "Sex",
                options=["Male", "Female"],
                horizontal=True
            )
            sex_encoded = 0 if sex == "Male" else 1
            
            age = st.slider(
                "Age (years)",
                min_value=1,
                max_value=100,
                value=30
            )
        
        with col2:
            sibsp = st.number_input(
                "Number of Siblings/Spouses Aboard",
                min_value=0,
                max_value=10,
                value=0
            )
            
            parch = st.number_input(
                "Number of Parents/Children Aboard",
                min_value=0,
                max_value=10,
                value=0
            )
            
            fare = st.number_input(
                "Ticket Fare (£)",
                min_value=0.0,
                max_value=512.0,
                value=32.0,
                step=0.1,
                format="%.2f"
            )
            
            embarked = st.selectbox(
                "Port of Embarkation",
                options=["C", "Q", "S"],
                format_func=lambda x: {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[x]
            )
        
        # Prediction button
        predict_button = st.button("🔮 Predict Survival", use_container_width=True)
        
        # Preprocess function for prediction
        def preprocess_input(data):
            # Create a dataframe with all features
            input_df = pd.DataFrame([{
                'pclass': data['pclass'],
                'sex': data['sex'],
                'age': data['age'],
                'sibsp': data['sibsp'],
                'parch': data['parch'],
                'fare': data['fare'],
                'embarked': data['embarked']
            }])
            
            # Use the same feature engineering function
            X = create_all_features(input_df)
            
            return X
        
        if predict_button:
            try:
                input_data = {
                    'pclass': pclass,
                    'sex': sex_encoded,
                    'age': age,
                    'sibsp': sibsp,
                    'parch': parch,
                    'fare': fare,
                    'embarked': embarked,
                }
                
                processed_data = preprocess_input(input_data)
                selected_model = valid_models[selected_model_name]
                
                prediction = selected_model.predict(processed_data)
                probability = selected_model.predict_proba(processed_data)
                
                st.write("---")
                st.subheader("Prediction Results")
                
                col_result1, col_result2, col_result3 = st.columns(3)
                
                with col_result1:
                    st.metric("Selected Model", selected_model_name)
                
                with col_result2:
                    survival_status = "Survived" if prediction[0] == 1 else "Did Not Survive"
                    st.metric("Prediction", survival_status)
                
                with col_result3:
                    survival_prob = probability[0][1] * 100
                    st.metric("Survival Probability", f"{survival_prob:.1f}%")
                
                if prediction[0] == 1:
                    st.success(f"✅ Predicted to SURVIVE with {survival_prob:.1f}% confidence")
                else:
                    st.error(f"❌ Predicted NOT to Survive with {100 - survival_prob:.1f}% confidence")
                
                st.progress(int(survival_prob))
                st.caption(f"Survival probability: {survival_prob:.1f}%")
                
                with st.expander("Show Detailed Analysis"):
                    summary_data = {
                        'Feature': ['Passenger Class', 'Sex', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare', 'Embarkation'],
                        'Value': [
                            f"{pclass} ({'First' if pclass==1 else 'Second' if pclass==2 else 'Third'})", 
                            sex, 
                            f"{age} years", 
                            sibsp, 
                            parch, 
                            f"£{fare:.2f}", 
                            f"{embarked} ({'Cherbourg' if embarked=='C' else 'Queenstown' if embarked=='Q' else 'Southampton'})"
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    st.table(summary_df)
                    
                    pred_details = {
                        'Metric': ['Predicted Class', 'Survival Probability', 'Non-Survival Probability'],
                        'Value': [
                            'Survived' if prediction[0] == 1 else 'Did Not Survive',
                            f"{probability[0][1]*100:.2f}%",
                            f"{probability[0][0]*100:.2f}%"
                        ]
                    }
                    pred_df = pd.DataFrame(pred_details)
                    st.table(pred_df)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("No valid models found for prediction. Training fresh models...")
        with st.spinner("Training models... Please wait."):
            models = train_fresh_models()
            st.success("Models trained successfully!")
            st.rerun()

# Footer
st.write("---")
st.caption("🚢 Built with Streamlit • Machine Learning for Titanic Survival Prediction")