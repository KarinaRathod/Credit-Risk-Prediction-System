import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Credit Risk Prediction", 
    page_icon="💳", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Helper Functions (Cached)
# =========================
@st.cache_data(show_spinner=False)
def load_data(filepath="dataset.csv"):
    """Loads the dataset and caches it to prevent reloading on every interaction."""
    return pd.read_csv(filepath)

@st.cache_resource(show_spinner="Training model...")
def build_and_train_pipeline(X, y):
    """Builds a robust preprocessing pipeline and trains the model. Cached for performance."""
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    # Robust preprocessing with imputation for missing values
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)
    
    return pipeline, X_train, X_test, y_train, y_test

# =========================
# Main Application
# =========================
st.title("💳 Credit Risk Prediction System")
st.markdown("This application predicts credit default risk using a Random Forest classifier. Upload batch files or input details manually to get instant risk assessments.")

# Load Data
try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Error loading data: Make sure `dataset.csv` is in the same directory. ({e})")
    st.stop()

# Sidebar Configuration
st.sidebar.header("⚙️ Data & Model Settings")
target_column = st.sidebar.selectbox(
    "Select Target Column (Default: Last Column)",
    df.columns,
    index=len(df.columns)-1,
    help="Select the column that represents the credit risk label (e.g., 1 for High Risk, 0 for Low Risk)."
)

# Define X and y
X = df.drop(target_column, axis=1)
y = df[target_column]

# Train Model (Cached)
pipeline, X_train, X_test, y_train, y_test = build_and_train_pipeline(X, y)

# =========================
# Application Layout (Tabs)
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview", 
    "📈 Model Performance", 
    "🔮 Single Prediction", 
    "📂 Batch Prediction"
])

# --- TAB 1: Data Overview ---
with tab1:
    st.subheader("Dataset Snapshot")
    st.dataframe(df.head(15), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Features", df.shape[1] - 1)
    col3.metric("Target Variable", target_column)

# --- TAB 2: Model Performance ---
with tab2:
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.subheader("Random Forest Classifier Evaluation")
    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)
        
    with col2:
        st.write("**Classification Report**")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap="Blues"), use_container_width=True)

# --- TAB 3: Single Prediction ---
with tab3:
    st.subheader("Predict Credit Risk (Manual Input)")
    st.markdown("Fill out the fields below to predict the credit risk for a single profile.")
    
    # Create a dynamic form layout using columns
    with st.form("single_prediction_form"):
        input_data = {}
        cols = st.columns(3) # Display inputs in a 3-column grid
        
        for i, col in enumerate(X.columns):
            col_container = cols[i % 3]
            if pd.api.types.is_numeric_dtype(X[col]):
                input_data[col] = col_container.number_input(
                    label=col,
                    min_value=float(X[col].min()),
                    max_value=float(X[col].max()),
                    value=float(X[col].median()), # Using median as a safer default than mean
                    step=float(X[col].std() / 10 if X[col].std() > 0 else 1.0)
                )
            else:
                input_data[col] = col_container.selectbox(
                    label=col,
                    options=X[col].dropna().unique()
                )
        
        submit_button = st.form_submit_button("Predict Risk", type="primary")

    if submit_button:
        input_df = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)[0]
        
        st.divider()
        if prediction == 1:
            st.error("⚠️ **High Credit Risk Detected** - Further review recommended.")
        else:
            st.success("✅ **Low Credit Risk Detected** - Profile appears safe.")

# --- TAB 4: Batch Prediction ---
with tab4:
    st.subheader("Batch Prediction Pipeline")
    st.markdown("Upload a CSV file with the same feature columns to get bulk predictions.")
    
    uploaded_file = st.file_uploader("Upload Unseen Data (CSV)", type=["csv"])

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        
        try:
            # Predict and append results
            predictions = pipeline.predict(batch_df)
            batch_df["Predicted_Risk"] = predictions
            
            st.success(f"Successfully processed {len(batch_df)} records.")
            st.dataframe(batch_df, use_container_width=True)

            # Download capabilities
            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name="credit_risk_predictions.csv",
                mime="text/csv",
                type="primary"
            )
        except Exception as e:
            st.error(f"❌ Schema Error: Ensure your uploaded CSV matches the training features. Details: {e}")