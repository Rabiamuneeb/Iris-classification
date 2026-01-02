import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="🌸 Iris Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title(" Made by Rabia Muneeb")
st.sidebar.markdown("Explore the Iris dataset, visualize patterns, train a model, or upload your own CSV!")

# Dark mode toggle
dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode")

# Apply custom heading colors
if dark_mode:
    st.markdown(
        """
        <style>
        body { background-color: #121212; color: #e0e0e0; }
        h1, h2, h3, h4 { color: #ffffff !important; } /* white headings */
        .stMarkdown, .stDataFrame, .stMetric { color: #e0e0e0; }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body { background-color: #ffffff; color: #000000; }
        h1, h2, h3, h4 { color: #2e86c1 !important; } /* blue headings */
        </style>
        """,
        unsafe_allow_html=True
    )

# File uploader
uploaded_file = st.sidebar.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("Iris.csv")
    return df

df = load_data(uploaded_file)

# -------------------------------
# Navigation
# -------------------------------
page = st.sidebar.radio("📂 Navigation", 
                        ["Dataset Overview", "Visualizations", "Model Trainer", "How to Use This App"])

# -------------------------------
# Page 1: Dataset Overview
# -------------------------------
if page == "Dataset Overview":
    st.title("📊 Dataset Overview")
    st.write("The Iris dataset is a classic dataset in machine learning. "
             "It contains measurements of iris flowers from three species: "
             "Setosa, Versicolor, and Virginica. This dataset is often used "
             "to teach classification because it’s simple yet powerful.")

    st.subheader("🔎 First Look at the Data")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📐 Summary Statistics")
    st.write(df.describe())

    if "Species" in df.columns:
        st.subheader("🌸 Class Distribution")
        st.bar_chart(df["Species"].value_counts())

# -------------------------------
# Page 2: Visualizations
# -------------------------------
elif page == "Visualizations":
    st.title("📈 Visualizations")
    st.write("Visualizations help us understand relationships between features.")

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    if len(numeric_cols) >= 2:
        st.subheader("Scatter Plot Explorer")
        x_axis = st.selectbox("Select X-axis", numeric_cols)
        y_axis = st.selectbox("Select Y-axis", numeric_cols)
        st.write(f"Scatter plot of **{x_axis} vs {y_axis}**")

        fig, ax = plt.subplots()
        if "Species" in df.columns:
            sns.scatterplot(data=df, x=x_axis, y=y_axis, hue="Species", palette="Set2", ax=ax)
        else:
            sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    st.write("This shows how numeric features are related to each other.")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# -------------------------------
# Page 3: Model Trainer
# -------------------------------
elif page == "Model Trainer":
    st.title("🤖 Train a Random Forest Classifier")
    st.write("A Random Forest is a machine learning model that uses many decision trees "
             "to make predictions. It’s chosen here because it’s simple, accurate, and "
             "works well for classification tasks like identifying flower species.")

    if "Species" not in df.columns:
        st.warning("Your dataset doesn’t have a 'Species' column, so model training isn’t available.")
    else:
        st.subheader("Select Features")
        features = [col for col in df.columns if col not in ["Id", "Species"]]
        selected_features = st.multiselect("Choose features", features, default=list(features))

        n_estimators = st.slider("Number of Trees in the Forest", 10, 200, 100)

        if selected_features:
            X = df[selected_features]
            y = df["Species"]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Model training
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Results
            st.subheader("📊 Model Performance")
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig3, ax3 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=model.classes_, yticklabels=model.classes_, ax=ax3)
            st.pyplot(fig3)

            st.subheader("Detailed Classification Report")
            st.text(classification_report(y_test, y_pred))

            # -------------------------------
            # Prediction Form
            # -------------------------------
            st.subheader("🔮 Try Your Own Prediction")
            st.write("Enter flower measurements below and click **Predict** to see the species.")

            sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
            sepal_width  = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
            petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
            petal_width  = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

            if st.button("Predict Species"):
                user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
                prediction = model.predict(user_input)
                st.success(f"🌸 The model predicts this flower is: **{prediction[0]}**")
        else:
            st.warning("Please select at least one feature to train the model.")

# -------------------------------
# Page 4: How to Use This App
# -------------------------------
elif page == "How to Use This App":
    st.title("📖 How to Use This App")
    st.write("This guide will walk you through the app step by step.")

    st.header("Step 1: Explore the Dataset")
    st.markdown("Go to **Dataset Overview** to see the first few rows, summary statistics, and class distribution.")

    st.header("Step 2: Visualize Relationships")
    st.markdown("In **Visualizations**, choose features for scatter plots and check correlations between numeric columns.")

    st.header("Step 3: Train a Model")
    st.markdown("In **Model Trainer**, select features and adjust the number of trees. "
                "The Random Forest classifier will predict flower species and show accuracy, confusion matrix, and a detailed report.")

    st.header("Step 4: Try Your Own Data")
    st.markdown("Upload a CSV file in the sidebar. If it has a `Species` column, you can train the model on your own dataset!")

    st.success("🌸 Tip: Use dark mode for a stylish look, and enjoy experimenting with different features!")