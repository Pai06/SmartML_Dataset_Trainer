import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from ml import preprocess, preprocess_sequences, train, train_rnn_lstm, evaluate_classification, evaluate_regression, predict

# Initialize Streamlit
st.set_page_config(
    page_title="Dataset Trainer and Predictor",
    page_icon="📈",
    layout="centered"
)

# Directories
working_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(working_directory)
data_directory = f"{parent_directory}/data"
trained_models_directory = f"{parent_directory}/trained"

# Title
st.title("💻 Dataset Trainer and Predictor")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV, XLS, XLSX)", type=["csv", "xls", "xlsx"])

df = None  # Initialize df

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

# Display dataset if read successfully
if df is not None:
    # Handle missing values
    df = df.fillna(df.mean())

    st.dataframe(df.head())

    # Columns for selection
    column1, column2, column3, column4 = st.columns(4)

    # Options
    scaler_type = ["Standard", "MinMax"]
    model_dict = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Classifier": SVC(),
        "Random Forest Classifier": RandomForestClassifier(),
        "XGBoost Classifier": XGBClassifier(),
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(),
        "XGBoost Regressor": XGBRegressor(),
        "RNN": "RNN",
        "LSTM": "LSTM",
        "GRU": "GRU"
    }

    # UI components
    with column1:
        target_column = st.selectbox("Select Target Column", list(df.columns))

    with column2:
        scaler = st.selectbox("Select Scaler", scaler_type)

    with column3:
        select_model = st.selectbox("Select Model", list(model_dict.keys()))

    with column4:
        model_name = st.text_input("Name of Model")

    # Additional hyperparameters for RNN, LSTM, and GRU
    if select_model in ["RNN", "LSTM", "GRU"]:
        st.sidebar.header(f"{select_model} Hyperparameters")
        num_layers = st.sidebar.slider("Number of Layers", 1, 5, 2)
        units_per_layer = []
        for i in range(num_layers):
            units = st.sidebar.slider(f"Number of Units in Layer {i + 1}", 10, 200, 50)
            units_per_layer.append(units)
        activation = st.sidebar.selectbox("Activation Function", ["relu", "tanh"])
        optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd"])
        learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, 0.001)
        dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)
        recurrent_dropout_rate = st.sidebar.slider("Recurrent Dropout Rate", 0.0, 0.5, 0.0)
        batch_size = st.sidebar.slider("Batch Size", 1, 128, 32)
        epochs = st.sidebar.slider("Epochs", 1, 100, 10)
        validation_split = st.sidebar.slider("Validation Split", 0.0, 0.5, 0.2)
        early_stopping = st.sidebar.checkbox("Early Stopping")
        bidirectional = st.sidebar.checkbox("Bidirectional")

    # Additional hyperparameters for traditional models
    else:
        st.sidebar.header(f"{select_model} Hyperparameters")

        if select_model in ["Logistic Regression", "Support Vector Classifier"]:
            penalty = st.sidebar.selectbox("Penalty", ["l1", "l2", "elasticnet", "none"])
            C = st.sidebar.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
            solver = st.sidebar.selectbox("Solver", ["liblinear", "saga"])

        if select_model in ["Random Forest Classifier", "Random Forest Regressor"]:
            n_estimators = st.sidebar.slider("Number of Estimators", 10, 500, 100)
            max_depth = st.sidebar.slider("Max Depth", 1, 50, 10)
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
            min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, 1)

        if select_model in ["XGBoost Classifier", "XGBoost Regressor"]:
            n_estimators = st.sidebar.slider("Number of Estimators", 10, 500, 100)
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
            max_depth = st.sidebar.slider("Max Depth", 1, 50, 6)
            min_child_weight = st.sidebar.slider("Min Child Weight", 1, 10, 1)

        if select_model in ["Linear Regression", "SVR", "SVR"]:
            C = st.sidebar.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
            epsilon = st.sidebar.slider("Epsilon", 0.01, 1.0, 0.1)

# Train model button
if st.button("Train Model"):
    if select_model in ["RNN", "LSTM", "GRU"]:
        X_train, X_test, y_train, y_test = preprocess_sequences(df, target_column, scaler)
        history = train_rnn_lstm(
            X_train, y_train, select_model, num_layers, units_per_layer, activation,
            optimizer, learning_rate, dropout_rate, recurrent_dropout_rate,
            epochs, batch_size, validation_split, early_stopping, bidirectional, model_name
        )
        performance_metrics = evaluate_regression(history.model, X_test, y_test)

        # Save trained model and target_column to session state
        st.session_state['trained_model'] = history.model
        st.session_state['target_column'] = target_column

        # Plot training history
        st.header(f"{select_model} Training History")
        fig_loss = px.line(y=history.history['loss'], labels={'x': 'Epoch', 'y': 'Loss'}, title='Training Loss Over Epochs')
        st.plotly_chart(fig_loss, use_container_width=True, config={'displayModeBar': False})

    else:
        X_train, X_test, y_train, y_test = preprocess(df, target_column, scaler)
        
        # Adjust model based on selected hyperparameters
        if select_model == "Logistic Regression":
            model = LogisticRegression(penalty=penalty, C=C, solver=solver)
        elif select_model == "Support Vector Classifier":
            model = SVC(C=C)
        elif select_model == "Random Forest Classifier":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        elif select_model == "XGBoost Classifier":
            model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight)
        elif select_model == "Linear Regression":
            model = LinearRegression()
        elif select_model == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        elif select_model == "XGBoost Regressor":
            model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight)
        elif select_model == "SVR":
            model = SVR(C=C, epsilon=epsilon)

        if select_model in ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"]:
            model = train(X_train, y_train, model, model_name, regression=True)
            performance_metrics = evaluate_regression(model, X_test, y_test)
        else:
            model = train(X_train, y_train, model, model_name, regression=False)
            performance_metrics = evaluate_classification(model, X_test, y_test)

        # Save trained model and target_column to session state
        st.session_state['trained_model'] = model
        st.session_state['target_column'] = target_column

        # Display performance metrics
        st.header("Test Performance Metrics")
        st.json(performance_metrics)

    if select_model in ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor", "RNN", "LSTM", "GRU"]:
        st.header("Performance Metrics")
        mse = performance_metrics.get("Mean Squared Error", "N/A")
        r2 = performance_metrics.get("R-squared", "N/A")

        st.write(f"**Mean Squared Error:** {mse}")
        st.write(f"**R-squared:** {r2}")

    # confusion matrix for classification
    else:
        st.header("Confusion Matrix")
        cm = performance_metrics["Confusion Matrix"]
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'], colorscale='Blues'))
        fig_cm.update_layout(title='Confusion Matrix', xaxis_title='Predicted labels', yaxis_title='True labels')
        st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})

else:
    if uploaded_file is None:
        st.write("Please upload a dataset to proceed.")


# prediction section
st.header("🔍 Model Prediction")

# Upload dataset for prediction
uploaded_prediction_file = st.file_uploader("Upload new data for prediction (CSV, XLS, XLSX)", type=["csv", "xls", "xlsx"])

if uploaded_prediction_file is not None:
    if uploaded_prediction_file.name.endswith('.csv'):
        df_prediction = pd.read_csv(uploaded_prediction_file)
    else:
        df_prediction = pd.read_excel(uploaded_prediction_file)

    st.dataframe(df_prediction.head())

    # Ensure target column is not present in prediction data
    if 'trained_model' in st.session_state:
        target_column = st.session_state['target_column']
        
        # Print the name of the target column being predicted
        st.write(f"Target Column for Prediction: **{target_column}**")

        if target_column in df_prediction.columns:
            st.error(f"The target column '{target_column}' should not be present in the prediction dataset.")
        else:
            if st.button("Predict"):
                # Prepare data for prediction
                if isinstance(st.session_state['trained_model'], Sequential):
                    X_pred = df_prediction.values  # For neural network models
                    if len(X_pred.shape) == 2:
                        X_pred = np.expand_dims(X_pred, axis=1)  # Ensure 3D input for neural networks
                else:
                    X_pred = df_prediction.values  # For traditional models

                predictions = predict(X_pred, st.session_state['trained_model'])
                st.header("Predictions")
                st.dataframe(predictions)
    else:
        st.write("Please train a model first before making predictions.")
