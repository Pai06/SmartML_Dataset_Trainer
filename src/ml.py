import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping 
import joblib
from tensorflow.keras.models import load_model as keras_load_model
import plotly.graph_objects as go

# Directories
working_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(working_directory)
trained_models_directory = f"{parent_directory}/trained"

# Preprocess data
def preprocess(df, target_column, scaler_type):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Numerical and categorical columns
    numerical_columns = X.select_dtypes(include=['number']).columns
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns

    # Numerical columns preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(numerical_columns) > 0:
        # Impute missing values
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_columns] = num_imputer.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = num_imputer.transform(X_test[numerical_columns])

        # Scaling based on scaler type
        if scaler_type == 'Standard':
            scaler = StandardScaler()
        elif scaler_type == 'MinMax':
            scaler = MinMaxScaler()

        X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    # Categorical columns preprocessing
    if len(categorical_columns) > 0:
        # Impute missing values
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_columns] = cat_imputer.fit_transform(X_train[categorical_columns])
        X_test[categorical_columns] = cat_imputer.transform(X_test[categorical_columns])

        # One-hot encoding
        encoder = OneHotEncoder()
        X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
        X_test_encoded = encoder.transform(X_test[categorical_columns])
        X_train_encoded = pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_columns))
        X_test_encoded = pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_columns))
        X_train = pd.concat([X_train.drop(columns=categorical_columns), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_columns), X_test_encoded], axis=1)

    return X_train, X_test, y_train, y_test

# Preprocess sequences
def preprocess_sequences(df, target_column, scaler_type, sequence_length=50):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if scaler_type == "Standard":
        scaler = StandardScaler()
    elif scaler_type == "MinMax":
        scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    X_sequences = []
    y_sequences = []

    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:i+sequence_length])
        y_sequences.append(y.iloc[i+sequence_length])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Train model
def train(X_train, y_train, model, model_name, regression=False):
    model.fit(X_train, y_train)

    # Save model
    with open(f"{trained_models_directory}/{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)

    return model

# Train RNN, LSTM, GRU
def train_rnn_lstm(
    X_train, y_train, model_type, num_layers, units_per_layer, activation,
    optimizer, learning_rate, dropout_rate, recurrent_dropout_rate,
    epochs, batch_size, validation_split, early_stopping, bidirectional, model_name,
    st_placeholder=None
):
    model = Sequential()

    for i in range(num_layers):
        if model_type == "RNN":
            if bidirectional:
                model.add(Bidirectional(SimpleRNN(units_per_layer[i], activation=activation, return_sequences=(i != num_layers - 1))))
            else:
                model.add(SimpleRNN(units_per_layer[i], activation=activation, return_sequences=(i != num_layers - 1)))
        elif model_type == "LSTM":
            if bidirectional:
                model.add(Bidirectional(LSTM(units_per_layer[i], activation=activation, return_sequences=(i != num_layers - 1), recurrent_dropout=recurrent_dropout_rate)))
            else:
                model.add(LSTM(units_per_layer[i], activation=activation, return_sequences=(i != num_layers - 1), recurrent_dropout=recurrent_dropout_rate))
        elif model_type == "GRU":
            if bidirectional:
                model.add(Bidirectional(GRU(units_per_layer[i], activation=activation, return_sequences=(i != num_layers - 1), recurrent_dropout=recurrent_dropout_rate)))
            else:
                model.add(GRU(units_per_layer[i], activation=activation, return_sequences=(i != num_layers - 1), recurrent_dropout=recurrent_dropout_rate))
        
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    if optimizer == "adam":
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        opt = SGD(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss="mean_squared_error")

    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=10))

    # Create a list to store losses
    losses = []

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Training Loss'))
    st_plot = st_placeholder.plotly_chart(fig, use_container_width=True)

    for epoch in range(epochs):
        history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks, verbose=0)
        losses.append(history.history['loss'][0])

        if st_placeholder:
            # Update dynamic graph
            fig.data[0].x = list(range(1, len(losses) + 1))
            fig.data[0].y = losses
            fig.layout.title = f'{model_type} Training Loss (Epoch {epoch + 1})'
            st_plot.plotly_chart(fig, use_container_width=True)

    # Save the model
    model_path = os.path.join(trained_models_directory, f"{model_name}.h5")
    model.save(model_path)

    return history


# Evaluate classification model
def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1) if len(y_pred.shape) > 1 else y_pred
    y_test_labels = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
    cm = confusion_matrix(y_test_labels, y_pred_labels)

    performance_metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Confusion Matrix": cm.tolist()
    }

    return performance_metrics

# Evaluate regression model
def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    performance_metrics = {
        "Mean Squared Error": mse,
        "R-squared": r2
    }

    return performance_metrics

def predict(df, model):
    if isinstance(df, pd.DataFrame):
        if target_column in df.columns:
            df = df.drop(columns=[target_column])
        X_pred = df.values
    else:
        X_pred = df
    
    # If the model is a neural network, ensure the input is 3D
    if isinstance(model, Sequential):
        if len(X_pred.shape) == 2:
            X_pred = np.expand_dims(X_pred, axis=1)  

    # Make predictions
    predictions = model.predict(X_pred)
    return predictions
