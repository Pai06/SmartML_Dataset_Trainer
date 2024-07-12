# SmartML Dataset Trainer
## Machine Learning Model Training and Evaluation
This project provides a Streamlit-based web application for training, evaluating, and predicting with machine learning models on CSV datasets. It supports both traditional models (e.g., Logistic Regression, Random Forest) and neural network models (e.g., RNN, LSTM, GRU). The platform includes functionality for preprocessing data, training models, evaluating model performance, and making predictions on new data.

## Features
* Preprocessing: Handles missing values, scales numerical features, and encodes categorical features.
* Model Training: Supports training various traditional models and neural networks.
* Model Evaluation: Evaluates model performance using relevant metrics and provides visualizations.
* Predictions: Allows users to input custom data and obtain predictions from trained models.

## Models Supported
### Traditional Models
* Logistic Regression
* Linear Regression
* Random Forest Classifier
* Random Forest Regressor
* Support Vector Classifier (SVC)
* Support Vector Regressor (SVR)
* XGBoost Classifier
* XGBoost Regressor
### Neural Network Models
* RNN
* LSTM
* GRU

## Setup & Installation
1. Clone the repository
   ```bash
   git clone https://github.com/Pai06/SmartML_Dataset_Trainer.git
   cd trainer_website
2. Install dependencies
   ```bash
   pip install streamlit
   pip install scikit-learn
   pip install tensorflow
   pip install xgboost
3. Run the application
   ```bash
   streamlit run main.py

## Usage
### Preprocessing Data
* Upload your CSV dataset.
* Select the target column.
* Choose the type of scaler for numerical features (StandardScaler or MinMaxScaler).
### Training Models
#### Traditional Models
* Select the model type.
* Set the hyperparameters (specific to each model).
* Train the model.
#### Neural Network Models
* Choose the type of neural network (RNN, LSTM, GRU).
* Configure the network architecture (number of layers, units per layer, activation function, etc.).
* Set training parameters (optimizer, learning rate, dropout rates, epochs, batch size, etc.).
* Train the model.
#### Evaluating Models
* View performance metrics such as accuracy, precision, recall, confusion matrix for classification models.
* View mean squared error and R-squared for regression models.
* Visualize training history and performance metrics.
#### Making Predictions
* Enter custom data or upload a new dataset.
* Use the trained model to make predictions on the new data.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

## Acknowledgements
This project uses several open-source libraries, including:
* Streamlit
* scikit-learn
* TensorFlow
* XGBoost
* Pandas
* NumPy

## License
This project is licensed under the MIT License.

  
