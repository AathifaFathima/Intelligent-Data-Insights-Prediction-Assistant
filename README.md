# Intelligent Data Insights & Prediction Assistant

## Project Description
This project is a Streamlit-based application that allows users
to upload CSV datasets, analyze data, visualize patterns,
train machine learning models, and generate predictions.

## Features
- CSV Dataset Upload
- Data Quality Analysis
- Exploratory Data Analysis (EDA)
- Data Cleaning
- Data Visualization
- Classification using Random Forest
- Regression using Random Forest
- Model Evaluation
- AI Prediction System
- Model Download (.pkl)
- Model Information Download (.json)

## Technologies Used
- Python
- Streamlit
- Pandas
- Matplotlib
- Scikit-learn
- Pytest

## How to Run

1. Install dependencies:

   python -m pip install streamlit pandas matplotlib scikit-learn pytest

2. Run the application:

   python -m streamlit run app.py

3. Open the Streamlit dashboard in your browser.

## Machine Learning Models

### Classification
Random Forest Classifier

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Regression
Random Forest Regressor

Evaluation Metrics:
- MAE
- RMSE
- R² Score

## Testing

Run automated tests:

python -m pytest tests/test_data_processing.py -v

The project includes 8 automated data processing tests.

## Project Structure

Intelligent_Data_Insights/
- app.py
- data_processing.py
- eda.py
- visualization.py
- model_training.py
- tests/
  - test_data_processing.py
- README.md