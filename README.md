# Bike Rental Demand Forecasting

## 1. Project Overview

This project focuses on forecasting bike rental demand using machine learning techniques.

The project uses the Kaggle Bike Sharing Demand dataset, which contains information about bike rental usage along with temporal, weather, and environmental features.

The objective is to develop machine learning models that can predict the number of bike rentals (count) for a given time and set of conditions.


## 2. Problem Statement

Bike rental demand varies depending on factors such as time of day, season, weather conditions, temperature, humidity, working day, and holidays.

Accurately forecasting rental demand can help bike-sharing services improve resource allocation, fleet availability, and operational planning.

This project develops a machine learning pipeline to predict bike rental demand from historical data.


## 3. Dataset

The project uses the Kaggle Bike Sharing Demand dataset.

The dataset files used in this project are:

- train.csv - training data containing the target variable count
- test.csv - test data used for final predictions
- sampleSubmission.csv - sample format for the final submission

The target variable is:

count = casual + registered

The casual and registered columns are not used as input features because they directly determine the target variable and would cause target leakage.


## 4. Project Objectives

The main objectives are:

1. Explore and understand the bike rental dataset.
2. Perform data preprocessing and cleaning.
3. Extract useful temporal features from the datetime column.
4. Train multiple machine learning regression models.
5. Evaluate the models using RMSLE.
6. Perform hyperparameter tuning to improve model performance.
7. Train the selected final model.
8. Generate predictions for the Kaggle test dataset.
9. Develop a Streamlit application for interactive prediction.


## 5. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook
- Git
- GitHub


## 6. Project Structure

bike-sharing-demand/
|
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
|
├── Data/
|   ├── train.csv
|   ├── test.csv
|   └── sampleSubmission.csv
|
├── models/
|   ├── final_model.pkl
|   └── feature_columns.pkl
|
├── notebooks/
|   ├── 01_dataset_exploration.ipynb
|   ├── 02_preprocessing_feature_engineering.ipynb
|   ├── 03_model_training.ipynb
|   ├── 04_evaluation.ipynb
|   ├── 05_model_improvement.ipynb
|   └── 06_final_prediction.ipynb
|
└── results/
    ├── model_comparison.csv
    ├── evaluation_summary.csv
    ├── tuning_results.csv
    └── final_submission.csv


## 7. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the structure and characteristics of the dataset.

The analysis included:

- Dataset dimensions and data types
- Missing-value analysis
- Duplicate-value analysis
- Statistical summary of numerical variables
- Distribution of bike rental demand
- Seasonal demand patterns
- Weather-related patterns
- Hourly rental demand
- Working-day and weekend patterns
- Relationships between numerical features
- Correlation analysis

The EDA helped identify important temporal and environmental patterns in bike rental demand and guided the feature engineering process.


## 8. Data Preprocessing

The following preprocessing steps were performed.

### Datetime Conversion

The datetime column was converted into a proper datetime format.

### Missing-Value Checking

The dataset was checked for missing values before model training.

### Duplicate Checking

Duplicate records were checked during exploratory analysis.

### Target and Feature Separation

The target variable was separated from the input features.

Target:

count

The casual and registered columns were excluded because they directly determine count.

### Train-Validation Split

The training data was divided into training and validation subsets.

The validation data was kept separate from the training process to evaluate how well the models generalize to unseen data.


## 9. Feature Engineering

Features were extracted from the datetime column to capture temporal patterns in bike rental demand.

The following features were created:

- year
- month
- day
- hour
- day_of_week
- is_weekend
- peak_hour

Additional existing features such as temperature, humidity, windspeed, working day, holiday, season, and weather were also used.

### Peak Hour

A peak_hour feature was created to represent periods with higher expected bike usage.

### Categorical Encoding

Categorical features such as season and weather were converted into numerical representations using one-hot encoding.

The resulting feature columns were aligned between the training and test datasets.


## 10. Machine Learning Models

Three regression models were initially trained and compared.

### 10.1 Linear Regression

Linear Regression was used as the baseline model.

It provides a simple reference against which more complex machine learning models can be compared.

### 10.2 Random Forest Regressor

Random Forest is an ensemble learning method that combines multiple decision trees.

It can capture nonlinear relationships and interactions between different features.

### 10.3 Gradient Boosting Regressor

Gradient Boosting builds decision trees sequentially, with each new tree attempting to reduce errors made by previous trees.

It is suitable for learning complex nonlinear relationships in the dataset.


## 11. Model Evaluation

The models were evaluated using the validation dataset.

The primary evaluation metric was:

Root Mean Squared Logarithmic Error (RMSLE)

RMSLE is suitable for the bike rental demand problem because it evaluates the difference between actual and predicted values on a logarithmic scale.

Lower RMSLE indicates better performance.

The evaluation included:

- RMSLE comparison
- Actual vs predicted analysis
- Prediction error analysis
- Feature importance analysis

The model comparison results are available in:

results/model_comparison.csv


## 12. Model Improvement and Hyperparameter Tuning

After evaluating the initial models, hyperparameter tuning was performed to improve the performance of the stronger models.

The tuning process explored parameters such as:

- Number of estimators
- Maximum tree depth
- Minimum samples per leaf
- Learning rate
- Different Gradient Boosting configurations

The performance of different configurations was compared using validation RMSLE.

The tuning results are stored in:

results/tuning_results.csv

The best configuration was selected based on the lowest validation RMSLE.


## 13. Final Model

After model evaluation and hyperparameter tuning, the selected model was trained using the available training data.

The trained final model was saved using Joblib:

models/final_model.pkl

The feature-column information required to reproduce the same model input structure was saved as:

models/feature_columns.pkl


## 14. Final Prediction

The final model was used to generate predictions for the Kaggle test dataset.

Since bike rental demand cannot be negative, the generated predictions were constrained to non-negative values.

The final prediction file was saved as:

results/final_submission.csv

The submission follows the required format:

datetime
count


## 15. Streamlit Application

A Streamlit application was developed to provide an interactive interface for the trained machine learning model.

The application loads the saved model and feature-column information:

models/final_model.pkl
models/feature_columns.pkl

Users can enter the required conditions through the interface and obtain a predicted bike rental demand.

### Running the Application

From the project directory, run:

streamlit run app.py

The Streamlit application will open in a web browser.


## 16. Results

The project contains the following result files.

### Model Comparison

results/model_comparison.csv

Contains the validation performance of the initial machine learning models.

### Evaluation Summary

results/evaluation_summary.csv

Contains the evaluation analysis performed after model training.

### Hyperparameter Tuning

results/tuning_results.csv

Contains the results of the hyperparameter tuning experiments.

### Final Submission

results/final_submission.csv

Contains the final predictions generated for the Kaggle test dataset.

The exact model performance values are based on the experimental results generated during the project.


## 17. How to Run the Project

### Step 1: Open the Project Directory

Open the bike-sharing-demand project folder.

### Step 2: Activate the Virtual Environment

Activate the project virtual environment:

.venv\Scripts\activate

### Step 3: Install Dependencies

Install the required Python packages:

pip install -r requirements.txt

### Step 4: Run the Notebooks

The notebooks can be executed in the following order:

01_dataset_exploration.ipynb
02_preprocessing_feature_engineering.ipynb
03_model_training.ipynb
04_evaluation.ipynb
05_model_improvement.ipynb
06_final_prediction.ipynb

### Step 5: Run the Streamlit Application

streamlit run app.py


## 18. Key Findings

The project demonstrated the importance of temporal and environmental factors in predicting bike rental demand.

Important predictive information included:

- Hour of the day
- Day of the week
- Season
- Weather conditions
- Temperature
- Humidity
- Working-day information

Feature engineering from the datetime column helped represent the temporal patterns present in the original dataset.


## 19. Conclusion

This project demonstrates an end-to-end machine learning workflow for bike rental demand forecasting.

The workflow includes:

- Dataset exploration
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Hyperparameter tuning
- Final model training
- Test-set prediction
- Interactive Streamlit deployment

The project provides a complete machine learning pipeline from raw bike rental data to an interactive prediction application.


## 20. Author

Bike Rental Demand Forecasting
Machine Learning Mini Project
VISMAYA G-PES1UG24CS645
PURVIK GOWDA K-PES1UG25CS838


