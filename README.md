# Loan_Approval_Prediction_ML-Project Overview
 
This project uses Machine Learning (ML) with a Random Forest Classifier to predict loan approval status based on applicant financial attributes. The model processes real-world data, cleans and prepares it, trains a classifier, and evaluates performance.
By automating loan approval predictions, this project demonstrates data preprocessing, feature engineering, model training, evaluation, and reporting—essential skills in data science and Machine Learning.

Programming Language: Python
Libraries Used:
   pandas, numpy (Data Handling)
   scikit-learn (Machine Learning)
   json (Report Formatting)
ML Algorithm: Random Forest Classifier
Data Source: Loan-Approval-Prediction-Dataset (from Kaggle)

How project works:
   Load and Clean the Data (preprocess_data.py)
      Reads raw_loan_data.csv.
      Handles missing values, encodes categorical variables, and normalizes numerical features.
      Saves the cleaned dataset as preprocessed_loan_data.csv.
   Train and Evaluate the Model (train_model.py)
      Loads preprocessed_loan_data.csv.
      Splits data into 80% training / 20% testing.
      Trains a Random Forest Classifier.
      Generates accuracy, confusion matrix, and classification report.
      Saves results to model_results.txt.

How to run project:
   Step 1) Install Dependencies: pip install -r requirements.txt
   Step 2) Preprocess the Data: python preprocess_data.py
   Step 3) Train the Model and View Results: python train_model.py
   Step 4) View the results: cat model_results.txt

   This project demonstrates end-to-end machine learning workflow for loan approval prediction. It effectively applies data preprocessing, model training, and performance evaluation to solve a real-world problem.
