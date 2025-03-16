# This file does the following.
#   - Trains a Random Forest Classifier using preprocessed loan data.
#   - Evaluates the model using accuracy, a confusion matrix, and a classification report.
#   - Saves the evaluation results to a file (model_results.txt).

# imports
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('preprocessed_loan_data.csv') # load the preprocessed dataset

# split the data into features (X) and target (y)
X = df.drop(columns=['loan_id', 'loan_status'])  # drop non-feature columns
y = df['loan_status']  # target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # split the data into training and test sets (80% train, 20% test)

# standardize numerical features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

y_pred = rf_classifier.predict(X_test_scaled) # make predictions on the test set

# evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)  # convert report to a dictionary

TN, FP, FN, TP = conf_matrix.ravel() # extract values from confusion matrix

output_file = "model_results.txt" # define output file

# open the file in write mode
with open(output_file, "w") as f:

    # write confusion matrix and breakdown
    f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
    f.write("Confusion Matrix and Breakdown:\n\n")
    f.write(f"{conf_matrix}\n\n")
    f.write(f"True Negatives (TN): {TN} - These are the instances where the model predicted 0 (Rejected loan) and the actual label was also 0 (Rejected loan).\n")
    f.write(f"False Positives (FP): {FP} - These are the instances where the model predicted 1 (Approved loan), but the actual label was 0 (Rejected loan).\n")
    f.write(f"False Negatives (FN): {FN} - These are the instances where the model predicted 0 (Rejected loan), but the actual label was 1 (Approved loan).\n")
    f.write(f"True Positives (TP): {TP} - These are the instances where the model predicted 1 (Approved loan) and the actual label was also 1 (Approved loan).\n\n")

    # extract classification report metrics for each class and write classification report and breakdown
    precision_0 = class_report['0']['precision']
    recall_0 = class_report['0']['recall']
    f1_0 = class_report['0']['f1-score']
    support_0 = class_report['0']['support']

    precision_1 = class_report['1']['precision']
    recall_1 = class_report['1']['recall']
    f1_1 = class_report['1']['f1-score']
    support_1 = class_report['1']['support']

    f.write("Classification Report and Breakdown:\n\n")
    f.write(json.dumps(class_report, indent=4) + "\n\n")

    f.write("Class 0 (Rejected Loans):\n")
    f.write(f"  Precision: {precision_0:.4f} - When the model predicts a rejected loan, it is correct {precision_0*100:.2f}% of the time.\n")
    f.write(f"  Recall: {recall_0:.4f} - The model correctly identifies {recall_0*100:.2f}% of all rejected loans.\n")
    f.write(f"  F1-score: {f1_0:.4f} - Harmonic mean of precision and recall, giving a balance between the two.\n")
    f.write(f"  Support: {support_0} - Number of actual rejected loan cases in the test data.\n\n")

    f.write("Class 1 (Approved Loans):\n")
    f.write(f"  Precision: {precision_1:.4f} - When the model predicts an approved loan, it is correct {precision_1*100:.2f}% of the time.\n")
    f.write(f"  Recall: {recall_1:.4f} - The model correctly identifies {recall_1*100:.2f}% of all approved loans.\n")
    f.write(f"  F1-score: {f1_1:.4f} - Harmonic mean of precision and recall, giving a balance between the two.\n")
    f.write(f"  Support: {support_1} - Number of actual approved loan cases in the test data.\n\n")

    # write overall performance metrics
    f.write("Overall Model Performance:\n")
    f.write(f"  Macro Average F1-score: {class_report['macro avg']['f1-score']:.4f} - Average F1-score for both classes, treating them equally.\n")
    f.write(f"  Weighted Average F1-score: {class_report['weighted avg']['f1-score']:.4f} - Average F1-score considering class imbalance.\n")

print(f"Results saved to {output_file}")
