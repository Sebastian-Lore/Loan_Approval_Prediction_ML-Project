# This file processes the data (encoding, scaling, and saving it to a new CSV).

# imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# load the dataset
file_path = "loan_approval_dataset.csv"  # Update the file path if necessary
df = pd.read_csv(file_path)

le = LabelEncoder() # encode categorical columns with Label Encoding

df.columns = df.columns.str.strip() # clean column names of leading/trailing spaces

df['education'] = le.fit_transform(df['education'])
df['self_employed'] = le.fit_transform(df['self_employed'])

# split the data into features (X) and target (y)
X = df.drop(columns=['loan_id', 'loan_status'])  # drop non-feature columns
y = df['loan_status']  # target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # split the data into training and test sets (80% train, 20% test)

# standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

df_encoded = pd.get_dummies(df, columns=['education', 'self_employed'], drop_first=True) # One-Hot Encoding for categorical features (education, self_employed)

# convert 'loan_status' from string to numeric (0 = Rejected, 1 = Approved)
le_status = LabelEncoder()
df_encoded['loan_status'] = le_status.fit_transform(df_encoded['loan_status'])

df_encoded.to_csv('preprocessed_loan_data.csv', index=False) # save the preprocessed data to a new CSV file

print("Preprocessed data saved as 'preprocessed_loan_data.csv'") # confirm the data is saved
