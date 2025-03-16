# This file does the following.
#   Loads the CSV file into a Pandas DataFrame.
#   Checks for missing values, data types, and summary statistics.
#   Visualizes the distribution of key features.
#       Histograms: Help visualize the data in the income_annum, loan_amount, and cibil_score columns. 
#       Boxplot: Detects outliers in financial and loan-related fields and helps identify unexpected values, like negative asset values.
#       Correlation Heatmap: Shows relationships between numerical variables and helps understand how features impact each other.

# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset
file_path = "loan_approval_dataset.csv"  # Update the file path if necessary
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip() # clean column names of leading/trailing spaces

# print dataset overview
print("Dataset Overview:")
print(df.info(), "\n")

# check for missing values
print("Missing Values:")
print(df.isnull().sum(), "\n")

# print summary statistics
print("Summary Statistics:")
print(df.describe(), "\n")

# create a figure for histograms
df[['income_annum', 'loan_amount', 'cibil_score']].hist(bins=30, figsize=(12, 6))
plt.suptitle("Histograms of Income, Loan Amount, and CIBIL Score")
plt.show(block=False)  # show without blocking execution

# create a figure for boxplots
fig2, axes = plt.subplots(3, 3, figsize=(12, 8))
num_cols = ['income_annum', 'loan_amount', 'cibil_score', 'residential_assets_value', 
            'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

for i, col in enumerate(num_cols):
    sns.boxplot(y=df[col], ax=axes[i // 3, i % 3])
    axes[i // 3, i % 3].set_title(f"Boxplot of {col}")

plt.tight_layout()
plt.show(block=False)  # show without blocking execution

# create a figure for the correlation heatmap
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax3)
plt.title("Correlation Heatmap of Numerical Features")
plt.show(block=False)  # show without blocking execution

input("Press Enter to close all plots...")  # keeps all figures open until they are manually closed
