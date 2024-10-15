import pandas as pd
import numpy as np

# Load the CSV file
try:
    df = pd.read_csv('VeloCityX.csv')
except FileNotFoundError:
    print("Error: The file 'VeloCityX.csv' was not found.")
    exit()

# Display the first few rows to inspect the data
print("Original Data:\n", df.head())

# Check for missing values in all columns
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Drop rows with missing values (NaN)
df_clean = df.dropna()

# Remove duplicates across all columns
df_clean = df_clean.drop_duplicates()

# Check for inconsistent data types for all columns
print("\nData Types Before Conversion:")
print(df_clean.dtypes)

# Convert relevant columns to numeric, coercing errors to NaN
numeric_columns = ['Fan Challenges Completed', 'Predictive Accuracy (%)', 
                   'Virtual Merchandise Purchases', 'Sponsorship Interactions (Ad Clicks)', 
                   'Time on Live 360 (mins)', 'Real-Time Chat Activity (Messages Sent)']

for col in numeric_columns:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Re-check data types after conversion
print("\nData Types After Conversion:")
print(df_clean.dtypes)

# Check for outliers in all numeric columns using the IQR method
print("\nSummary Statistics (Before Removing Outliers):")
print(df_clean.describe())

for col in numeric_columns:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df_clean[~((df_clean[col] < (Q1 - 1.5 * IQR)) | (df_clean[col] > (Q3 + 1.5 * IQR)))]

# Summary statistics after outlier removal
print("\nSummary Statistics (After Removing Outliers):")
print(df_clean.describe())

# Check if there are any duplicates left
print("\nDuplicate Rows Left (Should be 0):")
print(df_clean.duplicated().sum())

# Save the cleaned data to a new CSV file
df_clean.to_csv('cleaned_data.csv', index=False)

print("\nData Cleaning Complete! The cleaned data has been saved as 'cleaned_data.csv'.")



