import pandas as pd
import numpy as np

# Sample dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Heidi', None, 'Ivan'],
    'Age': [24, 27, np.nan, 22, 30, 29, 31, 35, 28, 40],
    'Email': ['alice@example.com', 'bob@example', 'charlie@example.com', 'david@example.com', 'eve@example.com', 'frank@example.com', None, 'heidi@example.com', 'ivan@example.com', 'alice@example.com'],
    'Salary': [50000, 54000, 58000, 52000, None, 60000, 62000, 59000, 61000, 63000]
}

df = pd.DataFrame(data)

def data_profiling(df):
    profile = {
        'Number of Rows': df.shape[0],
        'Number of Columns': df.shape[1],
        'Missing Values': df.isnull().sum().to_dict(),
        'Duplicate Records': df.duplicated().sum(),
        'Data Types': df.dtypes.to_dict(),
        'Summary Statistics': df.describe().to_dict()
    }
    return profile

profile = data_profiling(df)
print("Data Profiling:", profile)

def data_cleansing(df):
    # Fill missing values in 'Age' with mean age
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    
    # Fill missing values in 'Salary' with mean salary
    df['Salary'].fillna(df['Salary'].mean(), inplace=True)
    
    # Drop rows with missing 'Name' or 'Email'
    df.dropna(subset=['Name', 'Email'], inplace=True)
    
    # Correct invalid email addresses
    df['Email'] = df['Email'].apply(lambda x: x if '@' in x else np.nan)
    df.dropna(subset=['Email'], inplace=True)
    
    return df

df_cleaned = data_cleansing(df)
print("Cleaned Data:\n", df_cleaned)

def data_validation(df):
    # Check for valid age range
    valid_age = df['Age'].between(0, 100)
    
    # Check for unique email addresses
    unique_email = df['Email'].duplicated(keep=False)
    
    validation_results = {
        'Valid Age': valid_age.all(),
        'Unique Email': not unique_email.any()
    }
    
    return validation_results

validation_results = data_validation(df_cleaned)
print("Data Validation Results:", validation_results)

def data_governance(df):
    # Create a data dictionary
    data_dictionary = {
        'Name': 'Full name of the individual',
        'Age': 'Age of the individual in years',
        'Email': 'Email address of the individual',
        'Salary': 'Annual salary of the individual in USD'
    }
    
    # Data quality metrics
    data_quality_metrics = {
        'Total Records': len(df),
        'Valid Age Records': df['Age'].between(0, 100).sum(),
        'Valid Email Records': df['Email'].notnull().sum(),
        'Duplicate Email Records': df['Email'].duplicated().sum()
    }
    
    return data_dictionary, data_quality_metrics

data_dictionary, data_quality_metrics = data_governance(df_cleaned)
print("Data Dictionary:", data_dictionary)
print("Data Quality Metrics:", data_quality_metrics)
