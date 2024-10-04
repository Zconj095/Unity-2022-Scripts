import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset
data = {
    'Values': [5, 7, 8, 8, 10, 6, 7, 8, 9, 10, 12, 15, 14, 13, 15, 10, 10, 10, 9, 8]
}

df = pd.DataFrame(data)

# Calculate frequency pattern measures
mean = df['Values'].mean()
median = df['Values'].median()
mode = df['Values'].mode()[0]
variance = df['Values'].var()
std_dev = df['Values'].std()

# Print the measures
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev}")

# Create a frequency table
frequency_table = df['Values'].value_counts().reset_index()
frequency_table.columns = ['Value', 'Frequency']
print(f"\nFrequency Table:\n{frequency_table}")

# Plot histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['Values'], bins=10, kde=True)
plt.title('Histogram of Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean}')
plt.axvline(median, color='g', linestyle='-', label=f'Median: {median}')
plt.axvline(mode, color='b', linestyle='-', label=f'Mode: {mode}')
plt.legend()
plt.show()

# Plot boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Values'])
plt.title('Boxplot of Values')
plt.xlabel('Value')
plt.show()
