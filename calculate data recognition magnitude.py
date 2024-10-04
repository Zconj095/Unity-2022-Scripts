import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'Feature1': [5, 7, 8, 9, 10, 6, 7, 8, 9, 10],
    'Feature2': [10, 11, 12, 13, 14, 10, 11, 12, 13, 14],
    'Label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Split the data into training and testing sets
X = df[['Feature1', 'Feature2']]
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Calculate data recognition magnitude (accuracy)
accuracy = accuracy_score(y_test, y_pred)
data_recognition_magnitude = accuracy
print("Data Recognition Magnitude:", data_recognition_magnitude)

