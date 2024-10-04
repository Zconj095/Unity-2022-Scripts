import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Sample training data 
X = np.array([[1, 2], [3, 4], [5, 6]]) # EM field feature vectors
y = np.array([1, 2, 1]) # EM source classes 

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Create SVM model 
model = SVC()

# Train model on data  
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate predictions
print(model.score(X_test, y_test))

# Function to classify new EM field data
def classify_em_field(features):
    prediction = model.predict([features]) 
    return prediction[0]

# Example usage 
em_features = [7, 8]
em_source = classify_em_field(em_features) 
print(em_source)