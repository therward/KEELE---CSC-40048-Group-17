# Logistic Regression with Automated Feature Selection

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset 
file_path = r"C:\Users\chels\OneDrive\Documents\Univeristy\Visualisation for Data Analytics (CSC-40048)\Covid Data Cleaned.csv"
df = pd.read_csv(file_path)

# Displaying the first few rows to understand the data 
print("First few rows of your dataset:")
print(df.head())

# Selecting features (X) and target variable (y)
X = df.drop(columns=['DIED', 'DATE_DIED'])  # Using all columns except the target and date
y = df['DIED']  # Target variable

# Scaling the features (very important for Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Automating Feature Selection (Testing different numbers of features)
best_accuracy = 0
best_num_features = 0
best_features = []

for num_features in range(5, X.shape[1] + 1):
    log_reg = LogisticRegression(max_iter=500, solver='liblinear')
    rfe = RFE(estimator=log_reg, n_features_to_select=num_features)
    X_selected = rfe.fit_transform(X_scaled, y)
    
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # Training the model
    log_reg.fit(X_train, y_train)
    
    # Making predictions
    y_pred = log_reg.predict(X_test)
    
    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Checking if this is the best model so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_num_features = num_features
        best_features = X.columns[rfe.support_]

print(f"\nBest Number of Features: {best_num_features}")
print(f"Best Features: {list(best_features)}")
print(f"Best Model Accuracy: {best_accuracy:.4f}")

# Training the final model with the best number of features
log_reg_final = LogisticRegression(max_iter=500, solver='liblinear')
rfe_final = RFE(estimator=log_reg_final, n_features_to_select=best_num_features)
X_final = rfe_final.fit_transform(X_scaled, y)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Training the final model
log_reg_final.fit(X_train, y_train)

# Making predictions
y_pred = log_reg_final.predict(X_test)

# Evaluating the final model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nFinal Logistic Regression Model Performance (Optimized Features):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
plt.title("Confusion Matrix - Logistic Regression (Optimized Features)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
