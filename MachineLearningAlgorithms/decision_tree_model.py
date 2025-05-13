print("Running Decision Tree Model...")
# Decision Tree Model - Grid Search Optimization

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# Load your dataset 
file_path = r"C:\Users\chels\OneDrive\Documents\Univeristy\Visualisation for Data Analytics (CSC-40048)\Covid Data Cleaned.csv"
df = pd.read_csv(file_path)

# Displaying the first few rows to understand the data 
print("First few rows of your dataset:")
print(df.head())

# Selecting features (X) and target variable (y)
X = df.drop(columns=['DIED', 'DATE_DIED'])  # Using all columns except the target and date
y = df['DIED']  # Target variable

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setting up Grid Search with Decision Tree
tree_clf = DecisionTreeClassifier(random_state=42)
param_grid = {
    'criterion': ['gini', 'entropy'],  # Testing both Gini and Entropy
    'max_depth': list(range(3, 21)),  # Testing depth from 3 to 20
    'min_samples_split': list(range(2, 11)),  # Testing minimum split samples
    'min_samples_leaf': list(range(1, 11))  # Testing minimum leaf samples
}

print("\nStarting Grid Search for Decision Tree... This may take a few minutes.")
start_time = time()
grid_search = GridSearchCV(tree_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
end_time = time()
print(f"\nGrid Search Completed in {end_time - start_time:.2f} seconds.")

# Displaying the best parameters found by Grid Search
print("\nBest Parameters from Grid Search:", grid_search.best_params_)
print(f"Best Model Accuracy (Cross-Validated): {grid_search.best_score_:.4f}")

# Building the final optimized Decision Tree model with the best settings
tree_optimized = grid_search.best_estimator_
y_pred_best = tree_optimized.predict(X_test)

# Evaluating the final optimized model
accuracy = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best, average='weighted')
recall = recall_score(y_test, y_pred_best, average='weighted')
f1 = f1_score(y_test, y_pred_best, average='weighted')

print(f"\nOptimized Decision Tree Model Performance (Grid Search):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_best))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
plt.title("Confusion Matrix - Optimized Decision Tree (Grid Search)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()