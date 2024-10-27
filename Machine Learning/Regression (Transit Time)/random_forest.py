# Import necessary libraries
import pandas as pd
import numpy as np

# For Data preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# For model building and evaluation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# For plotting graphs
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# ================================
# Data Loading and Preprocessing
# ================================

# Load Data from CSV file
data = pd.read_csv('/Users/olliehockey/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Documents/Year 2/Datathon/bootstrapped_augmented_dataset.csv')

# Replace all occurrences of '9.99', '999.9', and '9999999' with NaN
data.replace([9.99, 999.9, 9999999], np.nan, inplace=True)

# Drop rows with missing values
data.dropna(inplace=True)

# Convert 'Accel' and 'CentralPA' to numeric, handling non-numeric entries
data['Accel'] = pd.to_numeric(data['Accel'], errors='coerce')
data['CentralPA'] = pd.to_numeric(data['CentralPA'].replace('Halo', '360'), errors='coerce')

# Drop any rows with NaN values after conversion
data.dropna(inplace=True)

# Separate features (X) and target variable (y)
X = data.drop(columns=['Datetime', 'TransitTime', 'Geoeffective'])
y = data['TransitTime']

# Handle missing values using SimpleImputer (not necessary here since we've dropped NaNs)
# imputer = SimpleImputer(strategy="mean")
# X_imputed = imputer.fit_transform(X)

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================
# Data Splitting
# =====================================

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =====================================
# Model Training and Hyperparameter Tuning
# =====================================

# Define the Random Forest regressor
rfr = RandomForestRegressor(random_state=42)

# Define the hyperparameter grid for RandomizedSearchCV
param_dist = {
    "n_estimators": [100, 200, 500],            # Number of trees
    "max_depth": [None, 10, 20, 30],            # Depth of each tree
    "min_samples_split": [2, 5, 10],            # Minimum samples required to split a node
    "min_samples_leaf": [1, 2, 4],              # Minimum samples required at each leaf node
    "max_features": ["sqrt", "log2"],           # Number of features to consider at each split
    "bootstrap": [True, False],                 # Whether bootstrap samples are used
}

# Use KFold cross-validator for regression
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform Randomized Search Cross-Validation to find the best hyperparameters
random_search = RandomizedSearchCV(
    estimator=rfr,
    param_distributions=param_dist,
    n_iter=20,                   # Reduced number of iterations for efficiency
    scoring="neg_mean_squared_error",  # Use appropriate scoring for regression
    cv=kf,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit the model on the training Data
random_search.fit(X_train, y_train)

# Retrieve the best model from RandomizedSearchCV
best_model = random_search.best_estimator_

# =====================================
# Model Evaluation
# =====================================

# Make predictions on the test Data
y_pred = best_model.predict(X_test)

# Evaluate the model using regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Mean Squared Error (MSE): {:.2f}".format(mse))
print("Root Mean Squared Error (RMSE): {:.2f}".format(rmse))
print("Mean Absolute Error (MAE): {:.2f}".format(mae))
print("R-squared (R2 Score): {:.2f}".format(r2))

# =====================================
# Plotting Predicted vs Actual Values
# =====================================

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Transit Time")
plt.ylabel("Predicted Transit Time")
plt.title("Actual vs Predicted Transit Time")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.grid(True)
plt.show()

# Plotting Residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.xlabel("Predicted Transit Time")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Transit Time")
plt.axhline(0, color='k', linestyle='--', lw=2)
plt.grid(True)
plt.show()
