import xgboost as xgb
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



data = pd.read_csv('/Users/olliehockey/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Documents/Year 2/Datathon/bootstrapped_augmented_dataset.csv')
#data['Accel'] = pd.to_numeric(data['Accel'].str.replace('*', '', regex=False), errors='coerce')
#data['CentralPA'] = pd.to_numeric(data['CentralPA'].str.replace('*', '', regex=False), errors='coerce')
print(data.head())
data.replace([9.99, 999.9, 9999999], np.nan, inplace=True)

# Features (X) - All columns except the target 'arrival_time'
X = data.drop(columns=['Datetime','TransitTime', 'Geoeffective'])

y = data['TransitTime']
#80%-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_distributions = {
    'n_estimators': [50, 100, 200, 300],       # Number of boosting rounds
    'max_depth': [3, 4, 5, 6, 7],              # Maximum depth of trees
    'learning_rate': [0.01, 0.05, 0.1],        # Learning rate
    'subsample': [0.6, 0.8, 1.0],              # Subsample ratio for instances
    'colsample_bytree': [0.6, 0.8, 1.0],       # Subsample ratio for features
    'gamma': [0, 0.1, 0.2, 0.3],               # Minimum loss reduction for split
    'reg_alpha': [0, 0.01, 0.1],               # L1 regularization term
    'reg_lambda': [1, 1.5, 2],                 # L2 regularization term
}
#setup of model
xgboost_model = xgb.XGBRegressor()

random_search = RandomizedSearchCV(
    estimator=xgboost_model,                # The model to train
    param_distributions=param_distributions, # The hyperparameter space
    n_iter=50,                              # Number of random combinations to try
    scoring='neg_mean_squared_error',        # Scoring metric (MSE for regression)
    cv=3,                                   # 3-fold cross-validation
    verbose=2,                              # Show output during the process
    random_state=42,                        # Reproducibility
    n_jobs=-1                               # Use all available cores
)
#     n_estimators=100,    # Number of boosting rounds (trees)
#     max_depth=6,         # Maximum depth of each tree
#     learning_rate=0.1,   # Learning rate (controls the contribution of each tree)
#     subsample=0.8,       # Subsampling ratio of the training data
#     colsample_bytree=0.8 # Subsampling ratio of the features (columns)
# )

# Train the XGBoost model
random_search.fit(X_train, y_train)

print(f"Best hyperparameters: {random_search.best_params_}")

# Make predictions on the test set
y_pred = random_search.best_estimator_.predict(X_test)

# Evaluate performance using RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error:", mse)
print(f"Root Mean Squared Error:", rmse)
print(y_pred,y)

# Plot actual vs predicted CME arrival times
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # 45-degree line (ideal prediction line)
plt.xlabel('Actual CME Arrival Time')
plt.ylabel('Predicted CME Arrival Time')
plt.title('Actual vs Predicted CME Arrival Time')
plt.show()