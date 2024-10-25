from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV as rscv
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import r2_score

df = pd.read_csv('final_dataset-2.csv')
df = df.drop('Datetime', axis=1)
df = df.dropna()
targets = df['TransitTime']
features = df.drop('TransitTime', axis=1)

# Assume X is your feature matrix and y is your target variable
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=43)

# Define the SVR model and parameters for GridSearch
svr = SVR(kernel='rbf')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'epsilon': [0.01, 0.1, 0.2, 0.5]
}

# Perform grid search with cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=43)
grid_search = GridSearchCV(svr, param_grid, scoring='r2', cv=kfold, n_jobs=-1, verbose=3)
grid_search.fit(X_train_scaled, y_train)

# Get the best model and test it
best_svr = grid_search.best_estimator_
y_pred = best_svr.predict(X_test_scaled)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"R^2 Score: {r2}")

