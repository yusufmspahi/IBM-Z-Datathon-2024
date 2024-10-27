from random import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

df = pd.read_csv('final_dataset-2.csv')
df = df.drop(['Datetime', 'TransitTime'], axis=1)
df = df.dropna()
targets = df['Geoeffective']
features = df.drop('Geoeffective', axis=1)

# Assume X is your feature matrix and y is your target variable
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=43)

clf = svm.SVC(kernel='rbf') 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['linear', 'poly', 'rbf']
}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
grid_search = GridSearchCV(clf, param_grid, scoring=None, cv=kfold, n_jobs=-1, verbose=3)
grid_search.fit(X_train_scaled, y_train)

# Get the best model and test it
best_svr = grid_search.best_estimator_
y_pred = best_svr.predict(X_test_scaled)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(grid_search.best_params_)
