import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn import metrics

df = pd.read_csv('final_dataset-2.csv')
df = df.drop(['Datetime', 'TransitTime'], axis=1)
df = df.dropna()
targets = df['Geoeffective']
features = df.drop('Geoeffective', axis=1)

# Assume X is your feature matrix and y is your target variable
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=43)

grid={
    "C":np.logspace(-3,3,7), 
    "penalty":["l1","l2"]
    }

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
logreg = LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=kfold)
logreg_cv.fit(X_train, y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

y_pred = logreg_cv.best_estimator_.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
