import xgboost as xgb
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
#from imblearn.over_sampling import SMOTE



data = pd.read_csv('/Users/olliehockey/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Documents/Year 2/Datathon/Imputed_DataFrame.csv')
#Data['Accel'] = pd.to_numeric(Data['Accel'].str.replace('*', '', regex=False), errors='coerce')
#Data['CentralPA'] = pd.to_numeric(Data['CentralPA'].str.replace('halo', '360', regex=False), errors='coerce')


# Features (X) - All columns except the target 'arrival_time'
X = data.drop(columns=['Datetime','TransitTime', 'Geoeffective'])

y = data['Geoeffective']
#80%-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
param_distributions = {
    'n_estimators': [50, 100, 200, 300],       # Number of boosting rounds
    'max_depth': [3, 4, 5, 6, 7, 8, 9],              # Maximum depth of trees
    'learning_rate': [0.01, 0.05, 0.1],        # Learning rate
    'subsample': [0.6, 0.8, 1.0],              # Subsample ratio for instances
    'colsample_bytree': [0.6, 0.8, 1.0],       # Subsample ratio for features
    'gamma': [0, 0.1, 0.2, 0.3],               # Minimum loss reduction for split
    'reg_alpha': [0, 0.01, 0.1],               # L1 regularization term
    'reg_lambda': [1, 1.5, 2],                 # L2 regularization term
}
#setup of model
xgboost_model = xgb.XGBClassifier(scale_pos_weight=7.95,use_label_encoder=False, eval_metric='logloss')  # logloss is typical for binary classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(xgboost_model, X, y, cv=skf, scoring='roc_auc')

random_search = RandomizedSearchCV(
    estimator=xgboost_model,                # The model to train
    param_distributions=param_distributions, # The hyperparameter space
    n_iter=50,                              # Number of random combinations to try
    scoring='accuracy',        # Scoring metric (MSE for regression)
    cv=3,                                   # 3-fold cross-validation
    verbose=2,                              # Show output during the process
    random_state=42,                        # Reproducibility
    n_jobs=-1                               # Use all available cores
)
#     n_estimators=100,    # Number of boosting rounds (trees)
#     max_depth=6,         # Maximum depth of each tree
#     learning_rate=0.1,   # Learning rate (controls the contribution of each tree)
#     subsample=0.8,       # Subsampling ratio of the training Data
#     colsample_bytree=0.8 # Subsampling ratio of the features (columns)
# )

random_search.fit(X_train, y_train)

class_distribution = data['Geoeffective'].value_counts()
print("this is", class_distribution)
# Train the XGBoost model
random_search.fit(X_train, y_train)

print(f"Best hyperparameters: {random_search.best_params_}")

# Make predictions on the test set
y_pred = random_search.best_estimator_.predict(X_test)
y_pred_proba = random_search.best_estimator_.predict_proba(X_test)[:, 1]


#smote = SMOTE(random_state=42)
#X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
#xgboost_model.fit(X_resampled, y_resampled)

# Evaluate performance using accuracy, classification report, and confusion matrix

# Detailed classification metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
threshold = 0.3
y_pred_threshold = (y_pred_proba >= threshold).astype(int)

# Confusion matrix with new threshold
print(confusion_matrix(y_test, y_pred_threshold))

# Calculate ROC AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {auc}")
# Plot confusion matrix
plt.matshow(confusion_matrix(y_test, y_pred), cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot(fpr, tpr, color='blue', lw=2)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # 45-degree line (random guess)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()