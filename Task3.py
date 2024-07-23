# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load dataset
data = pd.read_csv('customer_data.csv')

# Data cleaning and preprocessing
# Handling missing values
data.fillna(method='ffill', inplace=True)

# Splitting features and target
X = data.drop('churn', axis=1)
y = data['churn']

# Define categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipelines
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
# Logistic Regression
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression())])

# Random Forest
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier())])

# Training Logistic Regression
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
print('Logistic Regression:')
print(classification_report(y_test, y_pred_lr))
print('Logistic Regression ROC-AUC:', roc_auc_score(y_test, y_pred_lr))

# Training Random Forest
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
print('Random Forest:')
print(classification_report(y_test, y_pred_rf))
print('Random Forest ROC-AUC:', roc_auc_score(y_test, y_pred_rf))

# Model Evaluation and Fine-Tuning
# Hyperparameter tuning for Random Forest
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

y_pred_best_rf = best_rf_model.predict(X_test)
print('Best Random Forest:')
print(classification_report(y_test, y_pred_best_rf))
print('Best Random Forest ROC-AUC:', roc_auc_score(y_test, y_pred_best_rf))

# Save the best model
joblib.dump(best_rf_model, 'best_rf_model.pkl')

# Example of loading the model
loaded_model = joblib.load('best_rf_model.pkl')
