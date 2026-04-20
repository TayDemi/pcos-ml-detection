import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as pylt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import shap
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# DATA ANALYSIS
# Load the dataset, EDA
df = pd.read_csv('PCOS_data.csv')

# Display the first few rows, EDA
print(df.head())
# Check data types, EDA
print(df.dtypes)
# Check for duplicates, EDA
duplicates = df.duplicated().sum
print(f"Number of duplicate records: {duplicates}")  # no duplicates
# Count Unique Values
print(df['PCOS (Y/N)'].value_counts())
print(df['Weight gain(Y/N)'].value_counts())
#
#
# # VISUALIZATION
# sbn.set(style="whitegrid")
#
# # List of features to visualize (excluding the target variable)
# feats = df.columns.difference(['PCOS (Y/N)', 'Sl. No', 'Patient File No.'])
#
# # Create histograms for each feature
# for feature in feats:
#     pylt.figure(figsize=(8, 6))
#     sbn.histplot(df[feature], bins=30, kde=True)  # kde=True adds a kernel density estimate
#     pylt.title(f'Histogram of {feature}')
#     pylt.xlabel(feature)
#     pylt.ylabel('Frequency')
#     pylt.show()


# Correlation Analysis
# Convert columns to numeric, errors='coerce' will replace non-numeric values with NaN
for col in df.columns:
    if df[col].dtype == 'object':  # Check if column is of object type (likely string)
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except ValueError:
            print(f"Could not convert column '{col}' to numeric. It may contain non-numeric values.")

# Now calculate the correlation matrix
corr_mat = df.corr()
print(corr_mat)
# visualizing corr mat
pylt.figure(figsize=(12, 8))
sbn.heatmap(corr_mat, annot=True, fmt=".2f", cmap='coolwarm', square=True)
pylt.title('Correlation Matrix HM')
pylt.show()

# DATA PREPROCESSING - DATA CLEANING
# Check for missing values
missing_vals = df.isnull().sum()
print("Missing values in each column:", missing_vals)
# Forward fill missing values if necessary-assuming the missing value is the same as the value right before it and
# vice-versa for bfill
df.ffill(inplace=True)
# Remove extra periods and convert to numeric
columns_to_clean = df.columns.difference(['PCOS (Y/N)', 'Sl. No', 'Patient File No.'])
for col in columns_to_clean:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)  # Remove periods
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coercing errors to NaN
# Check for NaN values after conversion
print(df.isnull().sum())
# Fill NaNs or drop them
df.fillna(df.mean(), inplace=True)  # Fill NaNs with column's means-average
# Prepare features and target variable
# Proceed with feature selection
X = df.drop(['PCOS (Y/N)', 'Sl. No', 'Patient File No.'], axis=1)
Y = df['PCOS (Y/N)']

# RFE code
# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Drop rows with missing values
X_train_clean = X_train.dropna()
Y_train_clean = Y_train[X_train.index.isin(X_train_clean.index)]
#
#
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
# Initialize the model - RANDOM FOREST and perform RFE
RF_model = RandomForestClassifier()

# Initialize variables to track the best number of features
best_n_features = 0
best_accuracy = 0

# Test RFE for different numbers of features
for n_features in range(1, X.shape[1] + 1):
    rfe = RFE(estimator=RF_model, n_features_to_select=n_features)
    rfe.fit(X_train, Y_train)

    # Predict on the test set
    Y_predict = rfe.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_predict)

    # Update best features if the current accuracy is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_features = n_features

# Final RFE with the best number of features
final_rfe = RFE(estimator=RF_model, n_features_to_select=best_n_features)
final_rfe.fit(X_train, Y_train)

# Convert any non-numeric columns to numeric, coercing errors to NaN
for col in X_train.columns:
    # Check if the column is not already a numeric type (int or float)
    if not pd.api.types.is_numeric_dtype(X_train[col]):
        print(f"Converting non-numeric column: {col}")
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        # After coercing, check for remaining NaNs in this column and fill them
        if X_train[col].isnull().any():
            print(f"NaN values found in column '{col}' after conversion. Filling with mean.")
            # For simplicity here, we use the column mean.
            X_train[col].fillna(X_train[col].mean(), inplace=True)
# Apply the same conversion to X_test to maintain consistency
print("Checking X_test data types before RFE:")
print(X_test.dtypes)
for col in X_test.columns:
    if not pd.api.types.is_numeric_dtype(X_test[col]):
        print(f"Converting non-numeric column in X_test: {col}")
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        # Fill NaNs in X_test using the mean from X_train to avoid data leakage
        # Ensure that the column exists in the original X_train and has a non-NaN mean
        if col in X_train.columns and not np.isnan(X_train[col].mean()):
            X_test[col].fillna(X_train[col].mean(), inplace=True)
        else:
            # Fallback if mean from X_train is NaN or column not in X_train (shouldn't happen)
            X_test[col].fillna(X_test[col].mean(), inplace=True)

# Ensure no infinite values after conversions and fillings
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(X_train.mean(), inplace=True)  # Fill any NaNs created by inf replacement
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
# Use mean from X_train for filling NaNs in X_test
for col in X_test.columns:
    if X_test[col].isnull().any() and col in X_train.columns and not np.isnan(X_train[col].mean()):
        X_test[col].fillna(X_train[col].mean(), inplace=True)
    elif X_test[col].isnull().any():
        # Fallback if mean from X_train is NaN or column not in X_train
        X_test[col].fillna(X_test[col].mean(), inplace=True)
print("Finished data type conversion and NaN handling.")
print("Final X_train data types:")
print(X_train.dtypes)
print("Final X_test data types:")
print(X_test.dtypes)

# Get the selected features
sel_feats = X.columns[final_rfe.support_]
print("Best Number of Features:", best_n_features)
print("Best Accuracy:", best_accuracy)
print("Selected Features:")
print(sel_feats)

# Filter training and test sets to include only selected features
X_train = X_train[sel_feats]
X_test = X_test[sel_feats]
#
# Retrain model with selected feats
RF_model.fit(X_train, Y_train)

from sklearn.model_selection import GridSearchCV
# Perform Grid Search on selected feats
par_g = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
g_search = GridSearchCV(estimator=RF_model, param_grid=par_g, cv=5)
# Fitting Grid Search into the training data
g_search.fit(X_train, Y_train)

# # Input data
# RF_model, sel_feats = joblib.load('PCOS_RF_MOD2.pkl')
#
# input_values = [[20, 84, 30.3, 2, 7.86, 2.90, 2.71, 2.10, 30.07, 1, 1, 0, 5, 4, 9]]
#
# input_df = pd.DataFrame(input_values, columns=sel_feats)
# # Make a prediction
# prediction = RF_model.predict(input_df)
#
# if prediction == 0:
#     print("PCOS Prediction: PCOS is not present")
# else:
#     print("PCOS Prediction: PCOS is present")

# Evaluation_RF
Y_predi = RF_model.predict(X_test)
print("RF Accuracy:", accuracy_score(Y_test, Y_predi))
print("RF Confusion Matrix:\n", confusion_matrix(Y_test, Y_predi))
print("RF Classification Report:\n", classification_report(Y_test, Y_predi))


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    pylt.figure(figsize=(8, 6))
    sbn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No PCOS', 'PCOS'],
                yticklabels=['No PCOS', 'PCOS'])
    pylt.title(f'Confusion Matrix - {model_name}')
    pylt.xlabel('Predicted')
    pylt.ylabel('True')
    pylt.show()


Y_predi = RF_model.predict(X_test)
print("Accuracy_RF:", accuracy_score(Y_test, Y_predi))
plot_confusion_matrix(Y_test, Y_predi, "Random Forest")

from sklearn.svm import SVC
# SUPPORT VECTOR MACHINE
SVM_model = SVC(kernel='linear')  # You can also try other kernels like 'rbf'

# Filter training and test sets to include only selected features
X_train = X_train[sel_feats]
X_test = X_test[sel_feats]

# Retrain model with selected feats
SVM_model.fit(X_train, Y_train)
# # Perform Grid Search on selected feats
# par_g = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
# g_search = GridSearchCV(estimator=SVM_model, param_grid=par_g, cv=5)
# g_search.fit(X_train, Y_train)

# Evaluation_SVM
Y_predi = SVM_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(Y_test, Y_predi))
print("SVM Confusion Matrix:\n", confusion_matrix(Y_test, Y_predi))
print("SVM Classification Report:\n", classification_report(Y_test, Y_predi))

# Confusion Matrix SVM
Y_predi = SVM_model.predict(X_test)
print("Accuracy_SVM:", accuracy_score(Y_test, Y_predi))
plot_confusion_matrix(Y_test, Y_predi, "Support Vector Machine")

from sklearn.linear_model import LogisticRegression
# LOGISTIC REGRESSION
LR_model = LogisticRegression(max_iter=10000, solver='saga')

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

LR_model.fit(X_train, Y_train)

# Evaluation_LR
Y_predi = LR_model.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, Y_predi))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_predi))
print("Classification Report:\n", classification_report(Y_test, Y_predi))

# Confusion Matrix
Y_predi = LR_model.predict(X_test_scaled)
print("Accuracy_LR:", accuracy_score(Y_test, Y_predi))
plot_confusion_matrix(Y_test, Y_predi, "Logistic Regression")

# Cross-Validation SCORES
RF_scores = cross_val_score(RF_model, X_train, Y_train, cv=10)
LR_scores = cross_val_score(LR_model, X_train, Y_train, cv=10)
SVM_scores = cross_val_score(SVM_model, X_train, Y_train, cv=10)

print(f"RF Cross-Validation Scores: {RF_scores}")
print(f"LR Cross-Validation Scores: {LR_scores}")
print(f"SVM Cross-Validation Scores: {SVM_scores}")
print(f"RF Mean Accuracy: {RF_scores.mean():.2f} ± {RF_scores.std():.2f}")
print(f"LR Mean Accuracy: {LR_scores.mean():.2f} ± {LR_scores.std():.2f}")
print(f"SVM Mean Accuracy: {SVM_scores.mean():.2f} ± {SVM_scores.std():.2f}")
# # SHAP Visualization
# # SHAP visualization_RF
# explainer = shap.Explainer(RF_model, X_train)
# SHAP = explainer(X_test)
# print("SHAP_VAL_SHAPE:", SHAP)  # Check if shap_values contains data
# print(X_test.head())  # FORMAT AND CONTEXT OF X_TEST:
# shap.summary_plot(SHAP, X_test)
# # SHAP visualization_SVM
# explainer = shap.Explainer(SVM_model, X_train)
# SHAP = explainer(X_test)
# print("SHAP_VAL_SHAPE:", SHAP)  # Check if shap_values contains data
# print(X_test.head())  # FORMAT AND CONTEXT OF X_TEST:
# shap.summary_plot(SHAP, X_test)
# # SHAP visualization_LR
# explainer = shap.Explainer(LR_model, X_train)
# SHAP = explainer(X_test)
# print("SHAP_VAL_SHAPE:", SHAP)  # Check if shap_values contains data
# print(X_test.head())  # FORMAT AND CONTEXT OF X_TEST:
# shap.summary_plot(SHAP, X_test)
# SHAP Visualization
# RF_model = model
# SVM_model = SVM_model
# LR_model = LR_model
# # SHAP visualization_RF
# explainer = shap.Explainer(RF_model, X_train)
# SHAP = explainer(X_test)
# # print("RF_SHAP_VAL_SHAPE:", SHAP)  # Check if shap_values contains data
# print(X_test.head())  # FORMAT AND CONTEXT OF X_TEST:
# shap.summary_plot(SHAP, X_test)
# # SHAP visualization_SVM
# explainer = shap.Explainer(SVM_model, X_train)
# SHAP = explainer(X_test)
# print("SVM_SHAP_VAL_SHAPE:", SHAP)  # Check if shap_values contains data
# print(X_test.head())  # FORMAT AND CONTEXT OF X_TEST:
# shap.summary_plot(SHAP, X_test)
# # SHAP visualization_LR
# explainer = shap.Explainer(LR_model, X_train)
# SHAP = explainer(X_test)
# print("LR_SHAP_VAL_SHAPE:", SHAP)  # Check if shap_values contains data
# print(X_test.head())  # FORMAT AND CONTEXT OF X_TEST:
# shap.summary_plot(SHAP, X_test)
#
# Save model as a new file and selected features
joblib.dump((RF_model, sel_feats), 'PCOS_MOD2.pkl')

# Input data
RF_model, sel_feats = joblib.load('PCOS_RF_MOD2.pkl')

input_values = [[20, 84, 30.3, 2, 7.86, 2.90, 2.71, 2.10, 30.07, 1, 1, 0, 5, 4, 9]]

input_df = pd.DataFrame(input_values, columns=sel_feats)
# Make a prediction
prediction = RF_model.predict(input_df)

if prediction == 0:
    print("PCOS Prediction: PCOS is not present")
else:
    print("PCOS Prediction: PCOS is present")
