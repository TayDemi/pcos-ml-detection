import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib


# DATA ANALYSIS
# Load the dataset
df = pd.read_csv('dataset-project.csv')
# Display the first few rows
print(df.head())
# Check data types
print(df.dtypes)


# Feature Engineering: Create Age Group and BMI Category
def categorize_age(age):
    if age < 20:
        return 'Teen'
    elif 20 <= age < 30:
        return 'Young Adult'
    elif 30 <= age < 40:
        return 'Adult'
    else:
        return 'Older Adult'


def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 24.9:
        return 'Normal'
    elif 25 <= bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obese'


#
# Apply the categorization functions
df['Age_Group'] = df['Age'].apply(categorize_age)
df['BMI_Category'] = df['BMI'].apply(categorize_bmi)

# Display the first few rows of the updated DataFrame
print(df.head())

# Display the list of all columns to see the newly added features
print("Updated feature columns:")
print(df.columns)

# Convert categorical variables to numeric using one-hot encoding
df = pd.get_dummies(df, columns=['Age_Group', 'BMI_Category'], drop_first=True)

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate records: {duplicates}")
# Count Unique Values
print(df['PCOS_Diagnosis'].value_counts())
#
# # VISUALIZATION
# sns.set(style="whitegrid")
#
# # List of features to visualize (excluding the target variable)
# feats = df.columns.difference(['PCOS_Diagnosis'])
#
# # Create histograms for each feature
# for feature in feats:
#     plt.figure(figsize=(8, 6))
#     sns.histplot(df[feature], bins=30, kde=True)
#     plt.title(f'Histogram of {feature}')
#     plt.xlabel(feature)
#     plt.ylabel('Frequency')
#     plt.show()

# Correlation Analysis
# Convert columns to numeric
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate the correlation matrix
corr_mat = df.corr()
print(corr_mat)
# Visualizing correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# DATA PREPROCESSING - DATA CLEANING
# Check for missing values
missing_vals = df.isnull().sum()
print("Missing values in each column:", missing_vals)

# Forward fill missing values if necessary
df.ffill(inplace=True)
#
# Prepare features and target variable
X = df.drop(['PCOS_Diagnosis'], axis=1)
Y = df['PCOS_Diagnosis']

# Data Splitting
# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Drop rows with missing values
X_train_clean = X_train.dropna()
Y_train_clean = Y_train[X_train.index.isin(X_train_clean.index)]

#  Model fitting
# RANDOM FOREST with RFE
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

# Get the selected features
sel_feats = X.columns[final_rfe.support_]
print("Best Number of Features:", best_n_features)
print("Best Accuracy:", best_accuracy)
print("Selected Features:")
print(sel_feats)
#
# Perform Grid Search on selected features
par_g = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
g_search = GridSearchCV(estimator=RF_model, param_grid=par_g, cv=5)
g_search.fit(X_train, Y_train)

# Filter training and test sets to include only selected features
X_train = X_train[sel_feats]
X_test = X_test[sel_feats]

# Retrain model with selected features
RF_model.fit(X_train, Y_train)
# Perform Grid Search on selected feats for RF
par_RF = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
RF_search = GridSearchCV(estimator=RF_model, param_grid=par_RF, cv=5)
# Fitting Grid Search into the training data
RF_search.fit(X_train, Y_train)
print('Best Params for RF:', RF_search.best_params_)
#
# Evaluation_RF
Y_predi = RF_model.predict(X_test)
print("Accuracy_RF:", accuracy_score(Y_test, Y_predi))
print("Confusion Matrix_RF:\n", confusion_matrix(Y_test, Y_predi))
print("Classification Report_RF:\n", classification_report(Y_test, Y_predi))


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No PCOS', 'PCOS'],
                yticklabels=['No PCOS', 'PCOS'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


Y_predi = RF_model.predict(X_test)
print("Accuracy_RF:", accuracy_score(Y_test, Y_predi))
plot_confusion_matrix(Y_test, Y_predi, "Random Forest")


# SUPPORT VECTOR MACHINE
SVM_model = SVC(kernel='linear')

# Filter training and test sets to include only selected features
X_train = X_train[sel_feats]
X_test = X_test[sel_feats]

# Retrain model with selected features
SVM_model.fit(X_train, Y_train)

# Retrain model with selected feats
SVM_model.fit(X_train, Y_train)
# Perform Grid Search on selected feats for SVM
par_SVM = {'C': [0.1, 1, 10, 20]}
SVM_search = GridSearchCV(estimator=SVM_model, param_grid=par_SVM, cv=5)
# Fitting Grid Search into the training data
SVM_search.fit(X_train, Y_train)
print('Best Params for SVM:', SVM_search.best_params_)

# Evaluation_SVM
Y_predi = SVM_model.predict(X_test)
print("Accuracy_SVM:", accuracy_score(Y_test, Y_predi))
print("Confusion Matrix_SVM:\n", confusion_matrix(Y_test, Y_predi))
print("Classification Report_SVM:\n", classification_report(Y_test, Y_predi))

# Confusion Matrix SVM
Y_predi = SVM_model.predict(X_test)
print("Accuracy_SVM:", accuracy_score(Y_test, Y_predi))
plot_confusion_matrix(Y_test, Y_predi, "Support Vector Machine")
#
#
# LOGISTIC REGRESSION
LR_model = LogisticRegression(max_iter=10000, solver='saga')

# Perform Grid Search on selected feats for LR
par_LR = {'C': [0.1, 1, 10, 20]}
LR_search = GridSearchCV(estimator=LR_model, param_grid=par_LR, cv=5)
# Fitting Grid Search into the training data
LR_search.fit(X_train, Y_train)
print('Best Params for RF:', LR_search.best_params_)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

LR_model.fit(X_train_scaled, Y_train)

# Evaluation_LR
Y_predi = LR_model.predict(X_test_scaled)
print("Accuracy_LR:", accuracy_score(Y_test, Y_predi))
print("Confusion Matrix_LR:\n", confusion_matrix(Y_test, Y_predi))
print("Classification Report_LR:\n", classification_report(Y_test, Y_predi))

# Confusion Matrix
Y_predi = LR_model.predict(X_test_scaled)
print("Accuracy_LR:", accuracy_score(Y_test, Y_predi))
plot_confusion_matrix(Y_test, Y_predi, "Logistic Regression")

# Cross-Validation
# RF_scores = cross_val_score(RF_model, X_train, Y_train, cv=10)
# LR_scores = cross_val_score(LR_model, X_train, Y_train, cv=10)
# SVM_scores = cross_val_score(SVM_model, X_train, Y_train, cv=10)
#
#
# print("/nCross-Validation Scores")
# rf_scores = cross_val_score(RF_model, X_train_scaled, Y_train)
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
#
#
# # # SHAP Visualization
# # # SHAP RF_Visualization
# # explainer = shap.Explainer(RF_model, X_train)
# # SHAP = explainer(X_test)
# # shap.summary_plot(SHAP, X_test)
# #
# # # SHAP LR_Visualization
# # explainer = shap.Explainer(LR_model, X_train)
# # SHAP = explainer(X_test)
# # shap.summary_plot(SHAP, X_test)
# #
# # # SHAP SVM_Visualization
# # explainer = shap.Explainer(SVM_model, X_train)
# # SHAP = explainer(X_test)
# # shap.summary_plot(SHAP, X_test)
# #
# # # Save model as a new file and selected features
# joblib.dump((RF_model, sel_feats), 'PCOS_RF_MODEL.pkl')
# joblib.dump((LR_model, sel_feats), 'PCOS_LR_MODEL.pkl')
# joblib.dump((SVM_model, sel_feats), 'PCOS_SVM_MODEL.pkl')
#
# # Input data for prediction
# RF_model, sel_feats = joblib.load('PCOS_RF_MODEL.pkl')
# LR_model, sel_feats = joblib.load('PCOS_LR_MODEL.pkl')
# SVM_model, sel_feats = joblib.load('PCOS_SVM_MODEL.pkl')
# input_values = [[24, 34.7, 1, 25.2, 20]]  # Adjust based on selected features
# input_df = pd.DataFrame(input_values, columns=sel_feats)
# # Make a prediction
# prediction = RF_model.predict(input_df)
# if prediction == 0:
#     print("PCOS Prediction: PCOS is not present")
# else:
#     print("PCOS Prediction: PCOS is present")
