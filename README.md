# PCOS Detection using Machine Learning

## Overview

This project focuses on the early detection of Polycystic Ovarian Syndrome (PCOS) using machine learning techniques. PCOS is a common endocrine disorder affecting women of reproductive age, and early detection can significantly improve treatment outcomes.

## Objective

To build and evaluate machine learning models that can accurately predict the presence of PCOS based on clinical and hormonal data.

## Dataset

The dataset used was sourced from kaggle contains medical, hormonal, and physical attributes of patients.


## Models Used

* Support Vector Machine (SVM)
* Logistic Regression
* Random Forest

## Methodology

* Data cleaning and preprocessing
* Feature selection
* Model training and evaluation
* Performance comparison

## Results
Using RFE, for PCOS_data the model achieved 99, 90, and 88 %  accuracy in RF, SVM and LR respectively 
while in dataset-project the achieved 88, 87, and 70 % accuracy for RF, SVM and LR respectively. Among the models, 
Random Forest consistently outperformed others, achieving high predictive accuracy across both datasets. 

## Tools & Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib / Seaborn

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/pcos-ml-detection.git
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:

   ```bash
   jupyter notebook
   ```

## Future Improvements

* Hyperparameter tuning
* Deployment as a web app
* Integration with real-time clinical systems

---

## Author

Oluwademilade Taylor
