# Credit Scoring Model

## Project Overview

This project predicts whether a person has good or bad credit risk using Machine Learning.

The project uses the German Credit dataset from the UCI Machine Learning Repository.

## Objective

The main objective is to build a Machine Learning model that classifies credit applicants into:

- Good Credit
- Bad Credit

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- UCI Machine Learning Repository

## Machine Learning Model

The project uses **Logistic Regression** for credit risk classification.

## Data Preprocessing

The following steps were performed:

1. Loaded the German Credit dataset.
2. Converted the target values into Good Credit and Bad Credit classes.
3. Applied one-hot encoding to categorical features.
4. Split the dataset into training and testing data.
5. Applied StandardScaler for feature scaling.

## Model Evaluation

The model was evaluated using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

### Results

| Metric | Score |
|---|---:|
| Accuracy | 71.00% |
| Precision | 78.08% |
| Recall | 81.43% |
| F1 Score | 79.61% |
| ROC-AUC | 0.75 |

## Confusion Matrix

The confusion matrix obtained from the test data was:

```text
[[28, 32],
 [26, 114]]