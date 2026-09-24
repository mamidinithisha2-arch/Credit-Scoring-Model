# Credit Scoring Model

## Project Overview

This project predicts whether a person is a good or bad credit risk using machine learning.

The project uses the German Credit dataset from the UCI Machine Learning Repository.

## Dataset

- Dataset: German Credit
- Number of records: 1000
- Number of original features: 20
- Target classes:
  - 1 = Good Credit
  - 0 = Bad Credit

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- UCI ML Repository

## Machine Learning Algorithm

Logistic Regression is used as the classification algorithm.

## Data Preprocessing

The following steps were performed:

1. Loaded the German Credit dataset.
2. Checked for missing values.
3. Converted the target values into binary classes.
4. Converted categorical features into numerical features using one-hot encoding.
5. Split the dataset into training and testing data.
6. Applied feature scaling using StandardScaler.

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve

## Project Structure

```text
Credit_Scoring_Model/
│
├── credit_scoring.py
├── README.md
└── venv/
```

## How to Run

Create and activate a virtual environment, then install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn ucimlrepo
```

Run the project:

```bash
python credit_scoring.py
```

## Conclusion

The project demonstrates how machine learning can be used for binary credit-risk classification. The model's performance is evaluated using multiple classification metrics and visualizations.