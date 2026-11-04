# CodeAlpha_CreditScoring

Machine Learning Internship at CodeAlpha, Task 1: Credit Scoring Model.

## Objective
Predict an individual's creditworthiness (good or risky borrower) from past financial data.

## Dataset
German Credit dataset (OpenML), loaded via scikit-learn.
- 1000 loan applicants and 20 features such as loan amount, duration, credit history, savings, employment and age
- Target: 1 = Risky (bad), 0 = Good
- Class split: about 70% good and 30% risky

## Approach
1. Load and explore the data
2. Feature engineering: credit_per_month, log_credit_amount, is_young, long_duration, credit_age_ratio
3. Train-test split (80% / 20%, stratified)
4. Preprocessing: StandardScaler for numeric and OneHotEncoder for categorical features
5. Train Logistic Regression, Decision Tree and Random Forest (balanced class weights)
6. Evaluate with Accuracy, Precision, Recall, F1-Score and ROC-AUC
7. Plot ROC curve, confusion matrix and feature importance

## Results

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------------|----------|-----------|--------|--------|---------|
| Logistic Regression | 0.7450   | 0.5506    | 0.8167 | 0.6577 | 0.8126  |
| Decision Tree       | 0.6150   | 0.4066    | 0.6167 | 0.4901 | 0.6655  |
| Random Forest       | 0.7600   | 0.6765    | 0.3833 | 0.4894 | 0.8050  |

**Best model: Logistic Regression** (ROC-AUC 0.8126, Recall 0.82).

## Conclusion
- Logistic Regression was the best model overall, correctly identifying 49 of 60 risky applicants in the test set.
- Random Forest had slightly higher accuracy but missed many risky applicants (Recall 0.38).
- In credit scoring, Recall and ROC-AUC matter more than accuracy, since approving a risky borrower is costly.
- Trade-off: Precision is 0.55, so some good applicants are flagged as risky.
- The top 3 important features were all engineered features: credit_per_month, log_credit_amount and credit_age_ratio, followed by credit_amount, checking account status, age and duration.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

## How to Run
1. Install the libraries: `pip install scikit-learn pandas matplotlib seaborn jupyter`
2. Open the notebook: `jupyter notebook CodeAlpha_Task1_CreditScoring.ipynb`
3. Run all cells (Kernel, then Restart & Run All)

## Author
Parmar sejal, Machine Learning Intern at CodeAlpha
