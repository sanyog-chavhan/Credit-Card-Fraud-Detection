# Credit Card Fraud Detection with CatBoost and SHAP

### Advanced Machine Learning and Explainability for Financial Transaction Fraud Detection

---

## Project Overview

Comprehensive credit card fraud detection project leveraging state-of-the-art machine learning and explainability techniques.

Using the Kaggle Credit Card Fraud Detection dataset - a highly imbalanced dataset with 284,807 anonymised transactions and only 492 fraudulent cases (~0.172%) - this project builds robust classification models that maximise detection while minimising false positives.

Key strengths include:

- Handling class imbalance with class weighting instead of oversampling
- Employing CatBoost as the primary classifier for superior performance
- Full model interpretability using SHAP (SHapley Additive exPlanations)
- A reusable pipeline combining preprocessing and modeling for deployment

---

## Dataset Description

- **Transactions:** 284,807
- **Fraudulent cases:** 492 (~0.172%)
- **Features:** 30 (including PCA components V1–V28, plus `Time` and `Amount`)
- **Label:** `Class` - 1 indicates fraud, 0 indicates non-fraud

---

## Project Workflow

### 1. Data Preprocessing

- Load dataset and perform data integrity checks.
- Shuffle data to ensure unbiased representation.
- Log-transform and robust scale the `Amount` feature.
- Standard scale the `Time` feature.
- Split into **training**, **validation**, and **test** sets.
- Calculate class weights for imbalance handling.

### 2. Exploratory Data Analysis (EDA)

- Visualise feature distributions and correlations.
- Examine the severe class imbalance.
- Study transaction amount behavior across classes.

### 3. Model Development

- Train baseline classifiers: Logistic Regression, Random Forest.
- Train advanced gradient boosting models: XGBoost, LightGBM.
- Develop a Multi-Layer Perceptron (Neural Network).
- Build Voting Ensemble combining multiple classifiers.
- Select **CatBoost** as the final model due to superior fraud class recall and balanced precision.

### 4. Model Evaluation

- Evaluate models on multiple metrics suited for imbalanced data:
  - Recall, Precision, F1-score for fraud class
  - Accuracy and Weighted metrics
  - Average Precision (AUPRC)
- CatBoost achieves a strong balance of high recall (detection rate), precision, and PR-AUC (~0.87).

### 5. Explainability with SHAP

- Generate SHAP global feature importance and local explanations.
- Interpret key features driving fraud detection decisions.
- Demonstrate model transparency and build trust for downstream use.

---

## Saved Artifacts and Reproducibility

- Pipeline comprising all preprocessing steps and trained CatBoost classifier saved as `fraud_detection_pipeline.joblib`.
- Jupyter Notebook (`Fraud_Detection.ipynb`) contains all code, EDA, modeling, and explanation steps.

Users can easily reuse the pipeline for inference or fine-tuning with minimal setup.

---

## Technologies Used

- **Programming Language:** Python
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, SHAP
- **Modeling:** CatBoost, XGBoost, LightGBM, Random Forest, Logistic Regression, Neural Network  
- **Environment:** Jupyter, VSCode  
- **Version Control:** GitHub

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate    # on Mac/Linux
venv\Scripts\activate       # on Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter notebook
```bash
jupyter notebook Fraud_Detection.ipynb
```

### 5. (Optional) Load the trained pipeline
```python
import joblib
model = joblib.load('saved_models/fraud_detection_pipeline.joblib')
predictions = model.predict(new_data)
```

---

## Results

| Model | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) |
|--------|-------------------|----------------|------------------|
| Logistic Regression | 0.69 | 0.85 | 0.76 |
| Random Forest | 0.89 | 0.85 | 0.87 |
| LightGBM | 0.67 | 0.82 | 0.74 |
| XGBoost | 0.97 | 0.85 | 0.90 |
| XGBoost (GridSearchCV) | 0.97 | 0.85 | 0.90 |
| **CatBoost (Final)** | **0.92** | **0.87** | **0.89** |
| Neural Network | 0.70 | 0.54 | 0.61 |
| Voting Ensemble | 0.97 | 0.85 | 0.90 |

Average Precision (AUPRC) for the top performing Models
- XGBoost: 0.9129
- LightGBM: 0.6091
- CatBoost: 0.9081
- Voting Ensemble: 0.9141

AUPRC (Area Under Precision-Recall Curve) measures model ability to balance recall (detecting fraud) and precision (avoiding false alarms) across thresholds, making it the preferred metric for highly imbalanced fraud detection. It was calculated on predicted probabilities using scikit-learn’s average_precision_score on the test set.

---

## Future Improvements

- Deploy model as a REST API using **FastAPI / Flask**.
- Incorporate real-time streaming data using **Kafka**.
- Experiment with **SMOTE + Tomek Links** or **ADASYN** for controlled oversampling.
- Automate hyperparameter tuning via **Optuna**.
- Integrate **MLflow** for experiment tracking and versioning.

---

## Acknowledgements

- **Dataset:** [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Libraries:** Scikit-learn, CatBoost, XGBoost, LightGBM, Random Forest, Logistic Regression, Neural Networks, SHAP, Matplotlib, Seaborn, Pandas
- **Author:** [Sanyog Chavhan](https://github.com/sanyog-chavhan)

---

## Repository Structure

```
Fraud-Detection/
│
├── Fraud_Detection.ipynb       # Main analysis notebook
├── fraud_detection_pipeline.joblib     # Pipeline
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
    
```

---

## Author Notes

This project reflects a complete end-to-end fraud detection workflow — from **data preprocessing** to a full fledged **pipeline**, implemented in a reproducible and production-conscious way.  
It demonstrates how thoughtful model design, even without oversampling, can produce a performant and trustworthy fraud detection system.

If you find this project insightful, feel free to ⭐️ the repo or connect with me on [LinkedIn](https://www.linkedin.com/in/sanyog-chavhan).

---
