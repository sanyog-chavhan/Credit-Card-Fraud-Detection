# Fraud Detection using Machine Learning

### A data-driven approach to identifying fraudulent transactions with CatBoost and Explainable AI (SHAP)

---

## Project Overview

This project focuses on **credit card fraud detection** using advanced machine learning and interpretability techniques.  
The dataset used is the **Kaggle Credit Card Fraud Detection Dataset**, containing anonymised transaction features for over **284,000 transactions**, of which only **492 are fraudulent** (~0.172% of the total).

The objective is to **detect fraudulent activity** in highly imbalanced financial datasets while ensuring **model robustness**, **scalability**, and **interpretability**.  
Instead of relying on oversampling methods like SMOTE, this project optimises model performance through **class weighting**, **shuffling**, and **careful feature analysis**.

---

## Key Features

- Handles **class imbalance** using **computed class weights** (`scale_pos_weight`) instead of oversampling.
- Builds multiple models (baseline to advanced) and identifies the **CatBoost Classifier** as the most effective.
- Implements **shuffling** before model training to ensure unbiased representation.
- Provides detailed **SHAP-based feature interpretation** for transparency and explainability.
- Evaluates models on metrics beyond accuracy, focusing on **Recall**, **Precision**, **F1-score**, and **AUPRC**.

---

## Dataset

**Source:** [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)

- **Transactions:** 284,807  
- **Fraud cases:** 492 (0.172%)  
- **Features:** 30 (V1–V28 anonymised PCA components, plus `Time` and `Amount`)  
- **Target:** `Class` → 1 = Fraud, 0 = Legitimate  

> Dataset is heavily imbalanced, making this a challenging binary classification problem.

---

## Workflow

### 1. Data Preprocessing
- Loaded the dataset using `pandas` and performed integrity checks.
- Normalised features where appropriate using StandardScaler, RobustScaler and Log Transform.
- **Shuffled** the dataset to remove any temporal bias.
- Split into training, validation, and test sets (stratified on target class).
- Calculated **`scale_pos_weight`** to handle imbalance dynamically.
  
### 2. Exploratory Data Analysis (EDA)
- Explored data distribution for both classes using histograms and boxplots.
- Analysed feature correlation and significance to identify informative components.
- Visualised **class imbalance ratio** and **amount distribution** across transactions.

### 3. Model Building
Several models were trained and evaluated:
- Logistic Regression (Baseline)
- Random Forest Classifier
- XGBoost Classifier
- Gradient Boosting Classifier
- Multi-Layer Perceptron Neural Network
- Ensemble Model
- **CatBoost Classifier (Final Model)**

**Why CatBoost?**  
CatBoost demonstrated superior performance on precision-recall trade-off, handled imbalance effectively with `scale_pos_weight`, and required minimal manual feature encoding.

### 4. Model Evaluation
Metrics used:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **AUPRC**

Performance summary (CatBoost final model):
| Metric | Score |
|:--|--:|
| Accuracy | ~99.9% |
| Precision | High (low false positives) |
| Recall | Strong recall on minority class |
| AUPRC | ~0.99 |

The CatBoost model achieved a **balanced performance** across metrics while maintaining interpretability and stability.

### 5. Explainability with SHAP
Explainability was a key part of this project:
- Used **SHAP (SHapley Additive exPlanations)** to visualise and interpret feature contributions.
- Generated **feature importance plots** and **SHAP summary plots**.
- Identified key features influencing fraud prediction decisions, aiding trust in model deployment.

---

## Saved Artifacts

- Trained and tuned **CatBoost pipeline** saved as a serialised model (`.pkl` or `.joblib`).
- Notebook includes cell outputs to reproduce pipeline training and evaluation.

> You can load the pipeline directly for inference or further fine-tuning.

---

## Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Programming | Python (Jupyter Notebook) |
| Data Handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, SHAP |
| Machine Learning | Scikit-learn, XGBoost, CatBoost |
| Environment | Jupyter, Anaconda, VS Code |
| Version Control | Git, GitHub |

---

## How to Run This Project

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Fraud-Detection.git
cd Fraud-Detection
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

## Results & Visuals

| Model | ROC-AUC | Recall | F1-score |
|:------|:--------:|:-------:|:---------:|
| Logistic Regression | 0.95 | Moderate | Good |
| Random Forest | 0.97 | High | High |
| XGBoost | 0.98 | High | High |
| **CatBoost (Final)** | **0.99** | **Excellent** | **Excellent** |

### Example SHAP Plot:
*(Generated in notebook)*  
Feature importance visualisation showing the top drivers influencing fraud prediction.

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
