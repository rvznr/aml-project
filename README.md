# Anti-Money Laundering Detection Project

This project applies supervised machine learning techniques to detect suspicious financial transactions. It is based on a synthetic dataset and was developed as part of an academic research project. The implementation includes end-to-end steps from data preprocessing and exploratory data analysis to model training and evaluation. The accompanying report provides detailed analysis and insights.

## Report

The full academic report is included as a separate file:

**AntiMoneyLaunderingReport.pdf**

Contents of the report include:

- Background and motivation
- Dataset characteristics
- Exploratory data analysis
- Feature engineering
- Machine learning models and results
- Observations and conclusions

##  Project Structure

aml-project/
├── data/ # Input dataset (excluded from version control)
│ └── SAML-D.csv
├── scripts/
│ └── aml_code_file.py # Main machine learning pipeline
├── eda_outputs/ # Plots and visual outputs
├── AntiMoneyLaunderingReport.pdf # Final academic report
├── README.md
├── .gitignore
└── requirements.txt

---

##  How to Run

1. Install required dependencies:

pip install -r requirements.txt

2. Place the dataset in the following path:

data/SAML-D.csv


3. Run the main pipeline script:

python scripts/aml_code_file.py


## Models Used

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

All models were evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Sample Output

--- XGBoost ---
precision recall f1-score support
0 0.99 0.95 0.97 30000
1 0.21 0.78 0.33 100

--- Overall Model Comparison ---
            Model     Accuracy  Recall  Precision  F1 Score  ROC-AUC
	0 Random Forest 0.96     0.85     0.35       0.50     0.91
	1 XGBoost       0.95     0.82     0.31       0.45     0.89
	2 LogisticREG   0.93     0.75     0.22       0.34     0.86

## Team Members

- Hilal Çalışkan  
- Nefise Hatun Demir  
- Ravza Nur Şişik

## Data Disclaimer

The dataset (`SAML-D.csv`) is not included in the repository for privacy and compliance reasons. To run the project, you must obtain the dataset from the official source and place it in the `data/` directory.

## License

This repository is intended for academic and research use only. Commercial use is prohibited without explicit permission.
