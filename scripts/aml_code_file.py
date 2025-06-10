import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")

print("AML Pipeline starts...")

df = pd.read_csv("data/SAML-D.csv")

print(df.shape)
print(df.isnull().sum())
print(df.select_dtypes(include=['object']).columns)
print(df.describe())
print(df['Is_laundering'].value_counts(normalize=True) * 100)

df['Currency_Mismatch'] = (df['Payment_currency'] != df['Received_currency']).astype(int)
df['High_Risk_Country'] = df['Sender_bank_location'].isin(['Turkey', 'Mexico', 'UAE']).astype(int)
df['Amount_Bin'] = pd.cut(df['Amount'], bins=[-np.inf, 2000, 10000, 50000, np.inf], labels=[0, 1, 2, 3]).astype(int)
df['Currency_Pair'] = df['Payment_currency'] + "→" + df['Received_currency']
df['Laundering_type'] = df['Laundering_type'].fillna('None')

df_pos = df[df["Is_laundering"] == 1]
df_neg = df[df["Is_laundering"] == 0].sample(n=150000, random_state=42)
df_sampled = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

X = df_sampled.drop(columns=['Is_laundering'])
y = df_sampled['Is_laundering']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

currency_pair_means = X_train.join(y_train).groupby('Currency_Pair')['Is_laundering'].mean()
X_train['Currency_Pair_encoded'] = X_train['Currency_Pair'].map(currency_pair_means)
X_test['Currency_Pair_encoded'] = X_test['Currency_Pair'].map(currency_pair_means).fillna(0)

for col in ['Payment_type', 'Sender_bank_location']:
    freq_map = X_train[col].value_counts()
    X_train[col + '_freq'] = X_train[col].map(freq_map)
    X_test[col + '_freq'] = X_test[col].map(freq_map)

drop_cols = ['Sender_account', 'Receiver_account', 'Payment_currency', 'Payment_type',
             'Sender_bank_location', 'Received_currency', 'Receiver_bank_location',
             'Currency_Pair', 'Laundering_type', 'Time', 'Date', 'Time_Category', 'Local_Hour']
X_train.drop(columns=drop_cols, inplace=True, errors='ignore')
X_test.drop(columns=drop_cols, inplace=True, errors='ignore')

scaler = StandardScaler()
X_train[['Amount']] = scaler.fit_transform(X_train[['Amount']])
X_test[['Amount']] = scaler.transform(X_test[['Amount']])

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                             max_depth=6, reg_alpha=1.0, reg_lambda=1.0,
                             scale_pos_weight=(len(y_train_res) - sum(y_train_res)) / sum(y_train_res),
                             random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, digits=2))

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
print("\n Overall Model Comparison:\n", results_df)
