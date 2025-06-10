#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##LIBRARIES##
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings


# In[ ]:


df = pd.read_csv("data/SAML-D.csv")

###EDA###

print(df.shape)
print(df.isnull().sum())
print(df.select_dtypes(include=['object']).columns)
print(df.describe())
print(df['Is_laundering'].value_counts(normalize=True) * 100)

labels = ['Not Suspicious', 'Suspicious']
sizes = df['Is_laundering'].value_counts(normalize=True) * 100
colors = ['#66b3ff', '#ff6666']

plt.figure(figsize=(5, 5))
plt.pie(sizes, labels=labels, autopct='%1.2f%%', colors=colors, startangle=90, wedgeprops=dict(width=0.4))
plt.title("Is_laundering Class Distribution")
plt.savefig("eda_outputs/class_distribution_donut.png")
plt.show()


df['Currency_Mismatch'] = (df['Payment_currency'] != df['Received_currency']).astype(int)
mismatch_ratio = df['Currency_Mismatch'].value_counts(normalize=True) * 100


labels = ['Matched', 'Mismatched']
colors = ['#66b3ff', '#ff9999']
plt.figure(figsize=(5, 5))
plt.pie(mismatch_ratio, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title("Currency Match vs Mismatch")
plt.tight_layout()
plt.savefig("eda_outputs/currency_mismatch_pie.png")
plt.show()


plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='Payment_currency', order=df['Payment_currency'].value_counts().index, palette='Set2')
plt.title("Distribution of Payment Currencies")
plt.xlabel("Payment Currency")
plt.ylabel("Transaction Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/payment_currency_countplot.png")
plt.show()


top_currencies = df['Payment_currency'].value_counts().drop("UK pounds").nlargest(10)
plt.figure(figsize=(8, 4))
sns.barplot(x=top_currencies.index, y=top_currencies.values, palette='Set3')
plt.title("Top Non-UK Payment Currencies")
plt.ylabel("Transaction Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/top_non_uk_payment_currencies.png")
plt.show()


plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='Received_currency', order=df['Received_currency'].value_counts().index, palette='Set3')
plt.title("Distribution of Received Currencies")
plt.xlabel("Received Currency")
plt.ylabel("Transaction Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/received_currency_countplot.png")
plt.show()


top_received = df['Received_currency'].value_counts().drop("UK pounds").nlargest(10)

plt.figure(figsize=(8, 4))
sns.barplot(x=top_received.index, y=top_received.values, palette='Set2')
plt.title("Top Non-UK Received Currencies")
plt.ylabel("Transaction Count")
plt.xlabel("Received Currency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/top_non_uk_received_currencies.png")
plt.show()


bins = [0, 1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1_000_000, float('inf')]
labels = ['0–1k', '1k–5k', '5k–10k', '10k–25k', '25k–50k', '50k–100k',
          '100k–250k', '250k–500k', '500k–1M', '1M+']

df['Amount_Range'] = pd.cut(df['Amount'], bins=bins, labels=labels)

range_counts = df['Amount_Range'].value_counts().sort_index()

plt.figure(figsize=(12, 5))
sns.barplot(x=range_counts.index, y=range_counts.values, palette='magma')
plt.title("Transaction Count by Amount Range")
plt.xlabel("Amount Range")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("eda_outputs/amount_range_barplot.png")
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Payment_type', order=df['Payment_type'].value_counts().index, palette='Set2')
plt.title("Distribution of Payment Types")
plt.xlabel("Payment Type")
plt.ylabel("Transaction Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/payment_type_countplot.png")
plt.show()

avg_amounts = df.groupby('Is_laundering')['Amount'].mean()

plt.figure(figsize=(6, 4))
avg_amounts.plot(kind='bar', color=['#66b3ff', '#ff6666'])
plt.xticks([0, 1], ['Not Suspicious', 'Suspicious'], rotation=0)
plt.ylabel("Average Amount")
plt.title("Average Transaction Amount by Laundering Status")
plt.tight_layout()
plt.savefig("eda_outputs/avg_amount_by_laundering_status.png")
plt.show()

location_counts = df['Sender_bank_location'].value_counts().nlargest(10)
location_percent = location_counts / location_counts.sum() * 100


location_df = pd.DataFrame({
    'Country': location_counts.index,
    'Count': location_counts.values,
    'Percentage': location_percent.values
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Count', y='Country', data=location_df, palette='Blues_r')

for index, row in location_df.iterrows():
    plt.text(row['Count'] + 50, index,
             f"{int(row['Count'])} ({row['Percentage']:.2f}%)",
             va='center', fontsize=9)

plt.title("Top 10 Sender Bank Locations – Count and Percentage")
plt.xlabel("Transaction Count")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("eda_outputs/bar_sender_location_count_percentage.png")
plt.show()

receiver_counts = df['Receiver_bank_location'].value_counts().nlargest(10)
receiver_percent = receiver_counts / receiver_counts.sum() * 100

receiver_df = pd.DataFrame({
    'Country': receiver_counts.index,
    'Count': receiver_counts.values,
    'Percentage': receiver_percent.values
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Count', y='Country', data=receiver_df, palette='Greens_r')

for index, row in receiver_df.iterrows():
    plt.text(row['Count'] + 50, index,
             f"{int(row['Count'])} ({row['Percentage']:.2f}%)",
             va='center', fontsize=9)

plt.title("Top 10 Receiver Bank Locations – Count and Percentage")
plt.xlabel("Transaction Count")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("eda_outputs/bar_receiver_location_count_percentage.png")
plt.show()


# In[4]:


## Data Preprocessing ##
df.drop(columns=['Time', 'Date', 'Time_Category', 'Local_Hour'], inplace=True, errors='ignore')

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

drop_cols = ['Sender_account', 'Receiver_account',
             'Payment_currency', 'Payment_type', 'Sender_bank_location',
             'Received_currency', 'Receiver_bank_location',
             'Currency_Pair', 'Laundering_type']
X_train.drop(columns=drop_cols, inplace=True, errors='ignore')
X_test.drop(columns=drop_cols, inplace=True, errors='ignore')

scaler = StandardScaler()
X_train[['Amount']] = scaler.fit_transform(X_train[['Amount']])
X_test[['Amount']] = scaler.transform(X_test[['Amount']])

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

## Data Modeling ## 

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

    print(f"\n Classification Report for {name}:\n")
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
print("\n Overall Model Performance:\n")
print(results_df)


# In[ ]:




