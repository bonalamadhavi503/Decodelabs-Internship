# =========================
# FRAUD DETECTION PIPELINE
# =========================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, recall_score, roc_auc_score

from imblearn.over_sampling import RandomOverSampler

from sklearn.datasets import make_classification


X, y = make_classification(
    n_samples=5000,
    n_features=20,          # like real fraud datasets
    n_informative=10,
    n_redundant=5,
    weights=[0.98, 0.02],   # highly imbalanced (fraud cases)
    random_state=42
)

df = pd.DataFrame(X)
df["Class"] = y


X = df.drop("Class", axis=1)
y = df["Class"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


ros = RandomOverSampler(random_state=42)

X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

print("Before Oversampling:", np.bincount(y_train))
print("After Oversampling:", np.bincount(y_train_res))




scaler = StandardScaler()

X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)




models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}



for name, model in models.items():

    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n")
    print(name)
    print("")

    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))