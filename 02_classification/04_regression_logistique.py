import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "heartdiseases.csv"))
data = pd.get_dummies(data, columns=["Gender", "Education", "Smoker"], drop_first=True, dtype=int)

X = data.drop("Heart_Disease", axis=1)
y = data["Heart_Disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_reg = LogisticRegression(solver="liblinear", max_iter=100, random_state=42)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]

print("Accuracy :", round(accuracy_score(y_test, y_pred), 3))
print("AUC :", round(roc_auc_score(y_test == "Yes", y_proba), 3))
print("Classes :", log_reg.classes_)
print("Intercept :", log_reg.intercept_.round(3))
print("Nombre d'itérations :", log_reg.n_iter_)
print("\nCoefficients :")
for name, coef in sorted(zip(X.columns, log_reg.coef_[0]), key=lambda c: abs(c[1]), reverse=True):
    print(f"  {name:<22} {coef:+.3f}")
print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test == "Yes", y_proba)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test == 'Yes', y_proba):.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC - Régression logistique")
plt.legend()
plt.grid(True)
plt.show()
