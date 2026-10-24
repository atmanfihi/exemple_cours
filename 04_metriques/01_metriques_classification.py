import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy  : {:.3f}".format(accuracy_score(y_test, y_pred)))
print("Precision : {:.3f}".format(precision_score(y_test, y_pred, average="macro")))
print("Recall    : {:.3f}".format(recall_score(y_test, y_pred, average="macro")))
print("F1-score  : {:.3f}".format(f1_score(y_test, y_pred, average="macro")))
print("AUC (one-vs-rest) : {:.3f}".format(roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=iris.target_names, cmap="Blues")
plt.title("Métriques de classification - Iris")
plt.show()
