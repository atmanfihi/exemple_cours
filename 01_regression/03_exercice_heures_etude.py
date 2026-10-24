import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([3, 5, 7, 9, 11, 11, 14, 16, 18, 19])

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print("β0 (interception) : {:.4f}".format(model.intercept_))
print("β1 (pente) : {:.4f}".format(model.coef_[0]))
print("Note prédite pour 7,5 heures : {:.2f}".format(model.predict([[7.5]])[0]))
print("Note prédite pour 6 heures : {:.2f}".format(model.predict([[6]])[0]))
print("R² : {:.4f}".format(model.score(X, y)))

plt.scatter(X, y, color="blue", label="Données observées")
plt.plot(X, y_pred, color="red", label="Droite de régression")
plt.scatter([7.5, 6], model.predict([[7.5], [6]]), color="green", zorder=5, label="Prédictions")
plt.xlabel("Heures d'étude (X)")
plt.ylabel("Note obtenue (Y)")
plt.title("Régression linéaire simple : Heures d'étude vs Note")
plt.legend()
plt.grid(True)
plt.show()
