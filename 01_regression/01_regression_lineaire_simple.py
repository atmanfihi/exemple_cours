import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict([[6]])
print("Prédiction pour X=6 :", y_pred[0])
print("Équation : y = {:.2f}x + {:.2f}".format(model.coef_[0], model.intercept_))

X_line = [[i] for i in range(1, 7)]
plt.scatter(X, y, color="blue", label="Données réelles")
plt.plot([i[0] for i in X_line], model.predict(X_line), color="green", label="Droite de régression")
plt.scatter(6, y_pred, color="red", label=f"Prédiction X=6 ({y_pred[0]:.2f})")
plt.title("Régression linéaire avec scikit-learn")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
