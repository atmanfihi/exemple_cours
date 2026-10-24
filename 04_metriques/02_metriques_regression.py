import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rng = np.random.default_rng(42)
X = np.linspace(1, 50, 60).reshape(-1, 1)
y = 5 * X.ravel() + 25 + rng.normal(0, 12, X.shape[0])

X_train, X_test, y_train, y_test = X[:45], X[45:], y[:45], y[45:]

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE  : {:.2f}".format(mse))
print("RMSE : {:.2f}".format(rmse))
print("MAE  : {:.2f}".format(mae))
print("R²   : {:.3f}".format(r2))

plt.scatter(y_test, y_pred, color="blue", label="Prédictions")
limites = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(limites, limites, color="red", linestyle="--", label="Prédiction parfaite")
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Qualité du modèle de régression (RMSE = {:.2f})".format(rmse))
plt.legend()
plt.grid(True)
plt.show()
