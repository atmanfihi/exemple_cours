import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = [[1], [2], [3], [4], [5]]
y = [30, 35, 40, 45, 50]

model = LinearRegression()
model.fit(X, y)

a = model.coef_[0]
b = model.intercept_
r2 = r2_score(y, model.predict(X))

print("Équation : y = {:.2f}x + {:.2f}".format(a, b))
print("Score R² :", round(r2, 3))
print("Salaire prédit pour 6 ans d'expérience : {:.1f} kDH".format(model.predict([[6]])[0]))

plt.scatter(X, y, color="blue", label="Données")
plt.plot(X, model.predict(X), color="red", label="Droite de régression")
plt.title("Régression linéaire simple")
plt.xlabel("Expérience (années)")
plt.ylabel("Salaire (kDH)")
plt.legend()
plt.grid(True)
plt.show()
