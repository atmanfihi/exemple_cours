import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

iris = load_iris()
X = iris.data

model = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = model.fit_predict(X)

print("Inertie :", round(model.inertia_, 2))
print("Silhouette :", round(silhouette_score(X, labels), 3))

inerties = []
for k in range(1, 11):
    inerties.append(KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(range(1, 11), inerties, marker="o")
axes[0].set_xlabel("Nombre de clusters")
axes[0].set_ylabel("Inertie")
axes[0].set_title("Méthode du coude")
axes[0].grid(True)

axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
axes[1].scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="red", marker="X", s=150)
axes[1].set_xlabel("Longueur des sépales")
axes[1].set_ylabel("Largeur des sépales")
axes[1].set_title("Clustering K-Means")
plt.show()
