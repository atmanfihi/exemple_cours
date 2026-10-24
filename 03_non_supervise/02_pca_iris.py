import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Variance expliquée par composante :", pca.explained_variance_ratio_.round(3))
print("Variance cumulée :", round(pca.explained_variance_ratio_.sum(), 3))

for i, name in enumerate(iris.target_names):
    plt.scatter(X_pca[iris.target == i, 0], X_pca[iris.target == i, 1], label=name)
plt.xlabel("Composante principale 1 ({:.1%})".format(pca.explained_variance_ratio_[0]))
plt.ylabel("Composante principale 2 ({:.1%})".format(pca.explained_variance_ratio_[1]))
plt.title("ACP du dataset Iris")
plt.legend()
plt.grid(True)
plt.show()
