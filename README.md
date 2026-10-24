# Technologies d'Intelligence Artificielle — Machine Learning

Exemples pratiques du cours **« Technologies d'IA — Apprentissage Automatique »** (ENSAM-Fès) avec Python et scikit-learn.

## Installation

```bash
pip install -r requirements.txt
```

## Contenu

### 1. Régression linéaire (`01_regression/`)

| Script | Contenu | Résultat |
|---|---|---|
| `01_regression_lineaire_simple.py` | X = [1..5], y = [2,4,6,8,10], prédiction pour X = 6 | `y = 2.00x + 0.00` → **12.0** |
| `02_salaire_experience.py` | Salaire (kDH) selon les années d'expérience | `y = 5.00x + 25.00`, **R² = 1.0** |
| `03_exercice_heures_etude.py` | Exercice : heures d'étude vs note | `β0 = 1.4667`, `β1 = 1.7879`, `R² = 0.991` |

Interprétation de l'exercice :

- **β1 = 1.7879** : chaque heure d'étude supplémentaire augmente la note de ~1,79 point.
- **β0 = 1.4667** : note de base estimée pour 0 heure d'étude.
- Note prédite : **14,88** pour 7,5 h et **12,19** pour 6 h.

### 2. Classification (`02_classification/`)

| Script | Modèle | Résultat |
|---|---|---|
| `01_iris_knn.py` | KNN (k = 3) + StandardScaler sur Iris | Accuracy **1.00** |
| `02_iris_arbre_decision.py` | Arbre de décision (profondeur 3) + visualisation | Accuracy **1.00** |
| `03_iris_random_forest.py` | Random Forest (100 arbres) + matrice de confusion | Accuracy **1.00** |
| `04_regression_logistique.py` | Régression logistique sur `data/heartdiseases.csv` + courbe ROC | Accuracy **0.75**, AUC **0.77** |

Le jeu de données `data/heartdiseases.csv` est **synthétique** (800 patients) et reproduit la structure du cours
(`Age`, `Gender`, `Education`, `Smoker`, `Income`, `Cholesterol`, `BMI`, `BloodPressure`, `Heart_Disease`).
Il est encodé avec `get_dummies`, standardisé, puis modélisé avec `LogisticRegression(solver="liblinear")`
pour éviter l'avertissement `ConvergenceWarning` vu en cours.

### 3. Apprentissage non supervisé (`03_non_supervise/`)

| Script | Contenu |
|---|---|
| `01_kmeans_iris.py` | K-Means (k = 3), méthode du coude, score de silhouette |
| `02_pca_iris.py` | ACP : 2 composantes, ~95,8 % de variance expliquée |

### 4. Métriques de performance (`04_metriques/`)

| Script | Métriques |
|---|---|
| `01_metriques_classification.py` | Accuracy, Precision, Recall, F1-score, AUC, matrice de confusion |
| `02_metriques_regression.py` | MSE, RMSE, MAE, R², graphique réel vs prédit |

## Exécution

Chaque script est autonome et affiche ses résultats dans la console (et une figure pour les scripts graphiques) :

```bash
python 01_regression/01_regression_lineaire_simple.py
```

Sans affichage graphique (serveur, CI) :

```bash
MPLBACKEND=Agg python 02_classification/01_iris_knn.py
```

## Structure

```
.
├── 01_regression
├── 02_classification
├── 03_non_supervise
├── 04_metriques
├── data
│   └── heartdiseases.csv
├── requirements.txt
└── README.md
```
