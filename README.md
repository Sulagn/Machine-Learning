<div align="center">

# 🤖 ML Models From Scratch
### A hands-on project for every major machine learning model

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](CONTRIBUTING.md)

<br/>

> **Learn every major ML model by building one small, focused project per model.**  
> No fluff. Just code, output, and key takeaways.

<br/>

[**Supervised**](#-supervised-learning) · [**Unsupervised**](#-unsupervised-learning) · [**Ensemble**](#-ensemble-methods) · [**Advanced**](#-advanced) · [**Setup**](#-setup) · [**Roadmap**](#-learning-roadmap)

</div>

---

## 📖 About This Repo

Each project in this repo is **self-contained** and teaches one model from scratch. Every script:

- 🟢 **Generates its own dataset** — no downloads needed
- 📊 **Produces visualisations** saved as `.png`
- 🧠 **Prints interpretations** of what the model learned
- ✅ **Works in a notebook or as a plain `.py` file**

| Category | Models Covered | Difficulty |
|---|---|---|
| Supervised | Linear Regression, Logistic Regression, Decision Tree, KNN, SVM, Naive Bayes | ⭐ Beginner |
| Unsupervised | K-Means, PCA, DBSCAN | ⭐⭐ Intermediate |
| Ensemble | Random Forest, XGBoost | ⭐⭐ Intermediate |
| Advanced | Neural Network (MLP) | ⭐⭐⭐ Advanced |

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/Sulagn/Machine-Learning.git
cd Machine-Learning
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

<details>
<summary><b>requirements.txt</b> (click to expand)</summary>

```
numpy
pandas
matplotlib
scikit-learn
xgboost
tensorflow        # only for Neural Network project
```

</details>

### 4. Run any project
```bash
python supervised/linear_regression_house_prices.py
```

---

## 🟦 Supervised Learning

> The model learns from **labelled examples** — input → correct output pairs.

---

### 1. 🏠 Linear Regression — House Price Prediction

**File:** `supervised/linear_regression_house_prices.py`

Predict house prices from size, bedrooms, age, and distance to city.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

**What you'll learn:**
- How a model fits a line through data (minimising MSE)
- Feature coefficients and their interpretation
- StandardScaler and why normalisation matters

**Key metrics explained:**
| Metric | What it means |
|---|---|
| MAE | Average prediction error in ₹ |
| RMSE | Error that penalises large mistakes more |
| R² | % of price variation explained (1.0 = perfect) |

**Expected output:**
```
Feature          Coefficient
size_sqft            148.23
bedrooms           19,874.11
age_years          -1,489.34
dist_km            -4,998.12

R²: 0.9412  (94.1% of variance explained)
```

---

### 2. 📧 Logistic Regression — Spam Classifier

**File:** `supervised/logistic_regression_spam.py`

Classify emails as spam or not-spam using word frequency features.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

**What you'll learn:**
- How probabilities are predicted (sigmoid function)
- Precision vs Recall trade-off
- Confusion matrix interpretation

---

### 3. 🚢 Decision Tree — Titanic Survival

**File:** `supervised/decision_tree_titanic.py`

Predict who survived the Titanic based on passenger features.

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=4)
```

**What you'll learn:**
- How trees split on features (Gini impurity)
- Overfitting and `max_depth` as a control
- Visualising a trained decision tree

<details>
<summary><b>Sample tree rules learned</b></summary>

```
|-- Sex = female
|   |-- Pclass <= 2 → SURVIVED ✅
|   |-- Pclass > 2
|       |-- Age <= 8 → SURVIVED ✅
|       |-- Age > 8  → NOT SURVIVED ❌
|-- Sex = male
|   |-- Age <= 6 → SURVIVED ✅
|   |-- Age > 6  → NOT SURVIVED ❌
```

</details>

---

### 4. 🌸 KNN — Iris Flower Classifier

**File:** `supervised/knn_iris.py`

Classify Iris flowers (Setosa, Versicolor, Virginica) by petal/sepal size.

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
```

**What you'll learn:**
- Distance-based classification (no training step!)
- The effect of `k` on decision boundaries
- Why feature scaling is critical for KNN

---

### 5. ✏️ SVM — Handwritten Digit Recognition

**File:** `supervised/svm_digits.py`

Classify hand-drawn digits 0–9 from the sklearn digits dataset.

```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=10, gamma=0.001)
```

**What you'll learn:**
- Margin maximisation and support vectors
- RBF kernel for non-linear boundaries
- Grid search for hyperparameter tuning

---

### 6. 🎬 Naive Bayes — Sentiment Analysis

**File:** `supervised/naive_bayes_sentiment.py`

Predict positive/negative sentiment from movie review text.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
```

**What you'll learn:**
- Bag-of-words and TF-IDF text representation
- Probabilistic classification (Bayes' theorem)
- Top features (most "positive" vs "negative" words)

---

## 🟣 Unsupervised Learning

> The model finds **hidden structure in unlabelled data**.

---

### 7. 🛒 K-Means — Customer Segmentation

**File:** `unsupervised/kmeans_customers.py`

Segment mall customers into groups by annual income and spending score.

```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=5, random_state=42)
```

**What you'll learn:**
- How K-Means iteratively assigns & updates centroids
- The Elbow Method to choose the right `k`
- Business interpretation of customer segments

---

### 8. 👤 PCA — Face Image Compression

**File:** `unsupervised/pca_faces.py`

Compress the Olivetti faces dataset and reconstruct with fewer dimensions.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_compressed = pca.fit_transform(X)
```

**What you'll learn:**
- Dimensionality reduction without losing much info
- Explained variance ratio
- Visualising original vs reconstructed faces

---

### 9. 📡 DBSCAN — Anomaly Detection

**File:** `unsupervised/dbscan_anomaly.py`

Detect outliers in GPS/sensor data — points that don't belong to any cluster.

```python
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.5, min_samples=5)
```

**What you'll learn:**
- Density-based clustering (no need to specify `k`)
- Core points, border points, and noise (anomalies)
- `eps` and `min_samples` tuning

---

## 🟢 Ensemble Methods

> Combine many weak models into one strong predictor.

---

### 10. 🏦 Random Forest — Loan Default Prediction

**File:** `ensemble/random_forest_loan.py`

Predict whether a bank customer will default on a loan.

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
```

**What you'll learn:**
- Bagging: training many trees on random subsets
- Feature importance scores
- Out-of-bag error as a free validation metric

---

### 11. 👔 XGBoost — Employee Attrition

**File:** `ensemble/xgboost_attrition.py`

Predict which employees are likely to leave a company (HR analytics).

```python
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)
```

**What you'll learn:**
- Boosting: trees that correct the errors of previous trees
- `learning_rate` vs `n_estimators` trade-off
- SHAP values for model explainability

---

## 🔴 Advanced

---

### 12. 🔢 Neural Network — MNIST Digit Classification

**File:** `advanced/neural_network_mnist.py`

Build a multi-layer perceptron to classify handwritten digits 0–9.

```python
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])
```

**What you'll learn:**
- Forward pass, backpropagation, gradient descent
- ReLU, Softmax activations
- Dropout as regularisation
- Training curves (loss & accuracy over epochs)

---

## 🗺️ Learning Roadmap

Follow this order for the smoothest learning curve:

```
Week 1 — Supervised Basics
  ├── Linear Regression      (understand loss, coefficients)
  ├── Logistic Regression    (extend to classification)
  └── KNN                    (distance-based intuition)

Week 2 — Supervised Advanced
  ├── Decision Tree          (non-linear boundaries)
  ├── SVM                    (margins and kernels)
  └── Naive Bayes            (probabilistic approach)

Week 3 — Unsupervised
  ├── K-Means                (clustering basics)
  ├── PCA                    (dimensionality reduction)
  └── DBSCAN                 (density & anomalies)

Week 4 — Ensemble + Deep Learning
  ├── Random Forest          (bagging)
  ├── XGBoost                (boosting)
  └── Neural Network         (deep learning intro)
```

---

## 📁 Project Structure

```
ml-models-from-scratch/
│
├── supervised/
│   ├── linear_regression_house_prices.py
│   ├── logistic_regression_spam.py
│   ├── decision_tree_titanic.py
│   ├── knn_iris.py
│   ├── svm_digits.py
│   └── naive_bayes_sentiment.py
│
├── unsupervised/
│   ├── kmeans_customers.py
│   ├── pca_faces.py
│   └── dbscan_anomaly.py
│
├── ensemble/
│   ├── random_forest_loan.py
│   └── xgboost_attrition.py
│
├── advanced/
│   └── neural_network_mnist.py
│
├── requirements.txt
└── README.md
```

---

## 🤝 Contributing

Got an improvement or a new model project to add? PRs are welcome!

1. Fork the repo
2. Create a branch: `git checkout -b feature/new-model`
3. Add your script following the same structure (generate data → train → evaluate → visualise → print takeaways)
4. Open a Pull Request

---

## 📄 License

MIT License — free to use, share, and modify. See [LICENSE](LICENSE) for details.

---

<div align="center">

Made for learners. ⭐ Star this repo if it helped you!

</div>
