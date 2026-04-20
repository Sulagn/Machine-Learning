# ============================================================
#  Project: House Price Prediction — Linear Regression
#  Goal   : Learn how Linear Regression works by predicting
#           house prices from features like size, rooms, etc.
#  Dataset: Synthetically generated (no download needed)
# ============================================================

# ---------- 0. Install dependencies (run once) --------------
# pip install numpy pandas matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# STEP 1 — Generate a Synthetic Dataset
# ============================================================
# We create a fake but realistic dataset so you need zero downloads.
# Each house has: size (sqft), bedrooms, age, distance to city centre.
# Price is derived from these features + some random noise.

np.random.seed(42)
n_samples = 300

size_sqft    = np.random.randint(500, 4000, n_samples)          # 500–4000 sqft
bedrooms     = np.random.randint(1, 6, n_samples)               # 1–5 bedrooms
age_years    = np.random.randint(0, 50, n_samples)              # 0–50 years old
dist_km      = np.random.uniform(1, 30, n_samples)              # 1–30 km from city

# Price formula (what we want the model to learn)
price = (
    150 * size_sqft          # +₹150 per sqft
    + 20000 * bedrooms       # +₹20k per bedroom
    - 1500 * age_years       # older house = lower price
    - 5000 * dist_km         # farther = cheaper
    + 50000                  # base price
    + np.random.normal(0, 30000, n_samples)  # realistic noise
)

# Build a DataFrame
df = pd.DataFrame({
    "size_sqft": size_sqft,
    "bedrooms":  bedrooms,
    "age_years": age_years,
    "dist_km":   dist_km,
    "price":     price.astype(int)
})

print("=" * 55)
print("STEP 1 — Dataset Preview")
print("=" * 55)
print(df.head(8).to_string(index=False))
print(f"\nShape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Price range: ₹{df['price'].min():,} – ₹{df['price'].max():,}")


# ============================================================
# STEP 2 — Explore the Data
# ============================================================
print("\n" + "=" * 55)
print("STEP 2 — Basic Statistics")
print("=" * 55)
print(df.describe().round(2).to_string())

# Correlation with price
print("\nCorrelation with price:")
print(df.corr()["price"].drop("price").round(3).to_string())


# ============================================================
# STEP 3 — Prepare Features & Target
# ============================================================
X = df[["size_sqft", "bedrooms", "age_years", "dist_km"]]  # features
y = df["price"]                                             # target

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features — important for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)       # use same scaler, don't re-fit

print("\n" + "=" * 55)
print("STEP 3 — Train/Test Split")
print("=" * 55)
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")


# ============================================================
# STEP 4 — Train the Model
# ============================================================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\n" + "=" * 55)
print("STEP 4 — Trained Model Coefficients")
print("=" * 55)
print(f"{'Feature':<15} {'Coefficient':>15}")
print("-" * 32)
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:<15} {coef:>15,.2f}")
print(f"\nIntercept : ₹{model.intercept_:,.2f}")
print("""
Interpretation:
  A positive coefficient means the feature increases price.
  A negative coefficient means it decreases price.
  Larger absolute value = stronger influence.
""")


# ============================================================
# STEP 5 — Evaluate the Model
# ============================================================
y_pred = model.predict(X_test_scaled)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("=" * 55)
print("STEP 5 — Model Evaluation (on test set)")
print("=" * 55)
print(f"MAE  (Mean Abs Error)  : ₹{mae:>12,.2f}")
print(f"RMSE (Root Mean Sq Err): ₹{rmse:>12,.2f}")
print(f"R²   (Variance explained): {r2:>10.4f}  ({r2*100:.1f}%)")
print("""
Metrics explained:
  MAE   → on average, predictions are off by this amount
  RMSE  → penalises large errors more than MAE
  R²    → 1.0 = perfect fit, 0 = model knows nothing
          > 0.85 is generally considered good
""")


# ============================================================
# STEP 6 — Make a Prediction on a New House
# ============================================================
new_house = pd.DataFrame([{
    "size_sqft": 1800,
    "bedrooms":  3,
    "age_years": 10,
    "dist_km":   8
}])

new_house_scaled   = scaler.transform(new_house)
predicted_price    = model.predict(new_house_scaled)[0]

print("=" * 55)
print("STEP 6 — Predict a New House")
print("=" * 55)
print("House details:")
print(f"  Size     : 1800 sqft")
print(f"  Bedrooms : 3")
print(f"  Age      : 10 years")
print(f"  Distance : 8 km from city")
print(f"\n  Predicted Price: ₹{predicted_price:,.0f}")


# ============================================================
# STEP 7 — Visualisations
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Linear Regression — House Price Prediction", fontsize=16, y=1.01)

# --- Plot 1: Actual vs Predicted ---
ax = axes[0, 0]
ax.scatter(y_test, y_pred, alpha=0.6, color="#378ADD", edgecolors="white", linewidth=0.5)
min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")
ax.set_xlabel("Actual Price (₹)")
ax.set_ylabel("Predicted Price (₹)")
ax.set_title("Actual vs Predicted Prices")
ax.legend()

# --- Plot 2: Residuals ---
residuals = y_test.values - y_pred
ax = axes[0, 1]
ax.scatter(y_pred, residuals, alpha=0.6, color="#7F77DD", edgecolors="white", linewidth=0.5)
ax.axhline(0, color="red", linestyle="--", linewidth=2)
ax.set_xlabel("Predicted Price (₹)")
ax.set_ylabel("Residual (Actual − Predicted)")
ax.set_title("Residual Plot\n(points should scatter randomly around 0)")

# --- Plot 3: Feature Coefficients ---
ax = axes[1, 0]
colors = ["#378ADD" if c > 0 else "#E24B4A" for c in model.coef_]
bars = ax.barh(X.columns, model.coef_, color=colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient Value")
ax.set_title("Feature Coefficients\n(blue = positive impact, red = negative)")
for bar, val in zip(bars, model.coef_):
    ax.text(val + (max(model.coef_) * 0.02), bar.get_y() + bar.get_height()/2,
            f"{val:,.0f}", va="center", fontsize=9)

# --- Plot 4: Price vs Size (key relationship) ---
ax = axes[1, 1]
ax.scatter(df["size_sqft"], df["price"], alpha=0.4, color="#639922", s=20, label="Data points")
# Draw regression line for size alone
size_range = np.linspace(df["size_sqft"].min(), df["size_sqft"].max(), 100)
# Use mean values for other features
mean_beds = df["bedrooms"].mean()
mean_age  = df["age_years"].mean()
mean_dist = df["dist_km"].mean()
line_data = pd.DataFrame({
    "size_sqft": size_range,
    "bedrooms":  mean_beds,
    "age_years": mean_age,
    "dist_km":   mean_dist
})
line_scaled = scaler.transform(line_data)
line_pred   = model.predict(line_scaled)
ax.plot(size_range, line_pred, color="#E24B4A", linewidth=2, label="Model trend")
ax.set_xlabel("Size (sqft)")
ax.set_ylabel("Price (₹)")
ax.set_title("Price vs Size\n(other features held at mean)")
ax.legend()

plt.tight_layout()
plt.savefig("house_price_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as 'house_price_results.png'")


# ============================================================
# STEP 8 — Key Takeaways
# ============================================================
print("\n" + "=" * 55)
print("KEY TAKEAWAYS")
print("=" * 55)
print("""
1. LINEAR REGRESSION fits a straight line:
   Price = w1*size + w2*beds + w3*age + w4*dist + bias

2. THE MODEL LEARNED (from data alone) that:
   - Bigger house → higher price  (+ve coefficient on size)
   - More bedrooms → higher price (+ve coefficient on bedrooms)
   - Older house → lower price    (-ve coefficient on age)
   - Farther away → lower price   (-ve coefficient on dist)

3. STANDARDISATION (StandardScaler) ensures all features
   are on the same scale, preventing large-valued features
   (like sqft) from dominating.

4. R² > 0.9 means our model explains 90%+ of the price
   variation — excellent for a first model!

5. LIMITATIONS: Linear Regression assumes a LINEAR
   relationship. Real house prices often need more complex
   models (polynomial, tree-based) to capture interactions.
""")