import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("students.csv")   # change file name if needed

print(df.head())

print(df.describe())


# =========================
# 2. SPLIT FEATURES & TARGET
# =========================
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]


# =========================
# 3. CATEGORICAL COLUMNS
# =========================
categorical_cols = X.select_dtypes(exclude=["number"]).columns


# =========================
# 4. PREPROCESSING
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)


# =========================
# 5. MODEL 
# =========================
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)


# =========================
# 6. PIPELINE
# =========================
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])


# =========================
# 7. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 8. TRAIN MODEL
# =========================
clf.fit(X_train, y_train)


# =========================
# 9. PREDICTION
# =========================
pred = clf.predict(X_test)


# =========================
# 10. EVALUATION
# =========================
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("\n===== MODEL PERFORMANCE =====")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# =========================
# 11. ACTUAL VS PREDICTED GRAPH
# =========================
plt.scatter(y_test, pred)
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Predicted (Random Forest)")
plt.show()


