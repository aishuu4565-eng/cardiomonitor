"""
CardioGuard - Stroke Prediction Model Training
Deep Neural Network (DNN) using TensorFlow/Keras
Run this file ONCE to generate all model files needed by app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 50)
print("  CardioGuard - DNN Model Training")
print("=" * 50)

CSV_PATH = "healthcare-dataset-stroke-data.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found: '{CSV_PATH}'\n"
        "Make sure the CSV file is in the SAME folder as train_model.py"
    )

df = pd.read_csv(CSV_PATH)
print(f"\n[1] Dataset loaded  →  {df.shape[0]} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. CLEAN DATA
# ─────────────────────────────────────────────
# Drop unused column
df.drop(columns=["id"], inplace=True, errors="ignore")

# BMI: replace "N/A" string & actual NaN with mean
df["bmi"] = df["bmi"].replace("N/A", np.nan)
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
bmi_mean = df["bmi"].mean()
df["bmi"].fillna(bmi_mean, inplace=True)

# Remove 'Other' gender (very few rows, causes encoder issues)
df = df[df["gender"] != "Other"].copy()

print(f"[2] Data cleaned    →  {df.shape[0]} rows remaining")
print(f"    BMI mean (for missing values) = {bmi_mean:.2f}")
print(f"    Stroke cases: {df['stroke'].sum()} / {len(df)}")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING  (must match app.py)
# ─────────────────────────────────────────────
def add_features(data: pd.DataFrame) -> pd.DataFrame:
    df2 = data.copy()

    df2["age_group"] = pd.cut(
        df2["age"],
        bins=[0, 30, 50, 70, 120],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    df2["high_glucose"] = (df2["avg_glucose_level"] >= 140).astype(int)
    df2["high_bmi"]     = (df2["bmi"] >= 30).astype(int)

    df2["risk_score"] = (
        df2["hypertension"].astype(int)
        + df2["heart_disease"].astype(int)
        + (df2["avg_glucose_level"] >= 140).astype(int)
        + (df2["bmi"] >= 30).astype(int)
        + (df2["age"] >= 50).astype(int)
    )

    df2["lifestyle_risk"] = (
        (df2["avg_glucose_level"] >= 140).astype(int)
        + (df2["bmi"] >= 30).astype(int)
        + (df2["smoking_status"] == "smokes").astype(int)
    )

    return df2

df = add_features(df)
print("[3] Feature engineering done")

# ─────────────────────────────────────────────
# 4. ENCODE CATEGORICAL COLUMNS
# ─────────────────────────────────────────────
CAT_COLS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

print("[4] Categorical columns encoded")

# ─────────────────────────────────────────────
# 5. DEFINE FEATURE COLUMNS  (must match app.py)
# ─────────────────────────────────────────────
TARGET = "stroke"

feature_columns = [
    "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status",
    "age_group", "high_glucose", "high_bmi",
    "risk_score", "lifestyle_risk"
]

X = df[feature_columns].astype(float)
y = df[TARGET].astype(int)

print(f"[5] Feature matrix shape: {X.shape}")

# ─────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 7. SCALE FEATURES
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("[6] Data split & scaled")

# ─────────────────────────────────────────────
# 8. CLASS WEIGHTS  (handle imbalanced dataset)
# ─────────────────────────────────────────────
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
print(f"[7] Class weights  →  {class_weight_dict}")

# ─────────────────────────────────────────────
# 9. BUILD DNN MODEL
# ─────────────────────────────────────────────
def build_dnn(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        # Input
        layers.Input(shape=(input_dim,)),

        # Hidden Layer 1
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Hidden Layer 2
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Hidden Layer 3
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),

        # Output (binary classification)
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model

model = build_dnn(X_train_sc.shape[1])
model.summary()

# ─────────────────────────────────────────────
# 10. CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=10,
        restore_best_weights=True,
        mode="max",
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1
    )
]

# ─────────────────────────────────────────────
# 11. TRAIN
# ─────────────────────────────────────────────
print("\n[8] Training DNN...")
history = model.fit(
    X_train_sc, y_train,
    validation_data=(X_test_sc, y_test),
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# 12. EVALUATE
# ─────────────────────────────────────────────
print("\n[9] Evaluation on Test Set:")
loss, acc, auc = model.evaluate(X_test_sc, y_test, verbose=0)
print(f"    Loss     : {loss:.4f}")
print(f"    Accuracy : {acc:.4f}")
print(f"    AUC      : {auc:.4f}")

y_pred_prob = model.predict(X_test_sc, verbose=0).flatten()
y_pred = (y_pred_prob >= 0.3).astype(int)   # lower threshold for rare class

print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Stroke", "Stroke"]))

print("    Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────────
# 13. SAVE ALL MODEL FILES
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

# Save Keras model
model.save("stroke_prediction_model.keras")

# Save sklearn objects
joblib.dump(scaler,          "models/scaler.pkl")
joblib.dump(encoders,        "models/encoders.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")
joblib.dump(bmi_mean,        "models/bmi_mean.pkl")

print("\n[10] All files saved:")
print("     ✅  stroke_prediction_model.keras")
print("     ✅  models/scaler.pkl")
print("     ✅  models/encoders.pkl")
print("     ✅  models/feature_columns.pkl")
print("     ✅  models/bmi_mean.pkl")
print("\n  Training complete! Now run:  python app.py")
print("=" * 50)