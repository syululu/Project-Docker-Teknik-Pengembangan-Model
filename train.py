
# ============================================
# TRAIN.PY
# Prediksi Penyakit Jantung Menggunakan XGBoost
# ============================================

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


# ============================================
# 1. KONFIGURASI
# ============================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "heart_disease_uci.csv"
OUTPUT_DIR = BASE_DIR / "model"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================
# 2. LOAD DATASET
# ============================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH}\n"
        "Periksa nama dan lokasi file CSV."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("PEMERIKSAAN AWAL DATASET")
print("=" * 60)

print(f"Jumlah baris awal : {df.shape[0]}")
print(f"Jumlah kolom awal : {df.shape[1]}")

print("\nNama kolom:")
print(df.columns.tolist())

print("\nTipe data:")
print(df.dtypes)

print("\nMissing value:")
print(df.isnull().sum())

print(f"\nDuplikasi baris: {df.duplicated().sum()}")


# ============================================
# 3. DATA CLEANING
# ============================================

# Rapikan nama kolom
df.columns = df.columns.str.strip().str.lower()

# Hapus baris duplikat
df = df.drop_duplicates().copy()

# Pastikan target tersedia
if "num" not in df.columns:
    raise ValueError("Kolom target 'num' tidak ditemukan.")

# Target tidak boleh kosong
df = df.dropna(subset=["num"]).copy()

# Pastikan target valid
valid_labels = {0, 1, 2, 3, 4}

if not set(df["num"].unique()).issubset(valid_labels):
    raise ValueError(
        "Target num harus berisi nilai 0, 1, 2, 3, atau 4."
    )

# Pastikan target numerik
df["num"] = pd.to_numeric(df["num"], errors="raise")

# Hapus kolom identitas
if "id" in df.columns:
    df = df.drop(columns=["id"])

print("\n" + "=" * 60)
print("HASIL PEMBERSIHAN DATA")
print("=" * 60)

print(f"Jumlah baris: {len(df)}")
print(f"Jumlah kolom: {len(df.columns)}")

print("\nMissing value setelah cleaning:")
print(df.isnull().sum())


# ============================================
# 4. FEATURE ENGINEERING
# ============================================

# Klasifikasi biner:
# num = 0     -> kelas 0
# num = 1..4  -> kelas 1

df["target"] = (df["num"] > 0).astype(int)

# Pisahkan fitur dan target
# Kolom num asli tidak boleh menjadi fitur
X = df.drop(columns=["num", "target"])
y = df["target"]

# Identifikasi fitur berdasarkan tipe data
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nFitur numerik:")
print(numeric_features)

print("\nFitur kategorikal:")
print(categorical_features)

print("\nDistribusi target:")
print(
    y.value_counts()
    .sort_index()
    .rename(index={
        0: "Tidak terindikasi",
        1: "Terindikasi"
    })
)


# ============================================
# 5. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n" + "=" * 60)
print("PEMBAGIAN DATA")
print("=" * 60)

print(f"Jumlah data train: {len(X_train)}")
print(f"Jumlah data test : {len(X_test)}")

print("\nProporsi kelas data train:")
print(y_train.value_counts(normalize=True).round(3))

print("\nProporsi kelas data test:")
print(y_test.value_counts(normalize=True).round(3))


# ============================================
# 6. PREPROCESSING
# ============================================

# Fitur numerik:
# Missing value diganti dengan median.
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

# Fitur kategorikal:
# Missing value diganti dengan modus.
# Kategori diubah menjadi representasi One-Hot.
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

# Gabungkan preprocessing numerik dan kategorikal
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)


# ============================================
# 7. MEMBANGUN MODEL XGBOOST
# ============================================

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Satukan preprocessing dan model.
# Seluruh pipeline nantinya disimpan dalam satu file PKL.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb_model)
    ]
)


# ============================================
# 8. TRAINING MODEL
# ============================================

print("\n" + "=" * 60)
print("TRAINING MODEL XGBOOST")
print("=" * 60)

model.fit(X_train, y_train)

print("Training selesai.")


# ============================================
# 9. PREDIKSI DAN EVALUASI
# ============================================

y_pred = model.predict(X_test).astype(int)

# Probabilitas kelas positif
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(
        accuracy_score(y_test, y_pred)
    ),
    "precision": float(
        precision_score(
            y_test, y_pred, zero_division=0
        )
    ),
    "recall": float(
        recall_score(
            y_test, y_pred, zero_division=0
        )
    ),
    "f1_score": float(
        f1_score(
            y_test, y_pred, zero_division=0
        )
    ),
    "roc_auc": float(
        roc_auc_score(y_test, y_proba)
    )
}

print("\n" + "=" * 60)
print("HASIL EVALUASI XGBOOST")
print("=" * 60)

for name, value in metrics.items():
    print(f"{name.upper():<12}: {value:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Tidak terindikasi",
            "Terindikasi"
        ],
        zero_division=0
    )
)


# ============================================
# 10. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Tidak terindikasi",
        "Terindikasi"
    ]
)

disp.plot(values_format="d")
plt.title("Confusion Matrix - XGBoost")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================
# 11. FEATURE IMPORTANCE
# ============================================

# Ambil nama fitur setelah One-Hot Encoding
feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

# Ambil importance dari model XGBoost
importance_values = (
    model.named_steps["classifier"]
    .feature_importances_
)

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
}).sort_values(
    by="importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(importance.to_string(index=False))

importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

plt.figure(figsize=(10, 7))

top_features = importance.head(15).sort_values(
    by="importance",
    ascending=True
)

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Fitur")
plt.title("Top 15 Feature Importance - XGBoost")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================
# 12. SIMPAN MODEL DALAM FORMAT PKL
# ============================================

model_path = OUTPUT_DIR / "xgboost_heart_disease.pkl"

joblib.dump(model, model_path)

print("\nModel berhasil disimpan:")
print(model_path)


# ============================================
# 13. SIMPAN METADATA
# ============================================

metadata = {
    "algorithm": "XGBoost",
    "model_type": "binary_classification",
    "target_column": "target",
    "original_target_column": "num",
    "target_mapping": {
        "0": "Tidak terindikasi (num = 0)",
        "1": "Terindikasi (num > 0)"
    },
    "feature_columns": X.columns.tolist(),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "model_file": model_path.name
}

with open(
    OUTPUT_DIR / "metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=4,
        ensure_ascii=False
    )

with open(
    OUTPUT_DIR / "metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================
# 14. RINGKASAN AKHIR
# ============================================

print("\n" + "=" * 60)
print("PROSES TRAINING SELESAI")
print("=" * 60)

print(f"Model       : {model_path}")
print(f"Metadata    : {OUTPUT_DIR / 'metadata.json'}")
print(f"Metrik      : {OUTPUT_DIR / 'metrics.json'}")
print(f"Confusion   : {OUTPUT_DIR / 'confusion_matrix.png'}")
print(f"Importance  : {OUTPUT_DIR / 'feature_importance.csv'}")
print(f"Grafik      : {OUTPUT_DIR / 'feature_importance.png'}")
