
from pathlib import Path
import json
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ==================================================
# 1. KONFIGURASI
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "xgboost_heart_disease.pkl"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "API prediksi indikasi penyakit jantung "
        "menggunakan model XGBoost."
    ),
    version="1.0.0",
)


# ==================================================
# 2. MEMUAT MODEL DAN METADATA
# ==================================================

def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"File metadata tidak ditemukan: {METADATA_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata.get("feature_columns")

    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(
            "metadata.json tidak memiliki feature_columns "
            "yang valid."
        )

    return model, metadata, feature_columns


model, metadata, FEATURE_COLUMNS = load_artifacts()

logger.info("Model berhasil dimuat.")
logger.info("Jumlah fitur: %s", len(FEATURE_COLUMNS))


# ==================================================
# 3. SKEMA INPUT API
# ==================================================

class PredictionRequest(BaseModel):
    data: dict[str, object] = Field(
        ...,
        description="Data karakteristik pasien sesuai fitur model.",
        examples=[
            {
                "age": 55,
                "sex": "Male",
                "cp": "typical angina",
                "trestbps": 140,
                "chol": 250,
                "fbs": "FALSE",
                "restecg": "normal",
                "thalch": 150,
                "exang": "FALSE",
                "oldpeak": 1.0,
                "slope": "flat",
                "ca": 0.0,
                "thal": "normal",
            }
        ],
    )


# ==================================================
# 4. ENDPOINT UTAMA
# ==================================================

@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API aktif.",
        "model": "XGBoost",
        "docs": "/docs",
        "health": "/health",
    }


# ==================================================
# 5. CEK STATUS API
# ==================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "metadata_loaded": metadata is not None,
        "total_features": len(FEATURE_COLUMNS),
    }


# ==================================================
# 6. MELIHAT DAFTAR FITUR
# ==================================================

@app.get("/features")
def get_features():
    return {
        "total_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
    }


# ==================================================
# 7. MELAKUKAN PREDIKSI
# ==================================================

@app.post("/predict")
def predict(request: PredictionRequest):
    input_data = request.data

    # Memastikan seluruh fitur tersedia
    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in input_data
    ]

    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Terdapat fitur yang belum diisi.",
                "missing_features": missing_features,
            },
        )

    try:
        # Menyusun data sesuai urutan fitur saat training.
        # Pipeline model akan melakukan preprocessing otomatis.
        input_df = pd.DataFrame(
            [{feature: input_data[feature]
              for feature in FEATURE_COLUMNS}],
            columns=FEATURE_COLUMNS,
        )

        prediction = int(model.predict(input_df)[0])

        # Probabilitas kelas positif (kelas 1)
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        positive_index = classes.index(1)
        positive_probability = float(
            probabilities[positive_index]
        )

        label = (
            "Terindikasi"
            if prediction == 1
            else "Tidak terindikasi"
        )

        return {
            "success": True,
            "prediction": prediction,
            "label": label,
            "probability": round(positive_probability, 4),
            "probability_percent": round(
                positive_probability * 100, 2
            ),
            "message": (
                "Hasil prediksi model machine learning, "
                "bukan diagnosis medis."
            ),
        }

    except HTTPException:
        raise

    except Exception as error:
        logger.exception("Prediksi gagal.")

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Terjadi kesalahan saat melakukan prediksi.",
                "error": str(error),
            },
        )
