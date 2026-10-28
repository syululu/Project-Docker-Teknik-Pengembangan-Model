
from pathlib import Path

import json
import requests
import pandas as pd
import streamlit as st


# ==================================================
# 1. KONFIGURASI APLIKASI
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

METADATA_PATH = BASE_DIR / "model" / "metadata.json"
DATASET_PATH = BASE_DIR / "heart_disease_uci.csv"

# Alamat API FastAPI
API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    layout="wide",
)

st.title("Prediksi Penyakit Jantung")
st.write(
    "Aplikasi prediksi indikasi penyakit jantung "
    "menggunakan model machine learning XGBoost."
)

st.info(
    "Masukkan karakteristik pasien pada formulir berikut. "
    "Data akan dikirim ke API untuk diprediksi."
)


# ==================================================
# 2. MEMUAT METADATA DAN DATA KATEGORIKAL
# ==================================================

@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


try:
    metadata = load_metadata()
    feature_columns = metadata["feature_columns"]
    numeric_features = metadata.get("numeric_features", [])
    categorical_features = metadata.get("categorical_features", [])

    dataset = load_dataset()

except FileNotFoundError as error:
    st.error(
        f"File tidak ditemukan: {error.filename}. "
        "Periksa lokasi metadata.json dan dataset CSV."
    )
    st.stop()

except Exception as error:
    st.error(f"Gagal memuat metadata atau dataset: {error}")
    st.stop()


# ==================================================
# 3. STATUS KONEKSI API
# ==================================================

with st.sidebar:
    st.header("Status Sistem")
    st.write("API:", API_URL)

    if st.button("Periksa Koneksi API", use_container_width=True):
        try:
            response = requests.get(
                f"{API_URL}/health",
                timeout=5,
            )
            response.raise_for_status()
            health = response.json()

            if (
                health.get("model_loaded")
                and health.get("metadata_loaded")
            ):
                st.success("API dan model siap digunakan.")
            else:
                st.warning("API aktif, tetapi model atau metadata belum siap.")

            st.json(health)

        except requests.RequestException as error:
            st.error(f"API tidak dapat dihubungi: {error}")


# ==================================================
# 4. FORMULIR INPUT PASIEN
# ==================================================

st.subheader("1. Masukkan Data Pasien")

st.caption(
    "Nama fitur mengikuti dataset yang digunakan ketika model dilatih. "
    "Pilihan kategori diambil dari dataset asli."
)

patient_data = {}

with st.form("prediction_form"):

    for feature in feature_columns:
        label = feature.replace("_", " ").title()

        if feature in numeric_features:
            # Ambil median sebagai nilai awal jika tersedia.
            column_values = pd.to_numeric(
                dataset[feature],
                errors="coerce",
            ).dropna() if feature in dataset.columns else pd.Series(
                dtype=float
            )

            default_value = (
                float(column_values.median())
                if not column_values.empty
                else 0.0
            )

            patient_data[feature] = st.number_input(
                label,
                value=default_value,
                help=f"Fitur numerik: {feature}",
                key=f"input_{feature}",
            )

        elif feature in categorical_features:
            if feature in dataset.columns:
                # Pertahankan nilai asli agar sesuai dengan data training.
                options = [
                    value.item() if hasattr(value, "item") else value
                    for value in dataset[feature].dropna().unique()
                ]
            else:
                options = []

            if options:
                patient_data[feature] = st.selectbox(
                    label,
                    options=options,
                    format_func=lambda value: str(value),
                    key=f"input_{feature}",
                    help=f"Kategori fitur: {feature}",
                )
            else:
                st.warning(
                    f"Pilihan kategori untuk {feature} "
                    "tidak ditemukan pada dataset."
                )

        else:
            st.warning(
                f"Jenis fitur '{feature}' belum tercantum "
                "dalam metadata."
            )

    submitted = st.form_submit_button(
        "Prediksi Penyakit Jantung",
        use_container_width=True,
    )


# ==================================================
# 5. MENGIRIM DATA KE API
# ==================================================

if submitted:
    if len(patient_data) != len(feature_columns):
        st.error(
            "Tidak semua fitur berhasil ditampilkan. "
            "Periksa metadata dan dataset sebelum melakukan prediksi."
        )
        st.stop()

    payload = {"data": patient_data}

    with st.spinner("Mengirim data ke API dan menjalankan prediksi..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

            if response.status_code != 200:
                try:
                    error_detail = response.json().get(
                        "detail",
                        response.text,
                    )
                except ValueError:
                    error_detail = response.text

                st.error(
                    f"Prediksi gagal (HTTP {response.status_code})."
                )
                st.json(error_detail)
                st.stop()

            result = response.json()

        except requests.ConnectionError:
            st.error(
                "Tidak dapat terhubung ke API. "
                "Jalankan FastAPI terlebih dahulu di terminal lain "
                "menggunakan perintah: "
                "python -m uvicorn api:app --reload"
            )
            st.stop()

        except requests.Timeout:
            st.error(
                "Permintaan ke API melewati batas waktu. "
                "Silakan coba kembali."
            )
            st.stop()

        except requests.RequestException as error:
            st.error(f"Terjadi kesalahan komunikasi dengan API: {error}")
            st.stop()

        except ValueError:
            st.error("API mengembalikan respons yang tidak valid.")
            st.stop()

    # ==================================================
    # 6. MENAMPILKAN HASIL PREDIKSI
    # ==================================================

    st.subheader("2. Hasil Prediksi")

    prediction = result.get("prediction")
    label = result.get("label", "Tidak diketahui")
    probability = result.get("probability_percent")

    if prediction == 1:
        st.warning(
            "Model memprediksi adanya indikasi penyakit jantung."
        )
    elif prediction == 0:
        st.success(
            "Model tidak memprediksi adanya indikasi penyakit jantung."
        )
    else:
        st.error("Respons API tidak memiliki kelas prediksi yang valid.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Kelas Prediksi", prediction)

    with col2:
        if probability is not None:
            st.metric(
                "Probabilitas Kelas Indikasi",
                f"{probability:.2f}%",
            )

    st.write("Label hasil:", label)

    st.caption(
        result.get(
            "message",
            "Hasil ini merupakan prediksi model, bukan diagnosis medis.",
        )
    )

    with st.expander("Lihat detail respons API"):
        st.json(result)
