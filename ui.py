import os
from pathlib import Path
import json

import pandas as pd
import requests
import streamlit as st


# =========================================================
# 1. KONFIGURASI
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
HEALTH_URL = f"{API_URL}/health"
PREDICT_URL = f"{API_URL}/predict"

MODEL_DIR = BASE_DIR / "model"
METADATA_PATH = MODEL_DIR / "metadata.json"
DATASET_PATH = BASE_DIR / "heart_disease_uci.csv"

# Foto medis asli dari Pexels
HERO_IMAGE_URL = (
    "https://images.pexels.com/photos/5207096/"
    "pexels-photo-5207096.jpeg"
    "?cs=srgb&dl=pexels-karola-g-5207096.jpg&fm=jpg"
)

st.set_page_config(
    page_title="CardioCheck | Prediksi Penyakit Jantung",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. DARK THEME CSS
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        color-scheme: dark;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(37, 99, 235, 0.10),
                transparent 28%
            ),
            #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #101a2d;
        border-right: 1px solid #24324a;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: #101a2d;
    }

    [data-testid="stSidebar"] * {
        color: #dbe5f5;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
        letter-spacing: -0.5px;
    }

    p, li, label, .stMarkdown {
        color: #cbd5e1;
    }

    [data-testid="stCaptionContainer"] {
        color: #94a3b8;
    }

    [data-testid="stMetric"] {
        background: #111c30;
        border: 1px solid #26364f;
        border-radius: 15px;
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    [data-testid="stMetricDelta"] {
        color: #93c5fd !important;
    }

    div[data-testid="stForm"] {
        background: #111c30;
        border: 1px solid #26364f;
        border-radius: 18px;
        padding: 24px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #26364f !important;
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #0c1526 !important;
        color: #f1f5f9 !important;
        border-color: #34445e !important;
        border-radius: 9px !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background-color: #111c30 !important;
        color: #f1f5f9 !important;
    }

    [role="option"] {
        color: #f1f5f9 !important;
    }

    [role="option"]:hover {
        background-color: #243653 !important;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 10px;
        border: 1px solid #3b82f6;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: #ffffff !important;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        border-color: #60a5fa;
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        transform: translateY(-1px);
    }

    .hero {
        background: linear-gradient(
            135deg,
            rgba(20, 39, 70, 0.98),
            rgba(15, 23, 42, 0.98)
        );
        border: 1px solid #2b4264;
        border-radius: 22px;
        padding: 28px;
        margin-bottom: 25px;
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.18);
    }

    .eyebrow {
        color: #60a5fa;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: clamp(27px, 3.3vw, 42px);
        line-height: 1.18;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 14px;
    }

    .hero-desc {
        font-size: 15px;
        line-height: 1.8;
        color: #bdcbe0;
        margin-bottom: 0;
    }

    .section-card {
        background: #111c30;
        border: 1px solid #26364f;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .section-subtitle {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.7;
        margin-top: -4px;
    }

    .result-card {
        padding: 25px;
        border-radius: 16px;
        margin: 12px 0 20px 0;
        border: 1px solid;
    }

    .result-positive {
        background: rgba(127, 29, 29, 0.20);
        border-color: #7f353b;
    }

    .result-negative {
        background: rgba(6, 78, 59, 0.20);
        border-color: #23745c;
    }

    .result-label {
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .result-value {
        color: #f8fafc;
        font-size: clamp(23px, 3vw, 34px);
        font-weight: 800;
        line-height: 1.3;
        margin-bottom: 12px;
    }

    .result-description {
        color: #cbd5e1;
        font-size: 14px;
        line-height: 1.7;
    }

    .status-pill {
        display: inline-block;
        background: #172554;
        color: #bfdbfe;
        border: 1px solid #1d4ed8;
        padding: 6px 11px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
    }

    .small-note {
        color: #94a3b8;
        font-size: 12px;
        line-height: 1.7;
    }

    .footer {
        border-top: 1px solid #26364f;
        margin-top: 40px;
        padding-top: 20px;
        color: #64748b;
        font-size: 12px;
        text-align: center;
        line-height: 1.8;
    }

    hr {
        border-color: #26364f !important;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    [data-testid="stExpander"] {
        background: #111c30;
        border: 1px solid #26364f;
        border-radius: 12px;
    }

    [data-testid="stProgress"] > div > div {
        background-color: #3b82f6;
    }

    /* Mengurangi elemen Streamlit yang tidak diperlukan */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD METADATA DAN DATASET
# =========================================================

@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"File metadata tidak ditemukan: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATASET_PATH}"
        )

    return pd.read_csv(DATASET_PATH)


try:
    metadata = load_metadata()
    dataset = load_dataset()

except Exception as error:
    st.error(f"Gagal memuat file aplikasi: {error}")
    st.info(
        "Pastikan metadata.json berada di folder model "
        "dan heart_disease_uci.csv berada di folder utama proyek."
    )
    st.stop()


# =========================================================
# 4. AMBIL KONFIGURASI FITUR
# =========================================================

FEATURE_COLUMNS = metadata.get("feature_columns", [])

NUMERIC_FEATURES = metadata.get("numeric_features", [])
CATEGORICAL_FEATURES = metadata.get("categorical_features", [])

if not FEATURE_COLUMNS:
    st.error(
        "Daftar feature_columns tidak ditemukan di metadata.json."
    )
    st.stop()

# Pastikan fitur sesuai dengan kolom yang tersedia di dataset.
missing_features = [
    feature for feature in FEATURE_COLUMNS
    if feature not in dataset.columns
]

if missing_features:
    st.error(
        "Fitur berikut tidak ditemukan dalam dataset: "
        + ", ".join(missing_features)
    )
    st.stop()

# Jika metadata tidak memisahkan fitur numerik dan kategorikal,
# tipe data dataset digunakan sebagai fallback.
if not NUMERIC_FEATURES and not CATEGORICAL_FEATURES:
    NUMERIC_FEATURES = [
        feature for feature in FEATURE_COLUMNS
        if pd.api.types.is_numeric_dtype(dataset[feature])
    ]

    CATEGORICAL_FEATURES = [
        feature for feature in FEATURE_COLUMNS
        if feature not in NUMERIC_FEATURES
    ]


# =========================================================
# 5. FUNGSI BANTUAN
# =========================================================

def get_numeric_default(feature):
    """Mengambil nilai tengah dataset sebagai nilai awal input."""

    series = pd.to_numeric(dataset[feature], errors="coerce")
    median_value = series.median()

    if pd.isna(median_value):
        return 0.0

    return float(median_value)


def get_numeric_bounds(feature):
    """Mengambil batas input dari rentang nilai dataset."""

    series = pd.to_numeric(dataset[feature], errors="coerce")
    series = series.dropna()

    if series.empty:
        return 0.0, 100.0

    minimum = float(series.min())
    maximum = float(series.max())

    if minimum == maximum:
        maximum = minimum + 1.0

    return minimum, maximum


def get_category_options(feature):
    """Mengambil kategori unik dari dataset."""

    values = dataset[feature].dropna().unique().tolist()

    # Mengurutkan kategori bila memungkinkan.
    try:
        values = sorted(values)
    except TypeError:
        values = sorted(values, key=lambda value: str(value))

    return values


def format_feature_name(feature):
    """Membuat nama fitur lebih mudah dibaca."""

    return (
        str(feature)
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .title()
    )


def check_api_health():
    """Memeriksa apakah API FastAPI dapat diakses."""

    try:
        response = requests.get(HEALTH_URL, timeout=3)

        if response.ok:
            try:
                return True, response.json()
            except ValueError:
                return True, {}

        return False, {}

    except requests.RequestException:
        return False, {}


def send_prediction(patient_data):
    """
    Mengirim input ke API.
    Format payload dipertahankan sesuai api.py:
    POST /predict dengan JSON {"data": patient_data}.
    """

    payload = {"data": patient_data}

    response = requests.post(
        PREDICT_URL,
        json=payload,
        timeout=30,
    )

    response.raise_for_status()
    return response.json()


def get_prediction_text(result):
    """Mengambil label prediksi dari respons API."""

    label = result.get("label")

    if label is not None and str(label).strip():
        return str(label)

    prediction = result.get("prediction")

    if prediction is not None:
        try:
            prediction_number = int(prediction)

            if prediction_number == 0:
                return "Risiko lebih rendah"

            return "Indikasi penyakit jantung"

        except (ValueError, TypeError):
            return str(prediction)

    return "Hasil tidak tersedia"


def get_probability_percent(result):
    """Mengambil probabilitas dengan beberapa format respons."""

    probability_percent = result.get("probability_percent")

    if probability_percent is not None:
        try:
            value = float(probability_percent)

            # Jika respons sudah dalam bentuk persen, gunakan langsung.
            return value if value > 1 else value * 100

        except (ValueError, TypeError):
            pass

    probability = result.get("probability")

    if probability is not None:
        try:
            value = float(probability)
            return value * 100 if value <= 1 else value
        except (ValueError, TypeError):
            pass

    return None


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 20px 0;">
            <div style="font-size: 34px;">🫀</div>
            <div style="
                font-size: 23px;
                font-weight: 800;
                color: #f8fafc;
                margin-top: 5px;
            ">CardioCheck</div>
            <div style="
                color: #94a3b8;
                font-size: 12px;
                margin-top: 5px;
            ">Heart Disease Prediction System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Sistem")

    api_online, health_data = check_api_health()

    if api_online:
        st.success("API terhubung")
    else:
        st.error("API tidak terhubung")

    st.caption(f"Alamat API: {API_URL}")

    st.divider()

    st.markdown("### Informasi Model")
    st.markdown(
        """
        - **Algoritma:** XGBoost
        - **Jenis:** Klasifikasi
        - **Input:** Data klinis
        - **Output:** Hasil prediksi model
        """
    )

    st.divider()

    st.markdown("### Petunjuk")
    st.markdown(
        """
        1. Isi seluruh data yang diminta.
        2. Periksa kembali nilai input.
        3. Tekan tombol prediksi.
        4. Baca hasil yang ditampilkan.
        """
    )

    st.markdown(
        """
        <div class="small-note" style="margin-top: 25px;">
            Sistem ini merupakan prototipe edukasi dan bukan
            pengganti pemeriksaan tenaga medis.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 7. HERO SECTION
# =========================================================

hero_col1, hero_col2 = st.columns([1.35, 1], gap="large")

with hero_col1:
    st.markdown(
        """
        <div style="padding: 20px 0 10px 0;">
            <div class="eyebrow">AI-POWERED HEALTH SCREENING</div>
            <div class="hero-title">
                Kenali Risiko.<br>
                Jaga Kesehatan Jantung.
            </div>
            <p class="hero-desc">
                Sistem prediksi berbasis machine learning yang
                membantu menganalisis data klinis dan menyajikan
                hasil klasifikasi secara cepat dan terstruktur.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin-top: 15px; margin-bottom: 22px;">
            <span class="status-pill">● XGBoost Model</span>
            &nbsp;
            <span class="status-pill">13 Clinical Features</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_col2:
    try:
        st.image(
            HERO_IMAGE_URL,
            use_container_width=True,
        )
    except Exception:
        st.markdown(
            """
            <div class="section-card"
                 style="text-align:center; padding:55px 20px;">
                <div style="font-size:65px;">🫀</div>
                <h3>CardioCheck</h3>
                <p>Heart Disease Prediction</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()


# =========================================================
# 8. RINGKASAN APLIKASI
# =========================================================

st.markdown("### Dashboard Overview")
st.markdown(
    '<p class="section-subtitle">'
    'Ringkasan sistem prediksi dan data yang digunakan.'
    '</p>',
    unsafe_allow_html=True,
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        label="Jumlah Fitur",
        value=len(FEATURE_COLUMNS),
        help="Jumlah fitur input yang digunakan model.",
    )

with metric2:
    st.metric(
        label="Jumlah Data Dataset",
        value=f"{len(dataset):,}",
        help="Jumlah baris pada dataset yang tersedia.",
    )

with metric3:
    st.metric(
        label="Status API",
        value="Online" if api_online else "Offline",
        help="Status koneksi ke layanan FastAPI.",
    )

st.markdown("")
st.divider()


# =========================================================
# 9. FORM INPUT PASIEN
# =========================================================

st.markdown("## Input Data Pasien")
st.markdown(
    """
    <p class="section-subtitle">
        Masukkan data sesuai informasi yang tersedia.
        Nilai awal numerik menggunakan median dataset, sedangkan
        pilihan kategori diambil dari nilai yang tersedia pada dataset.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.form("prediction_form", clear_on_submit=False):
    patient_data = {}

    # Dua kolom agar form lebih mudah dibaca.
    input_col1, input_col2 = st.columns(2, gap="large")

    for index, feature in enumerate(FEATURE_COLUMNS):
        target_col = input_col1 if index % 2 == 0 else input_col2

        with target_col:
            display_name = format_feature_name(feature)

            if feature in CATEGORICAL_FEATURES:
                options = get_category_options(feature)

                if not options:
                    st.warning(
                        f"Tidak ada pilihan kategori untuk {display_name}."
                    )
                    patient_data[feature] = None
                else:
                    # Mengambil nilai kategori pertama sebagai nilai awal.
                    default_index = 0

                    patient_data[feature] = st.selectbox(
                        display_name,
                        options=options,
                        index=default_index,
                        key=f"input_{feature}",
                        help=f"Pilih nilai untuk fitur {feature}.",
                    )

            elif feature in NUMERIC_FEATURES or pd.api.types.is_numeric_dtype(
                dataset[feature]
            ):
                default_value = get_numeric_default(feature)
                minimum, maximum = get_numeric_bounds(feature)

                # Integer digunakan bila nilai dataset memang bilangan bulat.
                numeric_series = pd.to_numeric(
                    dataset[feature],
                    errors="coerce",
                ).dropna()

                is_integer = (
                    not numeric_series.empty
                    and (numeric_series % 1 == 0).all()
                )

                if is_integer:
                    min_value = int(minimum)
                    max_value = int(maximum)
                    value = int(round(default_value))

                    patient_data[feature] = st.number_input(
                        display_name,
                        min_value=min_value,
                        max_value=max_value,
                        value=max(min_value, min(max_value, value)),
                        step=1,
                        key=f"input_{feature}",
                        help=f"Rentang berdasarkan dataset: {min_value}–{max_value}.",
                    )

                else:
                    # Sedikit toleransi agar pengguna dapat mengisi
                    # nilai desimal di luar rentang historis dataset.
                    range_padding = max(
                        abs(maximum - minimum) * 0.10,
                        1.0,
                    )

                    min_input = minimum - range_padding
                    max_input = maximum + range_padding

                    patient_data[feature] = st.number_input(
                        display_name,
                        min_value=float(min_input),
                        max_value=float(max_input),
                        value=float(default_value),
                        step=0.1,
                        format="%.2f",
                        key=f"input_{feature}",
                        help="Nilai awal menggunakan median dataset.",
                    )

            else:
                # Fallback jika tipe fitur tidak dikenali.
                options = get_category_options(feature)

                if options:
                    patient_data[feature] = st.selectbox(
                        display_name,
                        options=options,
                        key=f"input_{feature}",
                    )
                else:
                    patient_data[feature] = st.text_input(
                        display_name,
                        key=f"input_{feature}",
                    )

    st.markdown("---")

    submitted = st.form_submit_button(
        "🩺  Analisis Data dan Prediksi",
        use_container_width=True,
    )


# =========================================================
# 10. PROSES PREDIKSI
# =========================================================

if submitted:
    if not api_online:
        st.error(
            "API belum terhubung. Jalankan api.py terlebih dahulu "
            "sebelum melakukan prediksi."
        )

    elif any(value is None for value in patient_data.values()):
        st.error(
            "Ada fitur yang belum memiliki nilai. "
            "Periksa kembali form input."
        )

    else:
        with st.spinner("Model sedang menganalisis data..."):
            try:
                result = send_prediction(patient_data)

                st.session_state["last_prediction"] = result
                st.session_state["last_patient_data"] = patient_data

            except requests.exceptions.Timeout:
                st.error(
                    "Permintaan prediksi melewati batas waktu. "
                    "Coba kembali beberapa saat lagi."
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Tidak dapat terhubung ke API. "
                    "Pastikan FastAPI berjalan pada port 8000."
                )

            except requests.exceptions.HTTPError as error:
                response = error.response
                detail = ""

                if response is not None:
                    try:
                        detail = response.json().get(
                            "detail",
                            response.text,
                        )
                    except ValueError:
                        detail = response.text

                st.error(f"API mengembalikan kesalahan: {detail}")

            except (ValueError, TypeError) as error:
                st.error(f"Format respons API tidak sesuai: {error}")

            except Exception as error:
                st.error(f"Terjadi kesalahan: {error}")


# =========================================================
# 11. TAMPILKAN HASIL PREDIKSI
# =========================================================

if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]

    st.divider()
    st.markdown("## Hasil Analisis")
    st.markdown(
        """
        <p class="section-subtitle">
            Berikut hasil klasifikasi yang dikembalikan oleh model.
        </p>
        """,
        unsafe_allow_html=True,
    )

    prediction_text = get_prediction_text(result)
    probability_percent = get_probability_percent(result)

    # Tentukan warna kartu berdasarkan label/kelas prediksi.
    label_lower = prediction_text.lower()

    is_positive = any(
        keyword in label_lower
        for keyword in [
            "penyakit",
            "positive",
            "positif",
            "risk",
            "risiko tinggi",
            "high",
            "1",
        ]
    )

    # Jika respons menyediakan nilai kelas, gunakan sebagai petunjuk.
    raw_prediction = result.get("prediction")

    try:
        if int(raw_prediction) == 0:
            is_positive = False
        elif int(raw_prediction) == 1:
            is_positive = True
    except (ValueError, TypeError):
        pass

    result_class = (
        "result-positive" if is_positive else "result-negative"
    )

    st.markdown(
        f"""
        <div class="result-card {result_class}">
            <div class="result-label">Hasil Klasifikasi Model</div>
            <div class="result-value">{prediction_text}</div>
            <div class="result-description">
                Hasil ini merupakan keluaran model machine learning
                berdasarkan data yang dimasukkan.
                Interpretasi medis tetap memerlukan tenaga profesional.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result_col1, result_col2 = st.columns(2, gap="large")

    with result_col1:
        if probability_percent is not None:
            st.metric(
                label="Probabilitas Model",
                value=f"{probability_percent:.2f}%",
            )

            st.progress(
                min(max(probability_percent / 100, 0.0), 1.0)
            )
        else:
            st.metric(
                label="Probabilitas Model",
                value="Tidak tersedia",
            )

    with result_col2:
        st.metric(
            label="Kelas Prediksi",
            value=str(result.get("prediction", "-")),
        )

        st.caption(
            "Kelas prediksi ditampilkan sesuai respons API."
        )

    if result.get("message"):
        st.info(str(result["message"]))

    with st.expander("Lihat detail respons API"):
        st.json(result)

    with st.expander("Lihat kembali data input"):
        last_patient_data = st.session_state.get(
            "last_patient_data",
            {},
        )

        input_df = pd.DataFrame(
            list(last_patient_data.items()),
            columns=["Fitur", "Nilai"],
        )

        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "Penting: hasil prediksi ini bukan diagnosis medis. "
        "Probabilitas model tidak otomatis sama dengan probabilitas "
        "seseorang benar-benar mengalami penyakit jantung. "
        "Konsultasikan hasil pemeriksaan dengan tenaga medis."
    )


# =========================================================
# 12. FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        <strong style="color:#cbd5e1;">CardioCheck</strong><br>
        Heart Disease Prediction System · XGBoost<br>
        Prototipe untuk keperluan pembelajaran dan demonstrasi.
        Bukan alat diagnosis medis.
    </div>
    """,
    unsafe_allow_html=True,
)