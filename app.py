
import json
import sqlite3
from pathlib import Path
from datetime import datetime

import altair as alt
import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SmartAgriAI | Smart Farming",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "crop_model.pkl"

DB_PATH = BASE_DIR / "smartagri_history.db"

PAGES = [
    "Dashboard Home",
    "Model Analytics",
    "Crop Recommendation",
    "Soil Health",
    "Irrigation Advisory",
    "Farmer Profile",
    "Analysis History",
]

FEATURES = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall",
]


CROP_EMOJI = {
    # Cereals and grains
    "rice": "🌾",
    "maize": "🌽",
    "wheat": "🌾",
    "barley": "🌾",
    "sorghum": "🌾",
    "pearl millet": "🌾",
    "finger millet": "🌾",

    # Pulses and legumes
    "chickpea": "🫘",
    "kidneybeans": "🫘",
    "kidney beans": "🫘",
    "pigeonpeas": "🫘",
    "pigeon peas": "🫘",
    "mothbeans": "🫘",
    "mungbean": "🫘",
    "mung bean": "🫘",
    "blackgram": "🫘",
    "black gram": "🫘",
    "lentil": "🫘",
    "field pea": "🫘",
    "soybean": "🫘",

    # Fruits
    "pomegranate": "🍎",
    "banana": "🍌",
    "mango": "🥭",
    "grapes": "🍇",
    "watermelon": "🍉",
    "muskmelon": "🍈",
    "apple": "🍎",
    "orange": "🍊",
    "papaya": "🥭",
    "coconut": "🥥",

    # Vegetables
    "tomato": "🍅",
    "potato": "🥔",
    "onion": "🧅",
    "okra": "🥬",
    "chilli": "🌶️",
    "cabbage": "🥬",
    "cauliflower": "🥦",

    # Oilseeds and cash crops
    "mustard": "🌼",
    "groundnut": "🥜",
    "sesame": "🌱",
    "sunflower": "🌻",
    "cotton": "🌿",
    "jute": "🌱",
    "sugarcane": "🎋",
    "coffee": "☕",
}

PAGE_ICONS = {
    "Dashboard Home": "⌂",
    "Model Analytics": "▥",
    "Crop Recommendation": "❧",
    "Soil Health": "⚗",
    "Irrigation Advisory": "◉",
    "Farmer Profile": "♙",
    "Analysis History": "◷",
}


# =========================================================
# 2. STYLING
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f6faf7;
        color: #173f35;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #dcebe0;
    }

    h1, h2, h3 {
        color: #145c46;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 0 24px 0;
    }

    .brand-icon {
        font-size: 36px;
    }

    .brand-name {
        color: #145c46;
        font-size: 23px;
        font-weight: 800;
        line-height: 1.2;
    }

    .brand-subtitle {
        color: #71877e;
        font-size: 10px;
        letter-spacing: 1px;
        margin-top: 5px;
    }

    .nav-label {
        color: #71877e;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.7px;
        margin: 4px 0 12px 0;
    }

    .hero {
        background: linear-gradient(110deg, #e7f5e9, #fbfdfb);
        border: 1px solid #d8eadd;
        border-radius: 18px;
        padding: 28px;
        margin: 10px 0 22px 0;
    }

    .hero h1 {
        color: #145c46;
        font-size: clamp(26px, 3vw, 36px);
        margin: 0 0 10px 0;
    }

    .hero p {
        color: #496b60;
        font-size: 16px;
        margin: 0;
        line-height: 1.6;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid #dcebe0;
        border-radius: 14px;
        padding: 18px;
    }

    .muted {
        color: #71877e;
        font-size: 13px;
    }

    .success-panel {
        background: #e9f6ed;
        border: 1px solid #c9e8d1;
        border-radius: 14px;
        padding: 20px;
        margin: 12px 0;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dcebe0;
        border-radius: 14px;
        padding: 14px 16px;
        min-width: 0;
        overflow: hidden;
    }

    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
    }

    [data-testid="stMetricValue"] {
        color: #197653;
        font-size: clamp(16px, 1.45vw, 24px) !important;
        line-height: 1.35 !important;
        overflow-wrap: anywhere;
        word-break: normal;
    }

    [data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow-wrap: anywhere;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 6px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 10px;
        padding: 8px 10px;
        color: #234c40;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #edf7ef;
    }

    div.stButton > button {
        border-radius: 10px;
        border: 1px solid #cfe3d5;
        font-weight: 600;
        min-height: 42px;
    }

    div.stButton > button[kind="primary"] {
        background: #197653;
        border-color: #197653;
        color: #ffffff;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #145c46;
        border-color: #145c46;
        color: #ffffff;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #dcebe0;
        border-radius: 10px;
        overflow: hidden;
    }

    div[data-testid="stForm"] {
        background: #ffffff;
        padding: 18px;
        border: 1px solid #dcebe0;
        border-radius: 14px;
    }
    
    /* ===== GLOBAL INPUT STYLING ===== */

    /* Selected dropdown values: crop names like Rice */
    div[data-baseweb="select"] div[class*="singleValue"] {
        font-size: 1.08rem !important;
        font-weight: 800 !important;
        color: #173f35 !important;
    }

    /* Dropdown options */
    div[role="option"] {
        font-weight: 600 !important;
    }

    /* Numeric values: rainfall, temperature, etc. */
    div[data-testid="stNumberInput"] input {
        font-size: 1.15rem !important;
        font-weight: 800 !important;
        color: #173f35 !important;
    }

    /* Plus and minus buttons */
    div[data-testid="stNumberInput"] button {
        font-size: 1.25rem !important;
        font-weight: 900 !important;
        color: #145c46 !important;
    }

    /* Slider's current numeric value */
    div[data-testid="stSlider"] [data-testid="stThumbValue"] {
        font-size: 1.1rem !important;
        font-weight: 800 !important;
        color: #197653 !important;
    }

    /* Input labels across all pages */
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {
        font-weight: 600 !important;
        color: #234c40 !important;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. MODEL
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


model = load_model()


# =========================================================
# 4. DATABASE
# =========================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                farmer_name TEXT NOT NULL,
                village TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                result TEXT NOT NULL,
                input_json TEXT
            )
            """
        )
        conn.commit()


initialize_database()


def save_analysis(analysis_type, result, inputs=None):
    name = st.session_state.get("farmer_name", "Farmer")
    village = st.session_state.get("village", "")

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO analysis_history
            (created_at, farmer_name, village,
             analysis_type, result, input_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                name,
                village,
                analysis_type,
                str(result),
                json.dumps(inputs or {}),
            ),
        )
        conn.commit()


def get_history(limit=None):
    name = st.session_state.get("farmer_name", "Farmer")
    village = st.session_state.get("village", "")

    query = """
        SELECT id, created_at, farmer_name, village,
               analysis_type, result, input_json
        FROM analysis_history
        WHERE farmer_name = ? AND village = ?
        ORDER BY id DESC
    """
    params = [name, village]

    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))

    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


# =========================================================
# 5. SESSION STATE
# =========================================================

if "farmer_name" not in st.session_state:
    st.session_state.farmer_name = ""

if "village" not in st.session_state:
    st.session_state.village = ""

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Dashboard Home"


# =========================================================
# 6. HELPER FUNCTIONS
# =========================================================

def go_to_page(page):
    st.session_state.nav_page = page


def crop_emoji(crop):
    return CROP_EMOJI.get(str(crop).lower(), "🌱")


def pretty_date(value):
    try:
        return datetime.strptime(
            str(value), "%Y-%m-%d %H:%M:%S"
        ).strftime("%d %b %Y, %I:%M %p")
    except (ValueError, TypeError):
        return str(value)


def show_page_heading(title, subtitle):
    st.title(title)
    st.caption(subtitle)


def get_crop_importances():
    if model is None or not hasattr(model, "feature_importances_"):
        return None

    values = list(model.feature_importances_)

    if len(values) != len(FEATURES):
        return None

    labels = {
        "N": "Nitrogen",
        "P": "Phosphorus",
        "K": "Potassium",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "ph": "Soil pH",
        "rainfall": "Rainfall",
    }

    explanations = {
        "Nitrogen": "Supports leaf growth and plant development.",
        "Phosphorus": "Helps root growth and flowering.",
        "Potassium": "Supports plant strength and crop quality.",
        "Temperature": "Affects germination and crop growth.",
        "Humidity": "Influences moisture availability for plants.",
        "Soil pH": "Affects nutrient availability in soil.",
        "Rainfall": "Contributes to water availability for crops.",
    }

    df = pd.DataFrame({
        "Feature": [labels[f] for f in FEATURES],
        "Importance": values,
    })

    df["Explanation"] = df["Feature"].map(explanations)

    return df.sort_values("Importance", ascending=False)


# =========================================================
# 7. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <span class="brand-icon">🌱</span>
            <div>
                <div class="brand-name">SmartAgriAI</div>
                <div class="brand-subtitle">INTELLIGENT AGRICULTURE</div>
            </div>
        </div>
        <div class="nav-label">WORKSPACE</div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = st.radio(
        "Navigation",
        options=PAGES,
        key="nav_page",
        label_visibility="collapsed",
        format_func=lambda page: (
            f"{PAGE_ICONS[page]}    {page}"
        ),
    )

    st.divider()

    if st.session_state.farmer_name:
        st.markdown("### 👨‍🌾 FARMER ACCOUNT")
        st.markdown(f"**{st.session_state.farmer_name}**")
        st.caption(st.session_state.village)

        st.button(
            "✎  Edit Profile",
            use_container_width=True,
            on_click=go_to_page,
            args=("Farmer Profile",),
        )
    else:
        st.caption("Set up your farmer profile to begin.")

    st.divider()
    st.caption("🌿 Smart farming • Better tomorrow")


# =========================================================
# 8. FIRST-TIME FARMER PROFILE
# =========================================================

if not st.session_state.farmer_name:
    st.markdown(
        """
        <div class="hero">
            <h1>Welcome to SmartAgriAI 🌱</h1>
            <p>
                Your smart farming assistant for crop recommendations,
                soil information and irrigation guidance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Create your farmer profile")

    with st.form("initial_profile_form"):
        name = st.text_input("Farmer name")
        village = st.text_input("Village / location")

        create_profile = st.form_submit_button(
            "Continue to Dashboard",
            type="primary",
            use_container_width=True,
        )

    if create_profile:
        if not name.strip() or not village.strip():
            st.warning("Please enter both your name and village.")
        else:
            st.session_state.farmer_name = name.strip()
            st.session_state.village = village.strip()
            st.rerun()

    st.stop()


# =========================================================
# 9. DASHBOARD HOME — OVERVIEW
# =========================================================

if selected_page == "Dashboard Home":

    name = st.session_state.farmer_name
    village = st.session_state.village

    history = get_history()
    recent = history.head(4).copy()

    crop_history = history[
        history["analysis_type"] == "Crop Recommendation"
    ]
    soil_history = history[
        history["analysis_type"] == "Soil Health"
    ]
    irrigation_history = history[
        history["analysis_type"] == "Irrigation Advisory"
    ]

    latest_crop = (
        crop_history.iloc[0] if not crop_history.empty else None
    )
    latest_soil = (
        soil_history.iloc[0] if not soil_history.empty else None
    )
    latest_irrigation = (
        irrigation_history.iloc[0]
        if not irrigation_history.empty else None
    )

    # Header
    header_left, header_right = st.columns([3, 1])

    with header_left:
        st.markdown(f"### 🌿 Welcome back, {name}!")
        st.caption("Your smart farming assistant")

    with header_right:
        st.markdown(f"📍 **{village}**")
        st.caption("Your profile location")

    # Hero
    st.markdown(
        """
        <div class="hero">
            <h1>Grow Better. Farm Smarter.</h1>
            <p>
                Get crop suggestions and practical farming guidance
                based on your soil and climate inputs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Determine compact status values
    if latest_crop is not None:
        latest_crop_name = str(latest_crop["result"])
        if latest_crop_name.lower().startswith("recommended crop:"):
            latest_crop_name = latest_crop_name.split(":", 1)[1].strip()
    else:
        latest_crop_name = "Not available yet"

    soil_status = (
        str(latest_soil["result"])
        if latest_soil is not None else "Not checked yet"
    )
    irrigation_status = (
        str(latest_irrigation["result"])
        if latest_irrigation is not None else "Not checked yet"
    )

    # Compact overview cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total Analyses", len(history))
        st.caption("Saved on this device")

    with c2:
        st.metric(
            "Latest Crop",
            latest_crop_name if latest_crop is not None else "—",
            help=latest_crop_name,
        )
        st.caption(
            "Latest saved recommendation"
            if latest_crop is not None
            else "No recommendation yet"
        )

    with c3:
        st.metric(
            "Soil Health",
            "Checked" if latest_soil is not None else "Not checked",
            help=soil_status,
        )
        st.caption(
            pretty_date(latest_soil["created_at"])
            if latest_soil is not None
            else "Complete a soil analysis"
        )

    with c4:
        st.metric(
            "Irrigation",
            "Reviewed" if latest_irrigation is not None else "Not checked",
            help=irrigation_status,
        )
        st.caption(
            pretty_date(latest_irrigation["created_at"])
            if latest_irrigation is not None
            else "Get irrigation guidance"
        )

    st.write("")

    # Latest crop and quick actions
    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown("### 🌱 Latest Crop Recommendation")

        if latest_crop is not None:
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="muted">YOUR LATEST SAVED RESULT</div>
                    <div style="font-size:28px;font-weight:800;
                                color:#197653;margin:12px 0;">
                        {crop_emoji(latest_crop_name)}
                        {latest_crop_name}
                    </div>
                    <div class="muted">
                        Saved {pretty_date(latest_crop["created_at"])}
                    </div>
                    <p style="margin-top:12px;">
                        Review the saved prediction and the measurements
                        used to generate it before making a planting decision.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="section-card">
                    <div style="font-size:48px;">🌱</div>
                    <b>You haven't saved a crop recommendation yet.</b>
                    <p class="muted">
                        Start by entering your soil and climate measurements.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.button(
            "🌱  Get Crop Recommendation",
            type="primary",
            use_container_width=True,
            on_click=go_to_page,
            args=("Crop Recommendation",),
        )

    with right:
        st.markdown("### ⚡ Quick Actions")
        st.caption("Open the tool you need")

        quick_actions = [
            ("🌱  Find the Right Crop", "Crop Recommendation"),
            ("🧪  Check Soil Health", "Soil Health"),
            ("💧  View Irrigation Advisory", "Irrigation Advisory"),
            ("📊  Explore Model Analytics", "Model Analytics"),
        ]

        for label, target_page in quick_actions:
            st.button(
                label,
                key=f"quick_{target_page}",
                use_container_width=True,
                on_click=go_to_page,
                args=(target_page,),
            )

    st.write("")

    # Recent activity
    activity_left, activity_right = st.columns([3, 1])

    with activity_left:
        st.markdown("### 🕒 Recent Activity")
        st.caption("Your latest saved analyses and recommendations")

    with activity_right:
        st.button(
            "View All History →",
            use_container_width=True,
            on_click=go_to_page,
            args=("Analysis History",),
        )

    if recent.empty:
        st.markdown(
            """
            <div class="section-card" style="text-align:center;padding:28px;">
                <div style="font-size:35px;">🗓️</div>
                <b>No activity yet</b>
                <p class="muted">
                    Complete an analysis to see your recent activity here.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        activity = recent.copy()
        activity["created_at"] = activity["created_at"].apply(pretty_date)
        activity = activity.rename(columns={
            "created_at": "Date & Time",
            "analysis_type": "Type",
            "result": "Result",
        })

        st.dataframe(
            activity[["Date & Time", "Type", "Result"]],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;padding:10px;color:#477362;">
            🌿 <b>Healthy soil grows healthy crops.</b><br>
            <span class="muted">
                Smart decisions today can support sustainable farming.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )



# =========================================================
# 10. CROP RECOMMENDATION
# =========================================================

elif selected_page == "Crop Recommendation":

    st.title("🌱 Crop Recommendation")
    st.caption(
        "Find a crop suggestion using your soil nutrient "
        "and climate measurements."
    )

    # Check whether the trained model is available
    if model is None:
        st.error(
            "The trained model could not be loaded. "
            "Please check that models/crop_model.pkl exists."
        )
        st.stop()

    # Labels shown to the farmer
    feature_labels = {
        "N": "Nitrogen (N)",
        "P": "Phosphorus (P)",
        "K": "Potassium (K)",
        "temperature": "Temperature (°C)",
        "humidity": "Humidity (%)",
        "ph": "Soil pH",
        "rainfall": "Rainfall (mm)",
    }

    # Store results between Streamlit reruns
    if "crop_result" not in st.session_state:
        st.session_state.crop_result = None

    if "crop_inputs" not in st.session_state:
        st.session_state.crop_inputs = None

    if "crop_top_predictions" not in st.session_state:
        st.session_state.crop_top_predictions = []

    if "crop_saved" not in st.session_state:
        st.session_state.crop_saved = False

    # -----------------------------------------------------
    # INPUT FORM
    # -----------------------------------------------------

    st.markdown("### 1. Enter soil and climate details")

    st.info(
        "Use measurements from a soil test and reliable local "
        "weather information whenever possible."
    )

    with st.form("crop_recommendation_form"):

        col1, col2 = st.columns(2)

        with col1:
            nitrogen = st.number_input(
                "Nitrogen (N)",
                min_value=0.0,
                max_value=200.0,
                value=50.0,
                step=1.0,
                help="Nitrogen measurement in the units used by your dataset.",
            )

            phosphorus = st.number_input(
                "Phosphorus (P)",
                min_value=0.0,
                max_value=200.0,
                value=50.0,
                step=1.0,
                help="Phosphorus measurement in the units used by your dataset.",
            )

            potassium = st.number_input(
                "Potassium (K)",
                min_value=0.0,
                max_value=300.0,
                value=50.0,
                step=1.0,
                help="Potassium measurement in the units used by your dataset.",
            )

            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-10.0,
                max_value=60.0,
                value=25.0,
                step=0.5,
            )

        with col2:
            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0,
            )

            ph = st.number_input(
                "Soil pH",
                min_value=0.0,
                max_value=14.0,
                value=6.5,
                step=0.1,
                help="Enter the pH from your soil test.",
            )

            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                max_value=5000.0,
                value=100.0,
                step=5.0,
                help="Use rainfall measurements appropriate to your model's training data.",
            )

        submitted = st.form_submit_button(
            "🌱 Analyse and Recommend Crop",
            type="primary",
            use_container_width=True,
        )

    # -----------------------------------------------------
    # RUN MODEL WHEN FORM IS SUBMITTED
    # -----------------------------------------------------

    if submitted:

        values = {
            "N": nitrogen,
            "P": phosphorus,
            "K": potassium,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }

        try:
            # Keep feature order consistent with model training
            input_df = pd.DataFrame(
                [[values[feature] for feature in FEATURES]],
                columns=FEATURES,
            )

            # Predict the recommended crop
            prediction = str(model.predict(input_df)[0])

            # Save the result in session state so it survives reruns
            st.session_state.crop_result = prediction
            st.session_state.crop_inputs = values
            st.session_state.crop_top_predictions = []
            st.session_state.crop_saved = False

            # Show alternatives only if the model supports probabilities
            if (
                hasattr(model, "predict_proba")
                and hasattr(model, "classes_")
            ):
                probabilities = model.predict_proba(input_df)[0]
                classes = model.classes_

                ranked = sorted(
                    zip(classes, probabilities),
                    key=lambda item: item[1],
                    reverse=True,
                )[:3]

                st.session_state.crop_top_predictions = [
                    {
                        "crop": str(crop),
                        "probability": float(probability),
                    }
                    for crop, probability in ranked
                ]

        except Exception as exc:
            st.error(
                f"Could not generate a recommendation: {exc}"
            )

    # -----------------------------------------------------
    # DISPLAY PERSISTENT RESULT
    # -----------------------------------------------------

    if st.session_state.crop_result is not None:

        prediction = st.session_state.crop_result
        saved_inputs = st.session_state.crop_inputs or {}

        st.markdown("---")
        st.markdown("### 2. Your AI recommendation")

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(110deg, #e7f5e9, #ffffff);
                border: 1px solid #cfe5d5;
                border-radius: 16px;
                padding: 24px;
                margin: 10px 0;
            ">
                <div style="
                    color: #527565;
                    font-size: 13px;
                    font-weight: 600;
                ">
                    RECOMMENDED CROP
                </div>
                <div style="
                    color: #145c46;
                    font-size: 32px;
                    font-weight: 800;
                    margin-top: 8px;
                ">
                    🌱 {prediction}
                </div>
                <div style="
                    color: #527565;
                    margin-top: 8px;
                    font-size: 14px;
                ">
                    Prediction generated by your trained machine-learning model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # TOP THREE PREDICTIONS, WHEN AVAILABLE
        # -------------------------------------------------

        alternatives = st.session_state.crop_top_predictions

        if alternatives:
            st.markdown("#### Other model predictions")
            st.caption(
                "Probabilities describe the model's output, not the "
                "actual probability of a successful harvest."
            )

            for item in alternatives:
                crop_name = item["crop"]
                probability = item["probability"]

                left, right = st.columns([3, 1])

                with left:
                    st.write(
                        f"{crop_emoji(crop_name)} **{crop_name}**"
                    )

                with right:
                    st.write(f"{probability:.1%}")

                st.progress(
                    min(max(probability, 0.0), 1.0)
                )

        # -------------------------------------------------
        # INPUT SUMMARY
        # -------------------------------------------------

        st.markdown("#### Measurements used")

        summary = pd.DataFrame({
            "Measurement": [
                feature_labels[feature]
                for feature in FEATURES
            ],
            "Value": [
                saved_inputs.get(feature, "")
                for feature in FEATURES
            ],
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------------------------------
        # SAVE RESULT
        # -------------------------------------------------

        st.markdown("#### Save your result")

        if st.session_state.crop_saved:
            st.success(
                "This recommendation has been saved to your Analysis History."
            )
        else:
            if st.button(
                "💾 Save Recommendation to History",
                type="primary",
                use_container_width=True,
            ):
                try:
                    save_analysis(
                        "Crop Recommendation",
                        f"Recommended Crop: {prediction}",
                        saved_inputs,
                    )

                    st.session_state.crop_saved = True
                    st.success("Recommendation saved successfully!")
                    st.rerun()

                except Exception as exc:
                    st.error(f"Could not save the result: {exc}")

        st.caption(
            "Generate a new recommendation by changing the inputs above "
            "and submitting the form again."
        )

    # -----------------------------------------------------
    # FARMING DISCLAIMER
    # -----------------------------------------------------

    st.markdown("---")

    st.caption(
        "Important: This prediction is decision-support information, "
        "not a guarantee of crop suitability or yield. Consider local "
        "season, soil-test interpretation, water availability, pests, "
        "and advice from an agricultural expert before planting."
    )


# =========================================================
# 11. MODEL ANALYTICS
# =========================================================

elif selected_page == "Model Analytics":

    show_page_heading(
        "📊 Model Analytics",
        "Understand which factors influence the trained model.",
    )

    if model is None:
        st.error("The trained model could not be loaded.")
        st.stop()

    importance_df = get_crop_importances()

    if importance_df is None:
        st.warning(
            "Feature importance is not available for this saved model."
        )
    else:
        st.markdown("### What factors matter to the model?")
        st.caption(
            "Feature importance describes each feature's relative "
            "contribution to the model. It is not a measure of causation."
        )

        chart = (
            alt.Chart(importance_df)
            .mark_bar(cornerRadiusEnd=5, color="#23845d")
            .encode(
                x=alt.X(
                    "Importance:Q",
                    title="Relative model importance",
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y("Feature:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("Feature:N"),
                    alt.Tooltip(
                        "Importance:Q",
                        format=".1%",
                        title="Importance",
                    ),
                    alt.Tooltip("Explanation:N"),
                ],
            )
            .properties(height=300)
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown("### Understanding the factors")

        for _, row in importance_df.iterrows():
            with st.container(border=True):
                left, right = st.columns([1, 4])
                with left:
                    st.markdown(f"**{row['Feature']}**")
                    st.caption(f"{row['Importance']:.1%}")
                with right:
                    st.write(row["Explanation"])

    st.button(
        "Open Crop Recommendation",
        type="primary",
        on_click=go_to_page,
        args=("Crop Recommendation",),
    )


# =========================================================
# 12. SOIL HEALTH
# =========================================================


elif selected_page == "Soil Health":

    show_page_heading(
        "🧪 Soil Health",
        "Review soil nutrients and pH, understand potential concerns, "
        "and identify the next steps for soil testing.",
    )

    st.info(
        "Enter your soil-test values. The guidance below is a basic "
        "screening aid, not a laboratory diagnosis or fertilizer prescription."
    )

    with st.form("soil_form"):

        st.markdown("### 1. Soil measurements")
        st.caption(
            "Enter NPK values in the units used by your soil-test report."
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            n = st.number_input(
                "Nitrogen (N)",
                min_value=0.0,
                max_value=200.0,
                value=50.0,
                step=1.0,
            )

        with c2:
            p = st.number_input(
                "Phosphorus (P)",
                min_value=0.0,
                max_value=200.0,
                value=50.0,
                step=1.0,
            )

        with c3:
            k = st.number_input(
                "Potassium (K)",
                min_value=0.0,
                max_value=300.0,
                value=50.0,
                step=1.0,
            )

        ph_value = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1,
            help="Use the pH measured in your soil-test report.",
        )

        soil_submit = st.form_submit_button(
            "🔬 Analyze Soil Health",
            type="primary",
            use_container_width=True,
        )

    if soil_submit:

        st.session_state["soil_result"] = {
            "N": n,
            "P": p,
            "K": k,
            "ph": ph_value,
        }

        st.session_state["soil_saved"] = False

    # Keep the analysis visible after Streamlit reruns
    soil_result = st.session_state.get("soil_result")

    if soil_result is not None:

        n = soil_result["N"]
        p = soil_result["P"]
        k = soil_result["K"]
        ph_value = soil_result["ph"]

        st.markdown("---")
        st.markdown("### 2. Soil analysis overview")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Nitrogen (N)", f"{n:g}")
        m2.metric("Phosphorus (P)", f"{p:g}")
        m3.metric("Potassium (K)", f"{k:g}")
        m4.metric("Soil pH", f"{ph_value:.1f}")

        # pH screening
        st.markdown("### 3. Soil pH interpretation")

        if ph_value < 5.5:
            ph_status = "Strongly acidic screening range"
            ph_advice = (
                "Some crops may struggle in strongly acidic soil. "
                "Consider a laboratory soil test and local agricultural "
                "advice before applying lime or other amendments."
            )
            st.warning(f"⚠️ {ph_status}")

        elif ph_value < 6.0:
            ph_status = "Acidic screening range"
            ph_advice = (
                "Check the pH needs of your intended crop. A soil-test-based "
                "recommendation can help determine whether correction is needed."
            )
            st.info(f"ℹ️ {ph_status}")

        elif ph_value <= 7.5:
            ph_status = "Generally suitable pH range for many crops"
            ph_advice = (
                "This pH range suits many crops, but crop requirements vary. "
                "Check your crop-specific recommendations and soil-test report."
            )
            st.success(f"✅ {ph_status}")

        elif ph_value <= 8.5:
            ph_status = "Alkaline screening range"
            ph_advice = (
                "Some nutrients may become less available to certain crops. "
                "Review the soil-test report and seek local agricultural advice."
            )
            st.warning(f"⚠️ {ph_status}")

        else:
            ph_status = "Highly alkaline screening range"
            ph_advice = (
                "High pH can restrict nutrient availability for some crops. "
                "Get a soil test and expert advice before attempting correction."
            )
            st.warning(f"⚠️ {ph_status}")

        st.write(ph_advice)

        # NPK readings: report values, do not invent nutrient status
        st.markdown("### 4. Nutrient readings")

        nutrient_data = [
            ("Nitrogen (N)", n),
            ("Phosphorus (P)", p),
            ("Potassium (K)", k),
        ]

        for nutrient_name, nutrient_value in nutrient_data:
            with st.container(border=True):
                left, right = st.columns([2, 1])

                with left:
                    st.markdown(f"**{nutrient_name}**")

                with right:
                    st.markdown(f"**{nutrient_value:g}**")

                st.caption(
                    "Compare this reading with the limits and units in "
                    "your soil-test report. A value alone cannot establish "
                    "whether this nutrient is deficient or excessive."
                )

        # Practical next steps
        st.markdown("### 5. Recommended next steps")

        st.markdown(
            """
            - **Check your soil-test report:** Confirm the units and the
              laboratory's nutrient interpretation.
            - **Consider your crop:** Nutrient and pH requirements differ
              between rice, wheat, pulses, vegetables and other crops.
            - **Avoid guesswork:** Do not apply fertilizer or lime based
              only on these numbers.
            - **Retest when needed:** Follow local agricultural guidance
              for soil sampling and testing.
            """
        )

        # Save once per analysis
        st.markdown("### 6. Save analysis")

        if st.session_state.get("soil_saved", False):
            st.success("This soil analysis has been saved to your history.")

        else:
            if st.button(
                "💾 Save Soil Analysis",
                type="primary",
                use_container_width=True,
                key="save_soil_analysis",
            ):
                try:
                    save_analysis(
                        "Soil Health",
                        f"Soil pH: {ph_value:.1f} | {ph_status}",
                        {
                            "N": n,
                            "P": p,
                            "K": k,
                            "ph": ph_value,
                            "pH_interpretation": ph_status,
                        },
                    )

                    st.session_state["soil_saved"] = True
                    st.success("Soil analysis saved successfully!")
                    st.rerun()

                except Exception as exc:
                    st.error(f"Could not save soil analysis: {exc}")

    st.caption(
        "This tool provides basic screening guidance. Accurate nutrient "
        "status and fertilizer requirements depend on soil-test methods, "
        "units, crop needs and local recommendations."
    )

# =========================================================
# 14. FARMER PROFILE
# =========================================================

elif selected_page == "Farmer Profile":

    show_page_heading(
        "👤 Farmer Profile",
        "Manage the name and village used by this app.",
    )

    with st.form("profile_form"):
        name = st.text_input(
            "Farmer name",
            value=st.session_state.farmer_name,
        )
        village = st.text_input(
            "Village / location",
            value=st.session_state.village,
        )

        save_profile = st.form_submit_button(
            "Save Profile",
            type="primary",
            use_container_width=True,
        )

    if save_profile:
        if not name.strip() or not village.strip():
            st.warning("Name and village cannot be empty.")
        else:
            old_name = st.session_state.farmer_name
            old_village = st.session_state.village

            st.session_state.farmer_name = name.strip()
            st.session_state.village = village.strip()

            if (
                old_name != st.session_state.farmer_name
                or old_village != st.session_state.village
            ):
                st.info(
                    "Your profile changed for this session. Older history "
                    "records remain linked to the name and village saved "
                    "with those records."
                )

            st.success("Profile updated.")
            st.rerun()

    st.markdown("---")
    st.write(
        "This demo profile is stored in the current Streamlit session. "
        "It is not password-protected authentication."
    )

# =========================================================
# 15. IRRIGATION ADVISORY
# =========================================================

elif selected_page == "Irrigation Advisory":

    show_page_heading(
        "💧 Irrigation Advisory",
        "Get basic watering guidance using your crop, soil moisture, "
        "and recent rainfall."
    )

    st.info(
        "This advisory uses the values you enter. It does not access "
        "live weather or directly measure your field."
    )

 
# Keep crop selection OUTSIDE the form
crop_choice = st.selectbox(
    "Select your crop",
    ["Rice", "Wheat", "Maize", "Other"],
    key="irrigation_crop_choice"
)

custom_crop = ""
if crop_choice == "Other":
    custom_crop = st.text_input(
        "Enter your crop name",
        placeholder="e.g. Tomato, Potato, Cotton",
        key="custom_irrigation_crop"
    ).strip()

# Keep the remaining irrigation inputs inside the form
with st.form("irrigation_form"):
    col1, col2 = st.columns(2)

    with col1:
        # Keep your existing moisture slider here
        pass


        moisture = st.slider(
                "Current soil moisture (%)",
                min_value=0,
                max_value=100,
                value=40,
                help=(
                    "Enter a measured percentage if available. "
                    "If estimated, treat the result as approximate."
                ),
            )

        with col2:
            rainfall = st.number_input(
                "Rainfall in the last 24 hours (mm)",
                min_value=0.0,
                max_value=500.0,
                value=0.0,
                step=1.0,
            )

            temperature = st.number_input(
                "Current temperature (°C)",
                min_value=-10.0,
                max_value=60.0,
                value=30.0,
                step=1.0,
            )

        submitted = st.form_submit_button(
            "Generate Irrigation Advice",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        if crop_choice == "Other" and not custom_crop:
            st.warning("Please enter the name of your crop.")

        else:
            crop_name = (
                custom_crop if crop_choice == "Other"
                else crop_choice
            )

            if rainfall >= 20:
                status = "Significant recent rainfall"
                priority = "Check soil before watering"
                advice = (
                    "Recent rainfall may have reduced the need for "
                    "additional water. Check moisture in the root zone "
                    "before deciding to irrigate."
                )

            elif moisture < 25:
                status = "Low soil moisture"
                priority = "Higher priority to inspect"
                advice = (
                    "The entered moisture value is low. Check the crop's "
                    "water requirements, growth stage, and root-zone "
                    "moisture. Irrigation may be needed if the soil is dry."
                )

            elif moisture < 50:
                status = "Moderate soil moisture"
                priority = "Monitor"
                advice = (
                    "Continue monitoring the soil and crop. Irrigate "
                    "according to crop-specific moisture requirements "
                    "and local agricultural guidance."
                )

            else:
                status = "Relatively high soil moisture"
                priority = "Check before watering"
                advice = (
                    "Check the root zone before adding more water. "
                    "Avoid unnecessary irrigation and waterlogging."
                )

            if temperature >= 35:
                temperature_note = (
                    "High temperature may increase water demand. "
                    "Check field moisture more frequently."
                )
            else:
                temperature_note = (
                    "Consider local weather, soil type, and crop growth "
                    "stage when deciding irrigation timing."
                )

            result_text = (
                f"Crop: {crop_name}. Status: {status}. "
                f"Priority: {priority}. {advice} {temperature_note}"
            )

            st.session_state["irrigation_result"] = {
                "crop": crop_name,
                "moisture": moisture,
                "rainfall": rainfall,
                "temperature": temperature,
                "status": status,
                "priority": priority,
                "advice": advice,
                "temperature_note": temperature_note,
                "result_text": result_text,
            }

            st.session_state["irrigation_saved"] = False

    irrigation_result = st.session_state.get("irrigation_result")

    if irrigation_result:

        st.markdown("---")
        st.subheader("Your Irrigation Assessment")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Selected Crop",
            irrigation_result["crop"],
        )

        m2.metric(
            "Soil Moisture",
            f'{irrigation_result["moisture"]}%',
        )

        m3.metric(
            "Recent Rainfall",
            f'{irrigation_result["rainfall"]:g} mm',
        )

        st.markdown("### Current Assessment")
        st.write(f'**Status:** {irrigation_result["status"]}')
        st.write(f'**Priority:** {irrigation_result["priority"]}')

        st.info(irrigation_result["advice"])
        st.caption(irrigation_result["temperature_note"])

        st.warning(
            "This is general guidance, not a scientifically calibrated "
            "irrigation schedule. Crop type, soil type, rooting depth, "
            "growth stage, and local conditions affect water needs."
        )

        if not st.session_state.get("irrigation_saved", False):

            if st.form_submit_button(
                "💾 Save Irrigation Analysis",
                type="primary",
                use_container_width=True,
            ):
                try:
                    save_analysis(
                        "Irrigation Advisory",
                        irrigation_result["result_text"],
                        {
                            "crop": irrigation_result["crop"],
                            "soil_moisture_percent":
                                irrigation_result["moisture"],
                            "rainfall_last_24h_mm":
                                irrigation_result["rainfall"],
                            "temperature_c":
                                irrigation_result["temperature"],
                            "status": irrigation_result["status"],
                            "priority": irrigation_result["priority"],
                        },
                    )

                    st.session_state["irrigation_saved"] = True
                    st.success(
                        "Irrigation analysis saved to history."
                    )
                    st.rerun()

                except Exception as exc:
                    st.error(f"Could not save analysis: {exc}")

        else:
            st.success("This irrigation assessment is saved.")


# =========================================================
# 16. ANALYSIS HISTORY
# =========================================================




# =========================================================
# 16. FOOTER
# =========================================================

st.markdown("---")
st.caption(
    "SmartAgriAI • Intelligent Agriculture • "
    "Use model outputs alongside soil tests and local agricultural advice."
)

