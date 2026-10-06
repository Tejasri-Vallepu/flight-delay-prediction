import os
import sys

# Add project root to Python path
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, BASE_DIR)

import streamlit as st

# Import prediction function from our ML pipeline
from src.prediction import predict_delay


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -----------------------------
       Main application background
    ----------------------------- */

    .stApp {
        background: linear-gradient(
            135deg,
            #fff7fa 0%,
            #ffeef4 45%,
            #fff9fb 100%
        );
    }

    /* -----------------------------
       Hide Streamlit default menu
    ----------------------------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* -----------------------------
       Main content
    ----------------------------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* -----------------------------
       Hero section
    ----------------------------- */

    .hero {
        background: linear-gradient(
            135deg,
            #f8a8c4,
            #fbc2d4,
            #ffdce8
        );

        padding: 35px 40px;
        border-radius: 25px;

        box-shadow:
            0 10px 35px rgba(214, 93, 132, 0.18);

        margin-bottom: 25px;

        border: 1px solid #f7c5d6;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        color: #6d2140;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #7d4057;
        margin-bottom: 0;
    }

    /* -----------------------------
       Section headers
    ----------------------------- */

    .section-title {
        font-size: 23px;
        font-weight: 750;
        color: #6d2140;

        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* -----------------------------
       Cards
    ----------------------------- */

    .info-card {
        background: rgba(255, 255, 255, 0.90);

        border-radius: 18px;

        padding: 22px;

        border: 1px solid #f5cada;

        box-shadow:
            0 6px 20px rgba(214, 93, 132, 0.10);

        margin-bottom: 20px;
    }

    .info-title {
        font-size: 16px;
        font-weight: 700;
        color: #8b3154;
    }

    .info-value {
        font-size: 28px;
        font-weight: 800;
        color: #5f1d38;
    }

    /* -----------------------------
       Prediction result
    ----------------------------- */

    .prediction-card {
        background: linear-gradient(
            135deg,
            #fff0f5,
            #ffd8e6
        );

        border: 2px solid #f3a9c4;

        border-radius: 24px;

        padding: 35px;

        text-align: center;

        box-shadow:
            0 12px 35px rgba(214, 93, 132, 0.18);

        margin-top: 25px;
    }

    .prediction-icon {
        font-size: 48px;
    }

    .prediction-label {
        color: #7c3652;
        font-size: 18px;
        font-weight: 600;
    }

    .prediction-value {
        color: #691d3c;
        font-size: 42px;
        font-weight: 850;
        margin: 5px 0;
    }

    .prediction-note {
        color: #8a5267;
        font-size: 15px;
    }

    /* -----------------------------
       Footer
    ----------------------------- */

    .custom-footer {
        text-align: center;

        margin-top: 40px;

        padding: 20px;

        color: #8c5368;

        font-size: 14px;

        border-top: 1px solid #f3ccd9;
    }

    /* -----------------------------
       Input labels
    ----------------------------- */

    label {
        color: #6d2140 !important;
        font-weight: 600 !important;
    }

    /* -----------------------------
       Buttons
    ----------------------------- */

    .stButton > button {
        width: 100%;

        background: linear-gradient(
            135deg,
            #e987aa,
            #d96892
        );

        color: white;

        border: none;

        border-radius: 14px;

        padding: 14px 25px;

        font-size: 17px;

        font-weight: 700;

        box-shadow:
            0 6px 18px rgba(214, 93, 132, 0.25);

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(
            135deg,
            #d96892,
            #c85580
        );

        transform: translateY(-2px);

        box-shadow:
            0 9px 22px rgba(214, 93, 132, 0.30);
    }

    /* -----------------------------
       Metric styling
    ----------------------------- */

    [data-testid="stMetric"] {
        background: white;

        border: 1px solid #f4ccda;

        padding: 15px;

        border-radius: 16px;

        box-shadow:
            0 5px 15px rgba(214, 93, 132, 0.08);
    }

    /* -----------------------------
       Mobile adjustments
    ----------------------------- */

    @media (max-width: 768px) {

        .hero-title {
            font-size: 30px;
        }

        .hero {
            padding: 25px;
        }

        .prediction-value {
            font-size: 32px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            ✈️ Flight Delay Prediction
        </div>

        <div class="hero-subtitle">
            AI-powered flight departure delay prediction
            using machine learning.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TOP INFORMATION CARDS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        label="🤖 Model",
        value="ML Regression"
    )

with metric2:
    st.metric(
        label="📍 Prediction",
        value="Departure Delay"
    )

with metric3:
    st.metric(
        label="⏱️ Output",
        value="Minutes"
    )

with metric4:
    st.metric(
        label="⚡ Status",
        value="Ready"
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# FLIGHT INFORMATION SECTION
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🛫 Flight Information
    </div>
    """,
    unsafe_allow_html=True
)


# Create input card
with st.container(border=True):

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        st.markdown("### 🏷️ Airline")

        airline = st.text_input(
            "Airline Code",
            value="AA",
            max_chars=5,
            help="Example: AA, DL, UA"
        )

        origin = st.text_input(
            "Origin Airport",
            value="JFK",
            max_chars=5,
            help="Example: JFK"
        )

        destination = st.text_input(
            "Destination Airport",
            value="LAX",
            max_chars=5,
            help="Example: LAX"
        )


    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        st.markdown("### 📅 Date Information")

        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=6
        )

        day = st.number_input(
            "Day",
            min_value=1,
            max_value=31,
            value=15
        )

        day_of_week = st.number_input(
            "Day of Week",
            min_value=1,
            max_value=7,
            value=3,
            help="1 = Monday ... 7 = Sunday"
        )


    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        st.markdown("### ⏰ Schedule")

        scheduled_departure = st.number_input(
            "Scheduled Departure (HHMM)",
            min_value=0,
            max_value=2359,
            value=1200,
            help="Example: 1200 = 12:00 PM"
        )

        scheduled_arrival = st.number_input(
            "Scheduled Arrival (HHMM)",
            min_value=0,
            max_value=2359,
            value=1500,
            help="Example: 1500 = 3:00 PM"
        )

        scheduled_time = st.number_input(
            "Scheduled Time (minutes)",
            min_value=1,
            value=180
        )


# ============================================================
# FLIGHT DISTANCE
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

distance_col1, distance_col2 = st.columns([2, 1])

with distance_col1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                🛣️ Flight Distance
            </div>

            <div style="color:#8b5368;">
                Enter the scheduled flight distance.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with distance_col2:

    distance = st.number_input(
        "Distance",
        min_value=1,
        value=1000
    )


# ============================================================
# PREDICTION SECTION
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="section-title">
        🔮 Generate Prediction
    </div>
    """,
    unsafe_allow_html=True
)


# Center the prediction button
button_col1, button_col2, button_col3 = st.columns(
    [1, 2, 1]
)

with button_col2:

    predict_button = st.button(
        "✈️ Predict Flight Delay",
        use_container_width=True
    )


# ============================================================
# PREDICTION LOGIC
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Validate basic inputs
    # --------------------------------------------------------

    if not airline.strip():

        st.warning(
            "Please enter an airline code."
        )

        st.stop()


    if not origin.strip():

        st.warning(
            "Please enter the origin airport."
        )

        st.stop()


    if not destination.strip():

        st.warning(
            "Please enter the destination airport."
        )

        st.stop()


    # --------------------------------------------------------
    # Create input dictionary
    # --------------------------------------------------------

    input_data = {

        "AIRLINE": airline.strip().upper(),

        "ORIGIN_AIRPORT":
            origin.strip().upper(),

        "DESTINATION_AIRPORT":
            destination.strip().upper(),

        "MONTH":
            month,

        "DAY":
            day,

        "DAY_OF_WEEK":
            day_of_week,

        "SCHEDULED_DEPARTURE":
            scheduled_departure,

        "SCHEDULED_ARRIVAL":
            scheduled_arrival,

        "SCHEDULED_TIME":
            scheduled_time,

        "DISTANCE":
            distance
    }


    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    try:

        with st.spinner(
            "🤖 Analyzing flight information..."
        ):

            prediction = predict_delay(
                input_data
            )


        # Make sure result is not negative
        # for a delay prediction.
        prediction = max(
            0,
            float(prediction)
        )


        # ----------------------------------------------------
        # Prediction result
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-icon">
                    ✈️
                </div>

                <div class="prediction-label">
                    Estimated Departure Delay
                </div>

                <div class="prediction-value">
                    {prediction:.2f} minutes
                </div>

                <div class="prediction-note">
                    Predicted using the trained machine
                    learning regression model.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Result interpretation
        # ----------------------------------------------------

        st.markdown("<br>", unsafe_allow_html=True)

        result_col1, result_col2, result_col3 = st.columns(3)


        with result_col1:

            st.metric(
                "⏱️ Delay",
                f"{prediction:.2f} min"
            )


        with result_col2:

            st.metric(
                "🛫 Route",
                f"{origin.upper()} → {destination.upper()}"
            )


        with result_col3:

            st.metric(
                "🏷️ Airline",
                airline.upper()
            )


        # ----------------------------------------------------
        # Informational message
        # ----------------------------------------------------

        if prediction < 15:

            st.success(
                "🟢 The model predicts a relatively short "
                "departure delay."
            )

        elif prediction < 60:

            st.warning(
                "🟡 The model predicts a moderate "
                "departure delay."
            )

        else:

            st.error(
                "🔴 The model predicts a significant "
                "departure delay."
            )


    # --------------------------------------------------------
    # Error handling
    # --------------------------------------------------------

    except Exception as error:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(error)


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="section-title">
        💡 How This System Works
    </div>
    """,
    unsafe_allow_html=True
)


info1, info2, info3 = st.columns(3)


with info1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                01 · Flight Details
            </div>

            <p>
                Enter airline, airports, date,
                schedule and distance information.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with info2:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                02 · Feature Processing
            </div>

            <p>
                The application prepares the input
                using the same preprocessing pipeline
                used during model training.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with info3:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                03 · ML Prediction
            </div>

            <p>
                The trained regression model estimates
                the expected departure delay in minutes.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="custom-footer">

        ✈️ <b>Flight Delay Prediction & Analytics</b>

        <br>

        Machine Learning • Streamlit • Python

        <br><br>

        Built as an end-to-end ML prediction application.

    </div>
    """,
    unsafe_allow_html=True
)
