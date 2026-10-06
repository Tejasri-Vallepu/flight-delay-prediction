import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import joblib


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Flight Delay Classification",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# 2. PROJECT PATH
# ============================================================

# Get the main project folder.
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Location of the saved classification model.
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "classification_model.joblib"
)


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    # Load the complete model package.
    model_package = joblib.load(MODEL_PATH)

    return model_package


model_package = load_model()

# Extract components from the saved package.
preprocessor = model_package["preprocessor"]
model = model_package["model"]
feature_columns = model_package["features"]
model_name = model_package["model_name"]


# ============================================================
# 4. PAGE TITLE
# ============================================================

st.title("✈️ Flight Delay Prediction")

st.markdown(
    """
    Predict whether a flight is likely to be **delayed or on-time**
    using a machine learning classification model.
    """
)

st.divider()


# ============================================================
# 5. MODEL INFORMATION
# ============================================================

st.info(
    f"Classification Model Used: **{model_name}**"
)


# ============================================================
# 6. INPUT SECTION
# ============================================================

st.subheader("🛫 Flight Information")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# LEFT COLUMN
# ------------------------------------------------------------

with col1:

    airline = st.text_input(
        "Airline",
        value="AA",
        help="Example: AA, DL, UA"
    )

    origin = st.text_input(
        "Origin Airport",
        value="JFK",
        help="Example: JFK, LAX, ATL"
    )

    destination = st.text_input(
        "Destination Airport",
        value="LAX",
        help="Example: LAX, ORD, SFO"
    )

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
        value=1,
        help="1 = Monday, 7 = Sunday"
    )


# ------------------------------------------------------------
# RIGHT COLUMN
# ------------------------------------------------------------

with col2:

    scheduled_departure = st.number_input(
        "Scheduled Departure (HHMM)",
        min_value=0,
        max_value=2359,
        value=600,
        step=100,
        help="Example: 600 = 06:00, 1530 = 15:30"
    )

    scheduled_arrival = st.number_input(
        "Scheduled Arrival (HHMM)",
        min_value=0,
        max_value=2359,
        value=900,
        step=100,
        help="Example: 900 = 09:00, 1830 = 18:30"
    )

    scheduled_time = st.number_input(
        "Scheduled Flight Time (minutes)",
        min_value=1,
        max_value=1500,
        value=360
    )

    distance = st.number_input(
        "Distance (miles)",
        min_value=1,
        max_value=6000,
        value=2475
    )


st.divider()


# ============================================================
# 7. PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Flight Status",
    type="primary",
    use_container_width=True
)


# ============================================================
# 8. PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Validate text inputs
    # --------------------------------------------------------

    if not airline.strip():
        st.error("Please enter an airline code.")

    elif not origin.strip():
        st.error("Please enter the origin airport.")

    elif not destination.strip():
        st.error("Please enter the destination airport.")

    else:

        # ----------------------------------------------------
        # Convert text inputs to uppercase
        # ----------------------------------------------------

        airline = airline.strip().upper()
        origin = origin.strip().upper()
        destination = destination.strip().upper()


        # ----------------------------------------------------
        # Create engineered time features
        # ----------------------------------------------------

        dep_hour = scheduled_departure // 100
        dep_minute = scheduled_departure % 100

        arr_hour = scheduled_arrival // 100
        arr_minute = scheduled_arrival % 100

        # Dataset contains 24 as a possible hour value.
        # Convert it to 0 for cyclical encoding.
        if arr_hour == 24:
            arr_hour = 0


        # ----------------------------------------------------
        # Create cyclical features
        # ----------------------------------------------------

        dep_hour_sin = np.sin(
            2 * np.pi * dep_hour / 24
        )

        dep_hour_cos = np.cos(
            2 * np.pi * dep_hour / 24
        )

        arr_hour_sin = np.sin(
            2 * np.pi * arr_hour / 24
        )

        arr_hour_cos = np.cos(
            2 * np.pi * arr_hour / 24
        )


        # ----------------------------------------------------
        # Create route
        # ----------------------------------------------------

        route = (
            origin
            + "_"
            + destination
        )


        # ----------------------------------------------------
        # Create input DataFrame
        # ----------------------------------------------------

        input_data = {
            "MONTH": month,
            "DAY": day,
            "DAY_OF_WEEK": day_of_week,

            "AIRLINE": airline,

            "ORIGIN_AIRPORT": origin,
            "DESTINATION_AIRPORT": destination,

            "SCHEDULED_TIME": scheduled_time,
            "DISTANCE": distance,

            "DEP_HOUR": dep_hour,
            "DEP_MINUTE": dep_minute,

            "ARR_HOUR": arr_hour,
            "ARR_MINUTE": arr_minute,

            "DEP_HOUR_SIN": dep_hour_sin,
            "DEP_HOUR_COS": dep_hour_cos,

            "ARR_HOUR_SIN": arr_hour_sin,
            "ARR_HOUR_COS": arr_hour_cos,

            "ROUTE": route
        }


        input_df = pd.DataFrame(
            [input_data]
        )


        # ----------------------------------------------------
        # Make sure categorical columns are strings
        # ----------------------------------------------------

        categorical_features = [
            "AIRLINE",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "ROUTE"
        ]

        for column in categorical_features:

            input_df[column] = (
                input_df[column].astype(str)
            )


        # ----------------------------------------------------
        # Arrange features in exact training order
        # ----------------------------------------------------

        input_df = input_df[
            feature_columns
        ]


        # ----------------------------------------------------
        # Apply saved preprocessing
        # ----------------------------------------------------

        processed_input = (
            preprocessor.transform(
                input_df
            )
        )


        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = model.predict(
            processed_input
        )[0]


        # ----------------------------------------------------
        # Calculate probability
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(
                processed_input
            )[0][1]

        else:

            # LinearSVC does not provide predict_proba.
            # Convert decision score into a sigmoid-like
            # probability for display purposes.
            decision_score = model.decision_function(
                processed_input
            )[0]

            probability = (
                1 /
                (
                    1 +
                    np.exp(-decision_score)
                )
            )


        delay_probability = probability * 100
        ontime_probability = (
            1 - probability
        ) * 100


        # ====================================================
        # 9. DISPLAY RESULT
        # ====================================================

        st.divider()

        st.subheader("📊 Prediction Result")


        result_col1, result_col2 = st.columns(2)


        # ----------------------------------------------------
        # Prediction result
        # ----------------------------------------------------

        with result_col1:

            if prediction == 1:

                st.error(
                    "🔴 FLIGHT IS LIKELY TO BE DELAYED"
                )

            else:

                st.success(
                    "🟢 FLIGHT IS LIKELY TO BE ON-TIME"
                )


        # ----------------------------------------------------
        # Probability result
        # ----------------------------------------------------

        with result_col2:

            st.metric(
                "Probability of Delay",
                f"{delay_probability:.2f}%"
            )


        # ====================================================
        # 10. PROBABILITY DETAILS
        # ====================================================

        st.subheader("📈 Prediction Probability")

        probability_col1, probability_col2 = st.columns(2)

        with probability_col1:

            st.metric(
                "🔴 Delay Probability",
                f"{delay_probability:.2f}%"
            )

        with probability_col2:

            st.metric(
                "🟢 On-Time Probability",
                f"{ontime_probability:.2f}%"
            )


        # ----------------------------------------------------
        # Progress bar
        # ----------------------------------------------------

        st.write("Probability of Delay")

        st.progress(
            min(
                max(
                    delay_probability / 100,
                    0.0
                ),
                1.0
            )
        )


        # ====================================================
        # 11. FLIGHT SUMMARY
        # ====================================================

        st.subheader("✈️ Flight Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            st.write(
                f"**Airline:** {airline}"
            )

            st.write(
                f"**Origin:** {origin}"
            )

        with summary_col2:

            st.write(
                f"**Destination:** {destination}"
            )

            st.write(
                f"**Route:** {route}"
            )

        with summary_col3:

            st.write(
                f"**Distance:** {distance} miles"
            )

            st.write(
                f"**Scheduled Time:** {scheduled_time} minutes"
            )


        st.caption(
            "Prediction generated using the trained flight-delay "
            "classification model."
        )