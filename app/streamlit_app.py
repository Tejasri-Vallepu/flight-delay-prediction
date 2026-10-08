# Import Streamlit for creating the web interface.
import streamlit as st

# Import sys so that we can access the src folder.
import sys

# Add the src folder to Python's module search path.
sys.path.append("../src")

# Import our prediction function.
from src.prediction import predict_delay


# Configure the Streamlit page.
st.set_page_config(

    # Set browser page title.
    page_title="Flight Delay Prediction",

    # Use a wide dashboard layout.
    layout="wide"
)


# Display the application title.
st.title(
    "✈️ Flight Delay Prediction & Analytics"
)

# Display a short description.
st.write(
    "Predict estimated flight departure delay in minutes."
)


# Create two columns for cleaner UI.
col1, col2 = st.columns(2)


# Put airline input inside first column.
with col1:

    # Ask the user for airline code.
    airline = st.text_input(
        "Airline Code",
        value="AA"
    )

    # Ask for origin airport.
    origin = st.text_input(
        "Origin Airport",
        value="JFK"
    )

    # Ask for destination airport.
    destination = st.text_input(
        "Destination Airport",
        value="LAX"
    )

    # Ask for month.
    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=6
    )

    # Ask for day.
    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=15
    )


# Put remaining inputs inside second column.
with col2:

    # Ask for day of week.
    day_of_week = st.number_input(
        "Day of Week",
        min_value=1,
        max_value=7,
        value=3
    )

    # Ask for scheduled departure time in HHMM format.
    scheduled_departure = st.number_input(
        "Scheduled Departure (HHMM)",
        min_value=0,
        max_value=2359,
        value=1200
    )

    # Ask for scheduled arrival time.
    scheduled_arrival = st.number_input(
        "Scheduled Arrival (HHMM)",
        min_value=0,
        max_value=2359,
        value=1500
    )

    # Ask for scheduled flight duration.
    scheduled_time = st.number_input(
        "Scheduled Time (minutes)",
        min_value=1,
        value=180
    )

    # Ask for flight distance.
    distance = st.number_input(
        "Distance",
        min_value=1,
        value=1000
    )


# Create prediction button.
if st.button(
    "Predict Flight Delay"
):

    # Create dictionary containing user input.
    input_data = {

        # Airline code.
        "AIRLINE": airline,

        # Origin airport.
        "ORIGIN_AIRPORT": origin,

        # Destination airport.
        "DESTINATION_AIRPORT": destination,

        # Calendar information.
        "MONTH": month,
        "DAY": day,
        "DAY_OF_WEEK": day_of_week,

        # Scheduled flight information.
        "SCHEDULED_DEPARTURE":
            scheduled_departure,

        "SCHEDULED_ARRIVAL":
            scheduled_arrival,

        "SCHEDULED_TIME":
            scheduled_time,

        # Flight distance.
        "DISTANCE": distance
    }

    # Try generating the prediction.
    try:

        # Call the trained ML model.
        prediction = predict_delay(
            input_data
        )

        # Display the prediction.
        st.success(
            f"Predicted Departure Delay: "
            f"{prediction:.2f} minutes"
        )

    # Handle prediction errors.
    except Exception as error:

        # Display the error.
        st.error(
            f"Prediction failed: {error}"
        )