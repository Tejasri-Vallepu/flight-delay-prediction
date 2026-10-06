# Import pandas for creating and manipulating DataFrames.
import pandas as pd

# Import numpy for mathematical transformations.
import numpy as np


# Define a function that prepares one flight record for prediction.
def prepare_input(data):
    """
    Convert user/API input into the same feature format
    used during model training.
    """

    # Convert the input dictionary into a one-row DataFrame.
    df = pd.DataFrame([data])

    # Convert categorical values to strings.
    df["AIRLINE"] = df["AIRLINE"].astype(str)

    # Convert origin airport to string.
    df["ORIGIN_AIRPORT"] = (
        df["ORIGIN_AIRPORT"].astype(str)
    )

    # Convert destination airport to string.
    df["DESTINATION_AIRPORT"] = (
        df["DESTINATION_AIRPORT"].astype(str)
    )

    # Extract departure hour from HHMM format.
    df["DEP_HOUR"] = (
        df["SCHEDULED_DEPARTURE"] // 100
    )

    # Extract departure minute.
    df["DEP_MINUTE"] = (
        df["SCHEDULED_DEPARTURE"] % 100
    )

    # Extract arrival hour.
    df["ARR_HOUR"] = (
        df["SCHEDULED_ARRIVAL"] // 100
    )

    # Extract arrival minute.
    df["ARR_MINUTE"] = (
        df["SCHEDULED_ARRIVAL"] % 100
    )

    # Convert midnight 24 to 0.
    df["ARR_HOUR"] = (
        df["ARR_HOUR"].replace(24, 0)
    )

    # Create cyclical departure hour features.
    df["DEP_HOUR_SIN"] = np.sin(
        2 * np.pi * df["DEP_HOUR"] / 24
    )

    # Create cyclical departure cosine feature.
    df["DEP_HOUR_COS"] = np.cos(
        2 * np.pi * df["DEP_HOUR"] / 24
    )

    # Create cyclical arrival sine feature.
    df["ARR_HOUR_SIN"] = np.sin(
        2 * np.pi * df["ARR_HOUR"] / 24
    )

    # Create cyclical arrival cosine feature.
    df["ARR_HOUR_COS"] = np.cos(
        2 * np.pi * df["ARR_HOUR"] / 24
    )

    # Create the route identifier.
    df["ROUTE"] = (
        df["ORIGIN_AIRPORT"]
        + "_"
        + df["DESTINATION_AIRPORT"]
    )

    # For a new prediction, use a neutral route-frequency value.
    # The production model can still process unseen routes.
    df["ROUTE_FREQUENCY"] = 0

    # Return the engineered DataFrame.
    return df