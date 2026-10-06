# Import os for file-path handling.
import os

# Import joblib for loading the trained package.
import joblib

# Import pandas for DataFrame operations.
import pandas as pd

# Import the feature preparation function.
from src.preprocessing import prepare_input

# Find project root.
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Build model path.
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "regression_model.joblib"
)

# Load the complete model package.
model_package = joblib.load(
    MODEL_PATH
)

# Extract preprocessing.
preprocessor = model_package["preprocessor"]

# Extract trained model.
model = model_package["model"]

# Extract feature list.
features = model_package["features"]

# Extract historical route frequencies.
route_frequency = model_package["route_frequency"]


# Create prediction function.
def predict_delay(input_data):

    # Prepare basic engineered features.
    df = prepare_input(input_data)

    # Get the route of the new flight.
    route = df["ROUTE"].iloc[0]

    # Look up the route frequency from training data.
    # If the route is unseen, use zero.
    df["ROUTE_FREQUENCY"] = route_frequency.get(
        route,
        0
    )

    # Select the exact training features.
    X = df[features]

    # Apply the saved preprocessing.
    X_processed = preprocessor.transform(X)

    # Generate prediction.
    prediction = model.predict(
        X_processed
    )[0]

    # Return prediction as a Python float.
    return float(prediction)