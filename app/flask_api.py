from flask import Flask, request, jsonify
import os
import sys

# Project root folder
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# src folder path
SRC_DIR = os.path.join(BASE_DIR, "src")

# Python ki src folder location cheptunnam
sys.path.insert(0, SRC_DIR)

# Ippudu prediction.py ni import cheyyachu
from src.prediction import predict_delay

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if data is None:
        return jsonify({
            "error": "JSON input is required"
        }), 400

    try:

        prediction = predict_delay(data)

        return jsonify({
            "predicted_departure_delay_minutes": round(prediction, 2)
        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )