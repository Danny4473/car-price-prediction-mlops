from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "car_price_model.pkl"


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Car Price Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({
            "error": "Trained model not found. Train the model first."
        }), 500

    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not data:
        return jsonify({
            "error": "Please provide car features as JSON"
        }), 400

    try:
        model = joblib.load(MODEL_PATH)
        input_data = pd.DataFrame([data])

        prediction = model.predict(input_data)[0]

        return jsonify({
            "predicted_selling_price": float(prediction)
        })

    except Exception as error:
        return jsonify({
            "error": "Prediction failed",
            "details": str(error)
        }), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
