from flask import Flask, request, render_template
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ✅ Initialize the Flask app
app = Flask(__name__)

# ✅ Load models and scaler
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
ann_model = load_model("ann_model.keras")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_values = list(request.form.values())

        # Check input count
        if len(input_values) != 64:
            return render_template("index.html", prediction_text=f"Error: Expected 64 values, got {len(input_values)}.")

        input_features = [float(val) for val in input_values]

        # Scale and predict
        scaled = scaler.transform([input_features])
        xgb_pred = xgb_model.predict_proba(scaled)[:, 1]
        ann_pred = ann_model.predict(scaled).flatten()
        combined = (xgb_pred + ann_pred) / 2
        result = "Bankrupt" if combined[0] > 0.5 else "Not Bankrupt"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

# ✅ Run app (for Render)
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))  # Render uses env variable PORT
    app.run(debug=False, host='0.0.0.0', port=port)
