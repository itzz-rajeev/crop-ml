from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import requests
from collections import Counter

app = Flask(__name__)

# -------------------------------
# Load all models
# -------------------------------
model_dir = "models"
models = {}
for file in os.listdir(model_dir):
    if file.endswith(".pkl"):
        model_name = file.replace(".pkl", "").replace("_", " ").title()
        with open(os.path.join(model_dir, file), "rb") as f:
            models[model_name] = pickle.load(f)

# -------------------------------
# Weather API function (with geocoding)
# -------------------------------
API_KEY = "e05280c5f7a05a29383d8dcfc54c724b"  # Replace with your API key
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
GEO_URL = "http://api.openweathermap.org/geo/1.0/direct"

def get_weather(city):
    try:
        geo_url = f"{GEO_URL}?q={city},IN&limit=1&appid={API_KEY}"
        geo_response = requests.get(geo_url).json()

        if not geo_response:
            return None

        lat = geo_response[0]["lat"]
        lon = geo_response[0]["lon"]

        url = f"{BASE_URL}?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(url).json()

        if response.get("cod") != 200:
            return None

        return {
            "temperature": response["main"]["temp"],
            "humidity": response["main"]["humidity"],
            "rainfall": response.get("rain", {}).get("1h", 0.0)
        }
    except Exception as e:
        print("Weather API error:", e)
        return None

# -------------------------------
# Generate reasons why crop is not suitable
# -------------------------------
def generate_reason(crop, temperature, humidity, rainfall, ph):
    reasons = []

    crop = crop.lower()

    # Temperature-based reasoning
    if crop in ["wheat", "barley"] and temperature > 35:
        reasons.append("Too hot for optimal growth")
    if crop in ["rice"] and temperature < 20:
        reasons.append("Needs warmer temperature")
    if crop in ["maize", "cotton"] and temperature < 18:
        reasons.append("Temperature too low")

    # Rainfall-based reasoning
    if crop == "rice" and rainfall < 100:
        reasons.append("Needs much higher rainfall")
    if crop == "chickpea" and rainfall > 150:
        reasons.append("Excess rainfall may damage the crop")
    if crop == "cotton" and rainfall < 50:
        reasons.append("Requires more rainfall")

    # Humidity-based reasoning
    if crop in ["coffee", "banana"] and humidity < 60:
        reasons.append("Needs higher humidity")
    if crop in ["onion", "garlic"] and humidity > 70:
        reasons.append("Excess humidity may cause fungal diseases")

    # Soil pH reasoning
    if ph < 5.5 and crop in ["wheat", "barley", "maize"]:
        reasons.append("Soil is too acidic")
    if ph > 8 and crop in ["rice", "banana"]:
        reasons.append("Soil is too alkaline")

    if not reasons:
        return "Less suitable compared to the recommended crop"
    return ", ".join(reasons)

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html", models=list(models.keys()))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        city = request.form.get("city")
        weather = get_weather(city)

        if not weather:
            return render_template(
                "index.html",
                models=list(models.keys()),
                error_msg=f"❌ Error: Could not fetch weather for {city}",
            )

        N = float(request.form.get("N"))
        P = float(request.form.get("P"))
        K = float(request.form.get("K"))
        ph = float(request.form.get("ph"))

        temperature = weather["temperature"]
        humidity = weather["humidity"]
        rainfall = weather["rainfall"]

        features = [N, P, K, temperature, humidity, ph, rainfall]

        predictions = {}
        all_preds = []
        reasons = {}

        for name, model in models.items():
            pred = model.predict([features])[0]
            predictions[name] = pred
            all_preds.append(pred)

        final_crop = Counter(all_preds).most_common(1)[0][0]

        for model_name, crop in predictions.items():
            if crop != final_crop:
                reasons[crop] = generate_reason(crop, temperature, humidity, rainfall, ph)

        return render_template(
            "index.html",
            models=list(models.keys()),
            final_crop=final_crop,
            city=city,
            predictions=predictions,
            temperature=temperature,
            humidity=humidity,
            rainfall=rainfall,
            reasons=reasons
        )
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
