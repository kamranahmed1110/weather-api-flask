from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Load your LSTM model (make sure to save it as a .h5 file after training)
model = tf.keras.models.load_model('temperature_prediction_model.h5')  # Replace with your actual model file path

# Load the dataset that was used for training the model (replace with the correct file path)
df = pd.read_csv('large_weather_data.csv')  # Replace with your actual dataset

# Separate the features (humidity and pressure) and target (temperature)
features = df[['Humidity', 'Pressure']].values
target = df[['Temperature']].values

# Initialize and fit the scalers
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

scaler_features.fit(features)
scaler_target.fit(target)

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Weather Prediction API! Use the /predict endpoint."

@app.route('/predict', methods=['GET'])
def predict():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    api_key = 'abb23ab1e5de1745ebc8ed05504b52f0'  # Your OpenWeather API key

    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude must be provided'}), 400

    # Create the API URL
    weather_api_url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}'

    # Make the request to the weather API
    response = requests.get(weather_api_url)
    if response.status_code != 200:
        return jsonify({'error': 'Failed to retrieve data from weather API'}), 500

    # Parse the JSON response
    weather_data = response.json()

    # Extract humidity and pressure from the API response
    if 'main' in weather_data:
        Humidity = weather_data['main']['humidity']
        Pressure = weather_data['main']['pressure']
        
    else:
        return jsonify({'error': 'Current weather data not found'}), 404

    # Prepare the input data for the model
    new_data = np.array([[Humidity, Pressure]])
    new_data_scaled = scaler_features.transform(new_data)

    # Reshape the data for LSTM input
    new_data_scaled_reshaped = new_data_scaled.reshape((1, new_data_scaled.shape[0], new_data_scaled.shape[1]))

    # Predict the temperature
    predicted_temp_scaled = model.predict(new_data_scaled_reshaped)
    predicted_temp = scaler_target.inverse_transform(predicted_temp_scaled)  # Inverse scale to get original temperature

    return jsonify({
    'predicted_temperature': float(predicted_temp[0][0]),  # Convert to native Python float
    'humidity': Humidity,
    'pressure': Pressure
})

if __name__ == '__main__':
    app.run(debug=True)
