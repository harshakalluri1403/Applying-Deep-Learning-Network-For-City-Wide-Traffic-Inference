from flask import Flask, request, jsonify
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

# Required imports from your original code
from keras.models import load_model
from keras.utils import custom_object_scope
from sklearn.preprocessing import MinMaxScaler

# Constants
MODEL_DIR = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\lstm"
WEATHER_API_KEY = "d9812e87c02c43b5a9590308250703"
WEATHER_BASE_URL = "http://api.weatherapi.com/v1"

app = Flask(__name__)

class TrafficPredictionAPI:
    def __init__(self):
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.speed_scaler = MinMaxScaler()
        self.flow_scaler = MinMaxScaler()
        self.occ_scaler = MinMaxScaler()
        self.time_scaler = MinMaxScaler()
        self.sequence_length = 12
        self.feature_columns = ['interval', 'flow', 'occ', 'error', 'estimated_speed',
                              'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'traffic_density']
        self.additional_features = ['distance', 'time_of_day', 'day_of_week']
        self.n_features = len(self.feature_columns) + len(self.additional_features)

    @tf.keras.utils.register_keras_serializable(package='custom_losses')
    def custom_loss(self, y_true, y_pred):
        weights = tf.constant([0.4, 0.3, 0.2, 0.1], dtype=tf.float32)
        squared_errors = tf.square(y_true - y_pred)
        weighted_errors = tf.multiply(squared_errors, weights)
        return tf.reduce_mean(weighted_errors)

    def load_trained_model(self, filename=None):
        if not filename:
            model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras') or f.endswith('.h5')]
            if not model_files:
                return False
            filename = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)[0]

        model_path = os.path.join(MODEL_DIR, filename)
        try:
            with custom_object_scope({'custom_loss': self.custom_loss}):
                self.model = load_model(model_path)
            self.load_model_config(filename)
            self.load_all_scalers(filename)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def load_model_config(self, model_filename):
        import json
        timestamp = model_filename.replace("traffic_model_", "").replace(".keras", "").replace(".h5", "")
        config_path = os.path.join(MODEL_DIR, f"traffic_model_{timestamp}_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.feature_columns = config['feature_columns']
                self.additional_features = config['additional_features']
                self.sequence_length = config['sequence_length']
                self.n_features = len(self.feature_columns) + len(self.additional_features)
            return True
        return False

    def load_all_scalers(self, model_filename):
        import pickle
        timestamp = model_filename.replace("traffic_model_", "").replace(".keras", "").replace(".h5", "")
        scaler_path = os.path.join(MODEL_DIR, f"traffic_model_{timestamp}_scalers.pkl")
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.scaler_X = scalers['scaler_X']
                self.scaler_y = scalers['scaler_y']
                self.speed_scaler = scalers.get('speed_scaler', MinMaxScaler())
                self.flow_scaler = scalers.get('flow_scaler', MinMaxScaler())
                self.occ_scaler = scalers.get('occ_scaler', MinMaxScaler())
                self.time_scaler = scalers.get('time_scaler', MinMaxScaler())
            return True
        return False

    def extract_weather_features(self, weather_data):
        if not weather_data:
            return {
                'temperature': 15.0,
                'precipitation': 0.0,
                'wind_speed': 5.0,
                'humidity': 50.0,
                'is_rainy': 0
            }
        current = weather_data.get('current', {})
        return {
            'temperature': current.get('temp_c', 15.0),
            'precipitation': current.get('precip_mm', 0.0),
            'wind_speed': current.get('wind_kph', 5.0),
            'humidity': current.get('humidity', 50.0),
            'is_rainy': 1 if current.get('precip_mm', 0) > 0 else 0
        }

    def get_weather_data(self, city="augsburg"):
        import requests
        url = f"{WEATHER_BASE_URL}/forecast.json?key={WEATHER_API_KEY}&q={city}&days=1&aqi=no"
        try:
            response = requests.get(url)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

    def predict_eta(self, start_detector, end_detector, input_data):
        if not self.model:
            return {"error": "Model not loaded"}

        try:
            # Parse input data
            current_time = datetime.strptime(input_data['current_time'], '%Y-%m-%d %H:%M:%S')
            sequence_data = np.array(input_data['sequence_data'])  # Expected shape: (sequence_length, n_features)
            distance = float(input_data['distance'])

            # Prepare sequence with additional features
            time_of_day = current_time.hour / 24.0
            day_of_week = current_time.weekday() / 7.0
            
            sequence_with_features = np.column_stack((
                sequence_data,
                np.full((self.sequence_length, 1), distance/1000),
                np.full((self.sequence_length, 1), time_of_day),
                np.full((self.sequence_length, 1), day_of_week)
            ))

            # Scale and predict
            sequence_scaled = self.scaler_X.transform(sequence_with_features)
            prediction_scaled = self.model.predict(sequence_scaled.reshape(1, self.sequence_length, -1), verbose=0)

            # Inverse transform predictions
            speed_pred = self.speed_scaler.inverse_transform(prediction_scaled[0, 0].reshape(-1, 1))[0, 0]
            flow_pred = self.flow_scaler.inverse_transform(prediction_scaled[0, 1].reshape(-1, 1))[0, 0]
            occ_pred = self.occ_scaler.inverse_transform(prediction_scaled[0, 2].reshape(-1, 1))[0, 0]
            time_pred = self.time_scaler.inverse_transform(prediction_scaled[0, 3].reshape(-1, 1))[0, 0]

            # Weather adjustments
            weather_data = self.get_weather_data()
            weather_features = self.extract_weather_features(weather_data)
            speed_adjustment = 1.0
            if weather_features['is_rainy']:
                speed_adjustment *= 0.8
            if weather_features['wind_speed'] > 30:
                speed_adjustment *= 0.9

            predicted_speed = max(1, speed_pred * speed_adjustment)
            eta_seconds = (distance / 1000) / predicted_speed * 3600
            arrival_time = current_time + datetime.timedelta(seconds=eta_seconds)

            return {
                "start_detector": start_detector,
                "end_detector": end_detector,
                "predicted_speed_kmh": float(predicted_speed),
                "predicted_flow": float(flow_pred),
                "predicted_occupancy": float(occ_pred),
                "eta_minutes": float(eta_seconds / 60),
                "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "estimated_arrival_time": arrival_time.strftime("%Y-%m-%d %H:%M:%S"),
                "weather_conditions": weather_features
            }
        except Exception as e:
            return {"error": str(e)}

# Initialize the prediction system
traffic_system = TrafficPredictionAPI()

# API Routes
@app.route('/api/load_model', methods=['POST'])
def load_model():
    data = request.get_json()
    filename = data.get('filename')
    success = traffic_system.load_trained_model(filename)
    return jsonify({
        "status": "success" if success else "failed",
        "message": "Model loaded successfully" if success else "Failed to load model"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    start_detector = data.get('start_detector')
    end_detector = data.get('end_detector')
    input_data = data.get('input_data')
    
    if not all([start_detector, end_detector, input_data]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    result = traffic_system.predict_eta(start_detector, end_detector, input_data)
    return jsonify(result)

@app.route('/api/weather', methods=['GET'])
def get_weather():
    city = request.args.get('city', 'augsburg')
    weather_data = traffic_system.get_weather_data(city)
    if weather_data:
        return jsonify(traffic_system.extract_weather_features(weather_data))
    return jsonify({"error": "Failed to fetch weather data"}), 500

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "model_loaded": traffic_system.model is not None,
        "feature_count": traffic_system.n_features,
        "sequence_length": traffic_system.sequence_length
    })

if __name__ == '__main__':
    # Load model on startup (optional)
    traffic_system.load_trained_model()
    app.run(debug=True, host='0.0.0.0', port=5000)