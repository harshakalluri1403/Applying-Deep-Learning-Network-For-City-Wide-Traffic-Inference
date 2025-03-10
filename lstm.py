import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import datetime
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import json
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

# Constants
WEATHER_API_KEY = "d9812e87c02c43b5a9590308250703"
WEATHER_BASE_URL = "http://api.weatherapi.com/v1"
MODEL_DIR = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\lstm"
METRICS_DIR = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\lstm\\metrics"

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

class TrafficPredictionSystem:
    def __init__(self, speed_data_path, detector_distances_path):
        self.speed_data_path = speed_data_path
        self.detector_distances_path = detector_distances_path
        self.speed_data = None
        self.detector_distances = None
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.graph = None
        self.mst_graph = None
        self.graph_metrics = {}
        self.sequence_length = 12
        self.metrics = {}
        # Define base features
        self.feature_columns = ['interval', 'flow', 'occ', 'error', 'estimated_speed']
        # Additional features: distance, time_of_day, day_of_week
        self.additional_features = ['distance', 'time_of_day', 'day_of_week']
        self.n_features = len(self.feature_columns) + len(self.additional_features)  # Total features
        self.n_outputs = 4  # speed, flow, occupancy, time

    def load_trained_model(self, filename=None):
        if not filename:
            model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras') or f.endswith('.h5')]
            if not model_files:
                print("No model files found.")
                return False
        
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)
            filename = model_files[0]
    
        model_path = os.path.join(MODEL_DIR, filename)
    
        try:
            self.model = load_model(model_path, custom_objects={
                'masked_mse': self.build_model().loss
            })
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def prepare_training_data(self):
        """Prepare sequences for LSTM training with correct feature dimensionality"""
        print("Preparing training data...")
        
        # Handle NaN values in the speed data
        self.speed_data = self.speed_data.ffill().bfill()
        
        grouped = self.speed_data.sort_values(['detid', 'timestamp'])
        X, y = [], []
        
        try:
            for detector, group in grouped.groupby('detid'):
                # Get detector distances
                detector_distances = self.detector_distances[
                    (self.detector_distances['Detector1'] == detector) |
                    (self.detector_distances['Detector2'] == detector)
                ]
                avg_distance = detector_distances['Distance (meters)'].mean()
                if np.isnan(avg_distance):
                    avg_distance = detector_distances['Distance (meters)'].median()
                if np.isnan(avg_distance):
                    avg_distance = 1000
                
                # Normalize distance
                normalized_distance = avg_distance / 1000  # Convert to kilometers
                
                # Get base feature values
                features = group[self.feature_columns].values
                
                # Create sequences with sliding window
                for i in range(len(features) - self.sequence_length):
                    sequence = features[i:i + self.sequence_length]
                    
                    # Calculate time features
                    timestamp = group.iloc[i]['timestamp']
                    time_of_day = timestamp.hour / 24.0
                    day_of_week = timestamp.weekday() / 7.0
                    
                    # Create sequence with all features
                    sequence_with_features = np.column_stack((
                        sequence,  # Base features
                        np.full((self.sequence_length, 1), normalized_distance),  # Distance
                        np.full((self.sequence_length, 1), time_of_day),  # Time of day
                        np.full((self.sequence_length, 1), day_of_week)  # Day of week
                    ))
                    
                    target = [
                        features[i + self.sequence_length, 4],  # speed
                        features[i + self.sequence_length, 1],  # flow
                        features[i + self.sequence_length, 2],  # occupancy
                        i * 5  # time offset
                    ]
                    
                    X.append(sequence_with_features)
                    y.append(target)
            
            if not X or not y:
                raise ValueError("No valid sequences could be created from the data")
            
            X_array = np.array(X, dtype=np.float32)
            y_array = np.array(y, dtype=np.float32)
            
            # Scale features
            n_samples = X_array.shape[0]
            n_timesteps = X_array.shape[1]
            n_features = X_array.shape[2]
            
            print(f"Input shape before scaling: {X_array.shape}")
            
            X_reshaped = X_array.reshape((n_samples * n_timesteps, n_features))
            X_scaled = self.scaler_X.fit_transform(X_reshaped)
            X_scaled = X_scaled.reshape((n_samples, n_timesteps, n_features))
            
            # Scale targets
            y_scaled = self.scaler_y.fit_transform(y_array)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )
            
            print(f"Training data shape: {X_train.shape}, Target data shape: {y_train.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            return None, None, None, None

    def build_model(self):
        """Build and compile the LSTM model with correct input shape"""
        print("Building LSTM model...")
        
        def custom_loss(y_true, y_pred):
            """Custom loss function with weighted MSE"""
            weights = tf.constant([0.4, 0.3, 0.2, 0.1], dtype=tf.float32)
            squared_errors = tf.square(y_true - y_pred)
            weighted_errors = tf.multiply(squared_errors, weights)
            return tf.reduce_mean(weighted_errors)
        
        # Clear any existing models
        tf.keras.backend.clear_session()
        
        try:
            input_shape = (self.sequence_length, self.n_features)
            print(f"Building model with input shape: {input_shape}")
            
            model = Sequential([
                Input(shape=input_shape),
                LSTM(128, return_sequences=True, 
                     kernel_initializer='he_normal',
                     recurrent_initializer='orthogonal'),
                BatchNormalization(),
                Dropout(0.3),
                
                LSTM(64, return_sequences=True,
                     kernel_initializer='he_normal',
                     recurrent_initializer='orthogonal'),
                BatchNormalization(),
                Dropout(0.3),
                
                LSTM(32,
                     kernel_initializer='he_normal',
                     recurrent_initializer='orthogonal'),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(16, activation='relu'),
                BatchNormalization(),
                Dropout(0.1),
                
                Dense(self.n_outputs, activation='linear')
            ])
            
            optimizer = Adam(learning_rate=0.001)
            
            model.compile(
                optimizer=optimizer,
                loss=custom_loss,
                metrics=['mae', 'mse']
            )
            
            model.summary()
            self.model = model
            return model
            
        except Exception as e:
            print(f"Error building model: {e}")
            return None

    def build_graph(self):
        """Build graph representations for path finding"""
        print("Building network graphs...")
        
        try:
            # Create regular graph
            self.graph = nx.Graph()
            
            # Add edges with weights from detector distances
            for _, row in self.detector_distances.iterrows():
                self.graph.add_edge(
                    row['Detector1'],
                    row['Detector2'],
                    weight=row['Distance (meters)']
                )
            
            # Create MST graph using Kruskal's algorithm
            self.mst_graph = nx.minimum_spanning_tree(self.graph)
            
            print(f"Network graphs built successfully:")
            print(f"- Main graph: {self.graph.number_of_nodes()} nodes, "
                  f"{self.graph.number_of_edges()} edges")
            print(f"- MST graph: {self.mst_graph.number_of_nodes()} nodes, "
                  f"{self.mst_graph.number_of_edges()} edges")
            
            # Validate graph connectivity
            if not nx.is_connected(self.graph):
                print("Warning: The main graph is not fully connected!")
                components = list(nx.connected_components(self.graph))
                print(f"Number of connected components: {len(components)}")
            
            # Calculate and store basic graph metrics
            self.graph_metrics = {
                'diameter': nx.diameter(self.graph),
                'average_shortest_path': nx.average_shortest_path_length(self.graph),
                'average_degree': sum(dict(self.graph.degree()).values()) / 
                                self.graph.number_of_nodes()
            }
            
            print("\nGraph metrics:")
            print(f"- Network diameter: {self.graph_metrics['diameter']:.2f} meters")
            print(f"- Average path length: "
                  f"{self.graph_metrics['average_shortest_path']:.2f} meters")
            print(f"- Average node degree: {self.graph_metrics['average_degree']:.2f}")
            
        except Exception as e:
            print(f"Error building graphs: {e}")
            raise

    def load_data(self):
        """Load and preprocess the speed and distance data"""
        print("Loading speed data...")
        self.speed_data = pd.read_excel(self.speed_data_path)
        
        # Clean speed data
        self.speed_data = self.speed_data.replace([np.inf, -np.inf], np.nan)
        self.speed_data = self.speed_data.ffill().bfill()
        
        # Convert day to datetime
        if not pd.api.types.is_datetime64_any_dtype(self.speed_data['day']):
            self.speed_data['day'] = pd.to_datetime(self.speed_data['day'])
        
        # Create timestamp column
        self.speed_data['timestamp'] = self.speed_data.apply(
            lambda row: row['day'] + pd.Timedelta(seconds=row['interval']), axis=1)
        
        print("Loading detector distances...")
        self.detector_distances = pd.read_excel(self.detector_distances_path)
        
        # Clean distance data
        self.detector_distances = self.detector_distances.replace([np.inf, -np.inf], np.nan)
        self.detector_distances = self.detector_distances.ffill().bfill()
        
        # Build graph for path finding
        self.build_graph()
        
        print("Data loaded and cleaned successfully.")
        
    def kruskal_shortest_path(self, start_detector, end_detector):
        """Find shortest path using Kruskal's algorithm"""
        def find(parent, i):
            if parent[i] == i:
                return i
            return find(parent, parent[i])

        def union(parent, rank, x, y):
            xroot = find(parent, x)
            yroot = find(parent, y)
            if rank[xroot] < rank[yroot]:
                parent[xroot] = yroot
            elif rank[xroot] > rank[yroot]:
                parent[yroot] = xroot
            else:
                parent[yroot] = xroot
                rank[xroot] += 1

        # Get all edges with weights
        edges = []
        for (u, v, d) in self.graph.edges(data=True):
            edges.append((u, v, d['weight']))
        
        # Sort edges by weight
        edges.sort(key=lambda x: x[2])
        
        vertices = list(self.graph.nodes())
        parent = {v: v for v in vertices}
        rank = {v: 0 for v in vertices}
        
        mst_edges = []
        for u, v, w in edges:
            if find(parent, u) != find(parent, v):
                mst_edges.append((u, v, w))
                union(parent, rank, u, v)
        
        # Create a new graph with MST edges
        mst = nx.Graph()
        for u, v, w in mst_edges:
            mst.add_edge(u, v, weight=w)
        
        try:
            path = nx.shortest_path(mst, start_detector, end_detector, weight='weight')
            distance = sum(mst[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            return path, distance
        except nx.NetworkXNoPath:
            return None, None

    
   
    

    def train_model(self, epochs=100, batch_size=32, patience=10):
        """Train the LSTM model with improved training process"""
        if self.model is None:
            self.build_model()
        
        try:
            X_train, X_test, y_train, y_test = self.prepare_training_data()
            
            # Validate data
            if any(x is None for x in [X_train, X_test, y_train, y_test]):
                raise ValueError("Data preparation failed")
            
            # Generate timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"traffic_model_{timestamp}.keras"
            model_path = os.path.join(MODEL_DIR, model_filename)
            
            # Enhanced callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001,
                    verbose=1
                )
            ]
            
            # Train model
            print(f"Training model with up to {epochs} epochs (patience={patience})...")
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # Calculate predictions and metrics
            y_pred_scaled = self.model.predict(X_test)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            y_actual = self.scaler_y.inverse_transform(y_test)
            
            # Calculate detailed metrics
            metrics = {
                "timestamp": timestamp,
                "epochs_trained": len(history.history['loss']),
                "max_epochs": epochs,
                "patience": patience,
                "final_train_loss": float(history.history['loss'][-1]),
                "final_val_loss": float(history.history['val_loss'][-1]),
                "model_metrics": {
                    "mse": float(mean_squared_error(y_actual, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_actual, y_pred))),
                    "mae": float(mean_absolute_error(y_actual, y_pred)),
                    "r2_score": float(r2_score(y_actual.flatten(), y_pred.flatten()))
                }
            }

            self.metrics = metrics
            
            # Save results
            self._save_training_results(history, metrics, timestamp, y_actual, y_pred)
            
            return history, metrics
            
        except Exception as e:
            print(f"Error during training: {e}")
            return None, None
    def _save_training_results(self, history, metrics, timestamp, y_actual, y_pred):
        """Save training metrics and enhanced visualizations"""
        # Save metrics to JSON
        metrics_filename = f"metrics_{timestamp}.json"
        metrics_path = os.path.join(METRICS_DIR, metrics_filename)
    
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    
        # Create figure with subplots
        plt.figure(figsize=(20, 15))
    
        # 1. Training History Plot
        plt.subplot(3, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
        # 2. Speed Prediction Scatter Plot
        plt.subplot(3, 2, 2)
        plt.scatter(y_actual[:, 0], y_pred[:, 0], alpha=0.5)
        plt.plot([y_actual[:, 0].min(), y_actual[:, 0].max()],
                [y_actual[:, 0].min(), y_actual[:, 0].max()],
                'r--', label='Perfect Prediction')
        plt.title('Predicted vs Actual Speed')
        plt.xlabel('Actual Speed (km/h)')
        plt.ylabel('Predicted Speed (km/h)')
        plt.legend()
        plt.grid(True)
    
        # 3. Flow Prediction Plot
        plt.subplot(3, 2, 3)
        plt.scatter(y_actual[:, 1], y_pred[:, 1], alpha=0.5)
        plt.plot([y_actual[:, 1].min(), y_actual[:, 1].max()],
                [y_actual[:, 1].min(), y_actual[:, 1].max()],
                'r--', label='Perfect Prediction')
        plt.title('Predicted vs Actual Flow')
        plt.xlabel('Actual Flow')
        plt.ylabel('Predicted Flow')
        plt.legend()
        plt.grid(True)
    
        # 4. ETA Prediction Plot
        plt.subplot(3, 2, 4)
        actual_eta = y_actual[:, 3]
        predicted_eta = y_pred[:, 3]
        plt.scatter(actual_eta, predicted_eta, alpha=0.5)
        plt.plot([actual_eta.min(), actual_eta.max()],
                [actual_eta.min(), actual_eta.max()],
                'r--', label='Perfect Prediction')
        plt.title('Predicted vs Actual ETA')
        plt.xlabel('Actual Time (minutes)')
        plt.ylabel('Predicted Time (minutes)')
        plt.legend()
        plt.grid(True)
    
        # 5. Error Distribution Plot
        plt.subplot(3, 2, 5)
        speed_errors = y_pred[:, 0] - y_actual[:, 0]
        sns.histplot(speed_errors, kde=True)
        plt.title('Speed Prediction Error Distribution')
        plt.xlabel('Prediction Error (km/h)')
        plt.ylabel('Frequency')
        plt.grid(True)
    
        # 6. Time Series Plot
        plt.subplot(3, 2, 6)
        sample_size = 100
        plt.plot(range(sample_size), y_actual[:sample_size, 0], label='Actual Speed', alpha=0.7)
        plt.plot(range(sample_size), y_pred[:sample_size, 0], label='Predicted Speed', alpha=0.7)
        plt.title('Speed Prediction Time Series (Sample)')
        plt.xlabel('Time Steps')
        plt.ylabel('Speed (km/h)')
        plt.legend()
        plt.grid(True)
    
        plt.tight_layout()
        plots_path = os.path.join(METRICS_DIR, f"training_analysis_{timestamp}.png")
        plt.savefig(plots_path)
        plt.close()
    
        # Save additional analysis plots
        self._save_additional_analysis(y_actual, y_pred, timestamp)
    def _save_additional_analysis(self, y_actual, y_pred, timestamp):
        """Save additional analysis plots"""
        # Create correlation matrix
        plt.figure(figsize=(12, 8))
        correlation_matrix = np.corrcoef([
            y_actual[:, 0],  # Actual Speed
            y_pred[:, 0],    # Predicted Speed
            y_actual[:, 1],  # Actual Flow
            y_pred[:, 1],    # Predicted Flow
            y_actual[:, 3],  # Actual Time
            y_pred[:, 3]     # Predicted Time
        ])
        
        labels = ['Actual Speed', 'Pred Speed', 'Actual Flow', 
                 'Pred Flow', 'Actual Time', 'Pred Time']
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   xticklabels=labels, 
                   yticklabels=labels)
        plt.title('Prediction Correlation Matrix')
        correlation_path = os.path.join(METRICS_DIR, f"correlation_matrix_{timestamp}.png")
        plt.savefig(correlation_path)
        plt.close()
    def find_shortest_path(self, start_detector, end_detector):
        """Find the shortest path between two detectors"""
        if not self.graph:
            raise ValueError("Graph not initialized. Call load_data() first.")
        
        if start_detector not in self.graph or end_detector not in self.graph:
            return None, None, "One or both detectors not found in the network."
        
        try:
            path = nx.shortest_path(self.graph, start_detector, end_detector, weight='weight')
        
            # Calculate total distance
            total_distance = 0
            path_edges = []
        
            for i in range(len(path) - 1):
                distance = self.graph[path[i]][path[i+1]]['weight']
                total_distance += distance
                path_edges.append((path[i], path[i+1], distance))
        
            return path, total_distance, path_edges
        
        except nx.NetworkXNoPath:
            return None, None, "No path exists between these detectors."
    def predict_eta(self, start_detector, end_detector, current_time=None, local_weather=None):
        """Enhanced ETA prediction with multiple path options"""
        if not current_time:
            current_time = datetime.datetime.now()
        
        # Get paths using both methods
        regular_path, regular_distance, regular_edges = self.find_shortest_path(start_detector, end_detector)
        kruskal_path, kruskal_distance = self.kruskal_shortest_path(start_detector, end_detector)
        
        if not regular_path and not kruskal_path:
            return {"error": "No path found between detectors"}
        
        # Get weather data
        weather_features = self.extract_weather_features(
            local_weather if local_weather else self.get_weather_data()
        )
        
        # Calculate predictions for both paths
        paths_info = []
        for path, distance, path_type in [
            (regular_path, regular_distance, "Regular"),
            (kruskal_path, kruskal_distance, "Kruskal")
        ]:
            if path:
                predictions = []
                for detector in path:
                    detector_data = self.speed_data[
                        self.speed_data['detid'] == detector
                    ].sort_values('timestamp')
                    
                    if len(detector_data) >= self.sequence_length:
                        # Prepare sequence data
                        recent_data = detector_data.iloc[-self.sequence_length:]
                        sequence = recent_data[self.feature_columns].values
                        
                        # Add time features
                        time_of_day = current_time.hour / 24.0
                        day_of_week = current_time.weekday() / 7.0
                        
                        sequence_with_features = np.column_stack((
                            sequence,
                            np.full((self.sequence_length, 1), distance/1000),
                            np.full((self.sequence_length, 1), time_of_day),
                            np.full((self.sequence_length, 1), day_of_week)
                        ))
                        
                        # Scale and predict
                        sequence_scaled = self.scaler_X.transform(sequence_with_features)
                        prediction_scaled = self.model.predict(
                            sequence_scaled.reshape(1, self.sequence_length, -1),
                            verbose=0
                        )
                        prediction = self.scaler_y.inverse_transform(prediction_scaled)
                        predictions.append(prediction[0])
                
                if predictions:
                    avg_prediction = np.mean(predictions, axis=0)
                    
                    # Apply adjustments
                    speed_adjustment = 1.0
                    if weather_features['is_rainy']:
                        speed_adjustment *= 0.8
                    if weather_features['wind_speed'] > 30:
                        speed_adjustment *= 0.9
                    
                    predicted_speed = max(1, avg_prediction[0] * speed_adjustment)
                    predicted_flow = avg_prediction[1]
                    predicted_occ = min(1, max(0, avg_prediction[2]))
                    
                    # Calculate ETA
                    hours = distance / 1000 / predicted_speed
                    eta_seconds = hours * 3600
                    arrival_time = current_time + datetime.timedelta(seconds=eta_seconds)
                    
                    paths_info.append({
                        "path_type": path_type,
                        "path": path,
                        "distance": distance,
                        "predicted_speed": predicted_speed,
                        "predicted_flow": predicted_flow,
                        "predicted_occupancy": predicted_occ,
                        "eta_minutes": eta_seconds / 60,
                        "arrival_time": arrival_time
                    })
        
        # Choose the best path based on ETA
        if paths_info:
            best_path = min(paths_info, key=lambda x: x["eta_minutes"])
            
            return {
                "start_detector": start_detector,
                "end_detector": end_detector,
                "best_path_type": best_path["path_type"],
                "path": best_path["path"],
                "total_distance_meters": best_path["distance"],
                "total_distance_km": best_path["distance"] / 1000,
                "predicted_speed_kmh": best_path["predicted_speed"],
                "predicted_flow": best_path["predicted_flow"],
                "predicted_occupancy": best_path["predicted_occupancy"],
                "eta_minutes": best_path["eta_minutes"],
                "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "estimated_arrival_time": best_path["arrival_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "weather_conditions": weather_features,
                "alternative_paths": paths_info
            }
        
        return {"error": "Could not calculate predictions"}
    def extract_weather_features(self, weather_data):
        """Extract relevant weather features from the API response"""
        if not weather_data:
            return {
                'temperature': 15.0,  # Default values if weather data is unavailable
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
    def get_weather_data(self, city="augsburg", days=1):
        """Fetch weather data from the WeatherAPI"""
        url = f"{WEATHER_BASE_URL}/forecast.json?key={WEATHER_API_KEY}&q={city}&days={days}&aqi=no"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Weather API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None
def main():
    """Enhanced main execution function with improved UI"""
    def print_header():
        print("\n" + "="*60)
        print("Traffic Prediction System".center(60))
        print("="*60 + "\n")

    def print_menu():
        print("\nMain Menu:")
        print("-" * 40)
        print("1. Train new model")
        print("2. Load existing model")
        print("3. Make prediction")
        print("4. View model metrics")
        print("5. Compare paths")
        print("6. View system status")
        print("7. Exit")
        print("-" * 40)

    def train_new_model(system):
        print("\nModel Training Configuration")
        print("-" * 40)
        try:
            # Get training parameters
            epochs = int(input("Enter maximum epochs (default 100): ") or "100")
            batch_size = int(input("Enter batch size (default 32): ") or "32")
            patience = int(input("Enter early stopping patience (default 10): ") or "10")
            
            print("\nStarting model training...")
            print("This may take several minutes. Please wait...")
            
            start_time = time.time()
            history, metrics = system.train_model(
                epochs=epochs,
                batch_size=batch_size,
                patience=patience
            )
            if metrics:
                training_time = time.time() - start_time
                print("\nTraining Summary:")
                print("-" * 40)
                print(f"Training time: {training_time/60:.2f} minutes")
                print(f"Epochs trained: {metrics['epochs_trained']}")
                print(f"Final training loss: {metrics['final_train_loss']:.6f}")
                print(f"Final validation loss: {metrics['final_val_loss']:.6f}")
    
                print("\nModel Performance Metrics:")
                print(f"RMSE: {metrics['model_metrics']['rmse']:.4f}")
                print(f"MAE: {metrics['model_metrics']['mae']:.4f}")
                print(f"R² Score: {metrics['model_metrics']['r2_score']:.4f}")
    
                print("\nTraining visualizations have been saved to:")
                print(f"- {METRICS_DIR}")
            else:
                print("\nTraining failed. Please check the error messages above.")
                
        except ValueError as e:
            print(f"\nError: Invalid input - {e}")
        except Exception as e:
            print(f"\nError during training: {e}")

    def load_model(system):
        print("\nLoad Existing Model")
        print("-" * 40)
        
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras') or f.endswith('.h5')]
        
        if not model_files:
            print("No existing models found. Please train a new model first.")
            return False
        
        print("\nAvailable models:")
        for i, model_file in enumerate(model_files, 1):
            modified_time = datetime.datetime.fromtimestamp(
                os.path.getmtime(os.path.join(MODEL_DIR, model_file))
            ).strftime("%Y-%m-%d %H:%M:%S")
            size_mb = os.path.getsize(os.path.join(MODEL_DIR, model_file)) / (1024*1024)
            print(f"{i}. {model_file}")
            print(f"   Last modified: {modified_time}")
            print(f"   Size: {size_mb:.2f} MB")
        
        try:
            model_choice = input("\nEnter model number (or press Enter for most recent): ")
            
            if model_choice.strip():
                idx = int(model_choice) - 1
                if 0 <= idx < len(model_files):
                    success = system.load_trained_model(model_files[idx])
                else:
                    print("Invalid model number.")
                    return False
            else:
                success = system.load_trained_model()
            
            if success:
                print("Model loaded successfully.")
                return True
            else:
                print("Failed to load model.")
                return False
            
        except ValueError:
            print("Invalid input. Please enter a number.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    

    def make_prediction(system):
        print("\nMake Prediction")
        print("-" * 40)
        
        if system.model is None:
            print("No model loaded. Please load or train a model first.")
            return
        
        try:
            start_detector = input("Enter starting detector ID: ")
            end_detector = input("Enter ending detector ID: ")
            
            print("\nCalculating predictions...")
            result = system.predict_eta(start_detector, end_detector)
            
            if "error" in result:
                print(f"\nError: {result['error']}")
            else:
                print("\nRoute Prediction Results:")
                print("-" * 40)
                print(f"Selected Path Type: {result['best_path_type']}")
                print(f"Path: {' -> '.join(result['path'])}")
                print(f"Total distance: {result['total_distance_km']:.2f} km")
                print(f"Predicted speed: {result['predicted_speed_kmh']:.2f} km/h")
                print(f"Predicted flow: {result['predicted_flow']:.2f}")
                print(f"Predicted occupancy: {result['predicted_occupancy']:.2f}")
                print(f"ETA: {result['eta_minutes']:.2f} minutes")
                
                print("\nTiming Information:")
                print(f"Current time: {result['current_time']}")
                print(f"Estimated arrival: {result['estimated_arrival_time']}")
                
                print("\nWeather Conditions:")
                print("-" * 30)
                weather = result['weather_conditions']
                print(f"Temperature: {weather['temperature']}°C")
                print(f"Precipitation: {weather['precipitation']} mm")
                print(f"Wind Speed: {weather['wind_speed']} kph")
                print(f"Humidity: {weather['humidity']}%")
                print(f"Rainy: {'Yes' if weather['is_rainy'] else 'No'}")
                
                print("\nAlternative Paths:")
                print("-" * 30)
                for path_info in result['alternative_paths']:
                    if path_info['path_type'] != result['best_path_type']:
                        print(f"\nPath Type: {path_info['path_type']}")
                        print(f"Distance: {path_info['distance']/1000:.2f} km")
                        print(f"ETA: {path_info['eta_minutes']:.2f} minutes")
        
        except Exception as e:
            print(f"Error making prediction: {e}")

    def view_metrics(system):
        print("\nView Model Metrics")
        print("-" * 40)
    
        metrics_files = [f for f in os.listdir(METRICS_DIR) if f.endswith('.json')]
    
        if not metrics_files:
            print("No metrics files found.")
            return
    
        print("\nAvailable metrics files:")
        for i, metrics_file in enumerate(metrics_files, 1):
            modified_time = datetime.datetime.fromtimestamp(
                os.path.getmtime(os.path.join(METRICS_DIR, metrics_file))
            ).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{i}. {metrics_file} (Created: {modified_time})")
    
        try:
            metrics_choice = input("\nEnter metrics file number (or press Enter for most recent): ")
        
            if metrics_choice.strip():
                idx = int(metrics_choice) - 1
                if 0 <= idx < len(metrics_files):
                    metrics_file = metrics_files[idx]
                else:
                    print("Invalid file number.")
                    return
            else:
                metrics_file = metrics_files[0]
        
            with open(os.path.join(METRICS_DIR, metrics_file), 'r') as f:
                metrics = json.load(f)
        
            print("\nMetrics Summary:")
            print("-" * 40)
            print(f"Training timestamp: {metrics['timestamp']}")
            print(f"Epochs trained: {metrics['epochs_trained']}")
            print(f"Final training loss: {metrics['final_train_loss']:.6f}")
            print(f"Final validation loss: {metrics['final_val_loss']:.6f}")
        
            print("\nModel Performance Metrics:")
            print(f"RMSE: {metrics['model_metrics']['rmse']:.4f}")
            print(f"MAE: {metrics['model_metrics']['mae']:.4f}")
            print(f"R² Score: {metrics['model_metrics']['r2_score']:.4f}")
        
            # Show associated plots
            plot_files = [
                f for f in os.listdir(METRICS_DIR)
                if f.startswith(f"training_analysis_{metrics['timestamp']}")
            ]
        
            if plot_files:
                print("\nAssociated visualization files:")
                for plot_file in plot_files:
                    print(f"- {os.path.join(METRICS_DIR, plot_file)}")
        
        except Exception as e:
            print(f"Error reading metrics: {e}")
            

    def view_system_status(system):
        print("\nSystem Status")
        print("-" * 40)
        
        # Data status
        print("Data Status:")
        if system.speed_data is not None:
            print(f"Speed data records: {len(system.speed_data):,}")
            print(f"Unique detectors: {system.speed_data['detid'].nunique()}")
            print(f"Date range: {system.speed_data['day'].min()} to {system.speed_data['day'].max()}")
        else:
            print("Speed data not loaded")
            
        if system.detector_distances is not None:
            print(f"Distance records: {len(system.detector_distances):,}")
        else:
            print("Distance data not loaded")
        
        # Model status
        print("\nModel Status:")
        if system.model is not None:
            print("Model is loaded")
            system.model.summary()
        else:
            print("No model loaded")
        
        # Directory status
        print("\nStorage Status:")
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras') or f.endswith('.h5')]
        metrics_files = [f for f in os.listdir(METRICS_DIR) if f.endswith('.json')]
        print(f"Saved models: {len(model_files)}")
        print(f"Metrics files: {len(metrics_files)}")
        
        # Memory usage
        import psutil
        process = psutil.Process(os.getpid())
        print(f"\nMemory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # Main execution
    print_header()
    
    # Default file paths
    default_speed_path = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\corrected_speed_data.xlsx"
    default_distance_path = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\detector_distances.xlsx"
    
    try:
        # Initialize system
        system = TrafficPredictionSystem(default_speed_path, default_distance_path)
        
        # Load data
        print("\nInitializing system and loading data...")
        system.load_data()
        
        while True:
            print_menu()
            choice = input("\nEnter your choice (1-7): ")
            
            if choice == '1':
                train_new_model(system)
            elif choice == '2':
                load_model(system)
            elif choice == '3':
                make_prediction(system)
            elif choice == '4':
                view_metrics(system)
            elif choice == '5':
                make_prediction(system)  # This now includes path comparison
            elif choice == '6':
                view_system_status(system)
            elif choice == '7':
                print("\nExiting program. Goodbye!")
                break
            else:
                print("\nInvalid choice. Please enter a number between 1 and 7.")
    
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

