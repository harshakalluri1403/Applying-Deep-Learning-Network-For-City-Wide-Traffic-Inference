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
import tensorflow.keras as keras

from tensorflow.keras import layers

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import custom_object_scope 
# Constants
WEATHER_API_KEY = "d9812e87c02c43b5a9590308250703"
WEATHER_BASE_URL = "http://api.weatherapi.com/v1"
MODEL_DIR = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\lstm"
METRICS_DIR = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\lstm\\metrics"

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
@tf.keras.utils.register_keras_serializable(package='custom_losses')
def custom_loss(y_true, y_pred):
    """
    Custom weighted loss function for multi-output prediction
    
    Args:
        y_true (tensor): True values
        y_pred (tensor): Predicted values
    
    Returns:
        tensor: Weighted mean squared error
    """
    weights = tf.constant([0.4, 0.3, 0.2, 0.1], dtype=tf.float32)
    squared_errors = tf.square(y_true - y_pred)
    weighted_errors = tf.multiply(squared_errors, weights)
    return tf.reduce_mean(weighted_errors)
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
        
        # Define base features - these can be updated later
        self.feature_columns = ['interval', 'flow', 'occ', 'error', 'estimated_speed']
        # Additional features: distance, time_of_day, day_of_week
        self.additional_features = ['distance', 'time_of_day', 'day_of_week']
        
        # n_features will be updated when building the model
        self.n_outputs = 4  # speed, flow, occupancy, time



    def prepare_training_data(self):
        """Prepare sequences with enhanced feature engineering"""
        print("Preparing training data with enhanced features...")
        
        # Handle NaN values and outliers
        self.speed_data = self.speed_data.replace([np.inf, -np.inf], np.nan)
        self.speed_data = self.speed_data.ffill().bfill()
        
        # Add time-based features
        self.speed_data['hour_sin'] = np.sin(2 * np.pi * self.speed_data['timestamp'].dt.hour / 24)
        self.speed_data['hour_cos'] = np.cos(2 * np.pi * self.speed_data['timestamp'].dt.hour / 24)
        self.speed_data['day_sin'] = np.sin(2 * np.pi * self.speed_data['timestamp'].dt.dayofweek / 7)
        self.speed_data['day_cos'] = np.cos(2 * np.pi * self.speed_data['timestamp'].dt.dayofweek / 7)
        
        # Add traffic density feature (flow/speed ratio)
        self.speed_data['traffic_density'] = self.speed_data['flow'] / (self.speed_data['estimated_speed'] + 1)
        
        # Update feature columns
        self.feature_columns = ['interval', 'flow', 'occ', 'error', 'estimated_speed', 
                            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'traffic_density']
        
        # Update n_features
        self.n_features = len(self.feature_columns) + len(self.additional_features)
        
        # Remove outliers for speed and flow
        for col in ['estimated_speed', 'flow']:
            if col in self.speed_data.columns:
                Q1 = self.speed_data[col].quantile(0.25)
                Q3 = self.speed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.speed_data[col] = self.speed_data[col].clip(lower_bound, upper_bound)
        
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
                    
                    # IMPORTANT CHANGE: Use a normalized time value (0-1 range)
                    # Instead of i * 5, use a normalized time value
                    time_value = min(i / 1000, 1.0)  # Normalize to 0-1 range
                    
                    target = [
                        features[i + self.sequence_length, 4],  # speed
                        features[i + self.sequence_length, 1],  # flow
                        features[i + self.sequence_length, 2],  # occupancy
                        time_value  # normalized time offset
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
            
            # Create separate scalers for each target
            self.speed_scaler = MinMaxScaler()
            self.flow_scaler = MinMaxScaler()
            self.occ_scaler = MinMaxScaler()
            self.time_scaler = MinMaxScaler()
            
            # Scale each target separately
            y_speed = self.speed_scaler.fit_transform(y_array[:, 0].reshape(-1, 1))
            y_flow = self.flow_scaler.fit_transform(y_array[:, 1].reshape(-1, 1))
            y_occ = self.occ_scaler.fit_transform(y_array[:, 2].reshape(-1, 1))
            y_time = self.time_scaler.fit_transform(y_array[:, 3].reshape(-1, 1))
            
            # Combine scaled targets
            y_scaled = np.column_stack((y_speed, y_flow, y_occ, y_time))
            
            # Save the original scalers for later use
            self.scaler_y.fit(y_array)  # Keep this for backward compatibility
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )
            
            print(f"Training data shape: {X_train.shape}, Target data shape: {y_train.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            return None, None, None, None
    def check_model_compatibility(self):
        """Check if the current model is compatible with the feature set"""
        if self.model is None:
            return False
        
        expected_features = len(self.feature_columns) + len(self.additional_features)
        
        # Get the input shape from the model
        input_shape = self.model.input_shape
        if input_shape is None:
            return False
        
        # Check if the number of features matches
        model_features = input_shape[-1]
        
        if model_features != expected_features:
            print(f"Model incompatibility: model expects {model_features} features, but data has {expected_features} features")
            return False
        
        return True


    def update_feature_columns(self, new_columns):
        """Update feature columns and recalculate n_features"""
        self.feature_columns = new_columns
        self.n_features = len(self.feature_columns) + len(self.additional_features)
        
        # Check if model needs to be rebuilt
        if self.model is not None and not self.check_model_compatibility():
            print("Feature columns changed. Model needs to be rebuilt.")
            self.model = None
            self.build_model()
        
        print(f"Feature columns updated. Total features: {self.n_features}")
        return self.n_features

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
    def load_trained_model(self, filename=None):
        """
        Load a pre-trained model with custom loss function
        
        Args:
            filename (str, optional): Name of the model file. Defaults to most recent.
        
        Returns:
            bool: Whether model was successfully loaded
        """
        if not filename:
            model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras') or f.endswith('.h5')]
            if not model_files:
                print("No model files found.")
                return False
            
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)
            filename = model_files[0]

        model_path = os.path.join(MODEL_DIR, filename)
        # Add this line before loading the model
        self.load_model_config(filename)

        # First, try to load the scalers
        scaler_loaded = self.load_scalers(filename)
        if not scaler_loaded:
            print("Warning: Could not load scalers. Predictions may be inaccurate.")
        
        try:
            # Try different approaches to load the model
            try:
                # First try with custom loss
                from keras.utils import custom_object_scope
                with custom_object_scope({'custom_loss': custom_loss}):
                    self.model = keras.models.load_model(model_path)
            except:
                # If that fails, try standard MSE loss
                self.model = keras.models.load_model(model_path)
            
            print(f"Model loaded successfully from {model_path}")
            
            # Check if the model is compatible with current feature set
            if not self.check_model_compatibility():
                print("Warning: Loaded model is incompatible with current feature set.")
                print("You may need to rebuild the model or adjust your features.")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            
            # If all loading attempts fail, build a new model
            print("Could not load the existing model. You need to train a new model.")
            return False


    def build_model(self):
        """Build an improved LSTM model with correct input shape"""
        print("Building enhanced LSTM model...")
        tf.keras.backend.clear_session()
        
        try:
            # IMPORTANT: Update n_features based on the actual feature columns
            self.n_features = len(self.feature_columns) + len(self.additional_features)
            
            input_shape = (self.sequence_length, self.n_features)
            print(f"Building model with input shape: {input_shape}")
            
            # Input layer
            inputs = Input(shape=input_shape)
            
            # First LSTM block
            x = LSTM(128, return_sequences=True, 
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='orthogonal')(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # Second LSTM block
            x = LSTM(64, return_sequences=False,
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='orthogonal')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # Dense layers
            x = Dense(32, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            x = Dense(16, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            # Output layer
            outputs = Dense(self.n_outputs, activation='linear')(x)
            
            # Create model
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Use a learning rate schedule
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=10000,
                decay_rate=0.9)
            optimizer = Adam(learning_rate=lr_schedule)
            
            # Use Huber loss for robustness to outliers
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.Huber(),  # More robust to outliers than MSE
                metrics=['mae', 'mse']
            )
            
            model.summary()
            self.model = model
            return model
            
        except Exception as e:
            print(f"Error building model: {e}")
            return None

    def save_model_config(self, timestamp):
        """Save model configuration including feature columns"""
        config = {
            'feature_columns': self.feature_columns,
            'additional_features': self.additional_features,
            'sequence_length': self.sequence_length
        }
        config_path = os.path.join(MODEL_DIR, f"traffic_model_{timestamp}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)
        print(f"Model configuration saved to {config_path}")

    def save_scalers(self, timestamp):
        """Save scalers to disk"""
        import pickle
        scaler_path = os.path.join(MODEL_DIR, f"traffic_model_{timestamp}_scalers.pkl")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y
            }, f)
        print(f"Scalers saved to {scaler_path}")
        return scaler_path

    def load_scalers(self, model_filename):
        """Load scalers associated with a model"""
        import pickle
        # Extract timestamp from model filename
        timestamp = model_filename.replace("traffic_model_", "").replace(".keras", "").replace(".h5", "")
        scaler_path = os.path.join(MODEL_DIR, f"traffic_model_{timestamp}_scalers.pkl")
        
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scalers = pickle.load(f)
                    self.scaler_X = scalers['scaler_X']
                    self.scaler_y = scalers['scaler_y']
                print(f"Scalers loaded from {scaler_path}")
                return True
            except Exception as e:
                print(f"Error loading scalers: {e}")
                return False
        else:
            print(f"No scalers found at {scaler_path}")
            print("Attempting to prepare scalers from data...")
            return self.prepare_scalers()

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
    def calculate_metrics(self, y_test, y_pred):
        """Calculate metrics with proper inverse scaling"""
        # Separate the predictions by target
        y_speed_pred = y_pred[:, 0].reshape(-1, 1)
        y_flow_pred = y_pred[:, 1].reshape(-1, 1)
        y_occ_pred = y_pred[:, 2].reshape(-1, 1)
        y_time_pred = y_pred[:, 3].reshape(-1, 1)
        
        # Separate the actual values by target
        y_speed_test = y_test[:, 0].reshape(-1, 1)
        y_flow_test = y_test[:, 1].reshape(-1, 1)
        y_occ_test = y_test[:, 2].reshape(-1, 1)
        y_time_test = y_test[:, 3].reshape(-1, 1)
        
        # Inverse transform each target separately
        y_speed_pred_inv = self.speed_scaler.inverse_transform(y_speed_pred)
        y_flow_pred_inv = self.flow_scaler.inverse_transform(y_flow_pred)
        y_occ_pred_inv = self.occ_scaler.inverse_transform(y_occ_pred)
        y_time_pred_inv = self.time_scaler.inverse_transform(y_time_pred)
        
        y_speed_test_inv = self.speed_scaler.inverse_transform(y_speed_test)
        y_flow_test_inv = self.flow_scaler.inverse_transform(y_flow_test)
        y_occ_test_inv = self.occ_scaler.inverse_transform(y_occ_test)
        y_time_test_inv = self.time_scaler.inverse_transform(y_time_test)
        
        # Combine the inverse transformed values
        y_pred_inv = np.column_stack((
            y_speed_pred_inv, y_flow_pred_inv, y_occ_pred_inv, y_time_pred_inv
        ))
        
        y_test_inv = np.column_stack((
            y_speed_test_inv, y_flow_test_inv, y_occ_test_inv, y_time_test_inv
        ))
        
        # Calculate metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        
        # Calculate metrics for each target separately
        speed_mse = mean_squared_error(y_speed_test_inv, y_speed_pred_inv)
        speed_rmse = np.sqrt(speed_mse)
        speed_mae = mean_absolute_error(y_speed_test_inv, y_speed_pred_inv)
        
        flow_mse = mean_squared_error(y_flow_test_inv, y_flow_pred_inv)
        flow_rmse = np.sqrt(flow_mse)
        flow_mae = mean_absolute_error(y_flow_test_inv, y_flow_pred_inv)
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "speed_rmse": float(speed_rmse),
            "speed_mae": float(speed_mae),
            "flow_rmse": float(flow_rmse),
            "flow_mae": float(flow_mae)
        }
    def train_model(self, epochs=100, batch_size=32, patience=15):
        """Train the LSTM model with improved metrics calculation"""
        # First, prepare the training data to update feature columns
        X_train, X_test, y_train, y_test = self.prepare_training_data()
        
        # Now rebuild the model with the UPDATED feature count
        self.n_features = len(self.feature_columns) + len(self.additional_features)
        print(f"Rebuilding model with {self.n_features} features")
        self.model = None  # Clear the existing model
        self.build_model() 


        
        try:
            X_train, X_test, y_train, y_test = self.prepare_training_data()
            
            # Validate data
            if any(x is None for x in [X_train, X_test, y_train, y_test]):
                raise ValueError("Data preparation failed")
            
            # Generate timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"traffic_model_{timestamp}.h5"
            model_path = os.path.join(MODEL_DIR, model_filename)
            
            # Save scalers along with the model
            self.save_all_scalers(timestamp)
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
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
                verbose=1,
                shuffle=True  # Ensure data is shuffled
            )
            
            # Calculate metrics with proper scaling
            print("Calculating model performance metrics...")
            y_pred = self.model.predict(X_test)
            metrics = self.calculate_metrics(y_test, y_pred)
            
            # Prepare metrics for saving
            metrics_data = {
                "timestamp": timestamp,
                "epochs_trained": len(history.history['loss']),
                "max_epochs": epochs,
                "patience": patience,
                "final_train_loss": float(history.history['loss'][-1]),
                "final_val_loss": float(history.history['val_loss'][-1]),
                **metrics  # Include all metrics
            }
            
            # Save training results
            self._save_training_results(history, metrics_data, timestamp, 
                                    self.get_inverse_transformed(y_test), 
                                    self.get_inverse_transformed(y_pred))
            
            print(f"Model saved to {model_path}")
            # Add this line after saving the model
            self.save_model_config(timestamp)

            return history, metrics_data
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return None, None


    def save_all_scalers(self, timestamp):
        """Save all scalers to disk"""
        import pickle
        scaler_path = os.path.join(MODEL_DIR, f"traffic_model_{timestamp}_scalers.pkl")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'speed_scaler': self.speed_scaler,
                'flow_scaler': self.flow_scaler,
                'occ_scaler': self.occ_scaler,
                'time_scaler': self.time_scaler
            }, f)
        print(f"All scalers saved to {scaler_path}")
        return scaler_path

    def load_all_scalers(self, model_filename):
        """Load all scalers associated with a model"""
        import pickle
        # Extract timestamp from model filename
        timestamp = model_filename.replace("traffic_model_", "").replace(".keras", "").replace(".h5", "")
        scaler_path = os.path.join(MODEL_DIR, f"traffic_model_{timestamp}_scalers.pkl")
        
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scalers = pickle.load(f)
                    self.scaler_X = scalers['scaler_X']
                    self.scaler_y = scalers['scaler_y']
                    
                    # Load target-specific scalers if available
                    if 'speed_scaler' in scalers:
                        self.speed_scaler = scalers['speed_scaler']
                        self.flow_scaler = scalers['flow_scaler']
                        self.occ_scaler = scalers['occ_scaler']
                        self.time_scaler = scalers['time_scaler']
                    else:
                        # Create new target-specific scalers if not available
                        self.speed_scaler = MinMaxScaler()
                        self.flow_scaler = MinMaxScaler()
                        self.occ_scaler = MinMaxScaler()
                        self.time_scaler = MinMaxScaler()
                    
                print(f"Scalers loaded from {scaler_path}")
                return True
            except Exception as e:
                print(f"Error loading scalers: {e}")
                return False
        else:
            print(f"No scalers found at {scaler_path}")
            print("Attempting to prepare scalers from data...")
            return self.prepare_scalers()

    def get_inverse_transformed(self, y):
        """Get inverse transformed values for all targets"""
        # Separate the predictions by target
        y_speed = y[:, 0].reshape(-1, 1)
        y_flow = y[:, 1].reshape(-1, 1)
        y_occ = y[:, 2].reshape(-1, 1)
        y_time = y[:, 3].reshape(-1, 1)
        
        # Inverse transform each target separately
        y_speed_inv = self.speed_scaler.inverse_transform(y_speed)
        y_flow_inv = self.flow_scaler.inverse_transform(y_flow)
        y_occ_inv = self.occ_scaler.inverse_transform(y_occ)
        y_time_inv = self.time_scaler.inverse_transform(y_time)
        
        # Combine the inverse transformed values
        return np.column_stack((
            y_speed_inv, y_flow_inv, y_occ_inv, y_time_inv
        ))

    def load_model_config(self, model_filename):
        """Load model configuration including feature columns"""
        timestamp = model_filename.replace("traffic_model_", "").replace(".keras", "").replace(".h5", "")
        config_path = os.path.join(MODEL_DIR, f"traffic_model_{timestamp}_config.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.feature_columns = config['feature_columns']
                    self.additional_features = config['additional_features']
                    self.sequence_length = config['sequence_length']
                    self.n_features = len(self.feature_columns) + len(self.additional_features)
                print(f"Model configuration loaded from {config_path}")
                return True
            except Exception as e:
                print(f"Error loading model configuration: {e}")
                return False
        else:
            print(f"No configuration found at {config_path}")
            print("Using default feature configuration")
            return False





    def predict_eta(self, start_detector, end_detector, current_time=None, local_weather=None):
        """Enhanced ETA prediction method with improved scaling"""
        # Check if check_model_compatibility method exists
        if hasattr(self, 'check_model_compatibility'):
            # Check model compatibility
            if not self.check_model_compatibility():
                print("Model is incompatible with current feature set. Rebuilding model...")
                self.model = None
                self.build_model()
        else:
            # If the method doesn't exist, force rebuild the model
            print("Rebuilding model to ensure compatibility...")
            self.n_features = len(self.feature_columns) + len(self.additional_features)
            self.model = None
            self.build_model()
        
        # Ensure scalers are ready
        if not hasattr(self, 'speed_scaler') or not hasattr(self.speed_scaler, 'scale_'):
            print("Preparing scalers for prediction...")
            if not self.prepare_scalers():
                return {"error": "Could not prepare scalers for prediction"}
        
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
                        # Prepare sequence data with enhanced features
                        recent_data = detector_data.iloc[-self.sequence_length:]
                        
                        # Add time-based features if they don't exist
                        if 'hour_sin' not in recent_data.columns:
                            recent_data['hour_sin'] = np.sin(2 * np.pi * recent_data['timestamp'].dt.hour / 24)
                            recent_data['hour_cos'] = np.cos(2 * np.pi * recent_data['timestamp'].dt.hour / 24)
                            recent_data['day_sin'] = np.sin(2 * np.pi * recent_data['timestamp'].dt.dayofweek / 7)
                            recent_data['day_cos'] = np.cos(2 * np.pi * recent_data['timestamp'].dt.dayofweek / 7)
                            recent_data['traffic_density'] = recent_data['flow'] / (recent_data['estimated_speed'] + 1)
                        
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
                        
                        # Use separate scalers for inverse transformation
                        speed_pred = self.speed_scaler.inverse_transform(prediction_scaled[0, 0].reshape(-1, 1))
                        flow_pred = self.flow_scaler.inverse_transform(prediction_scaled[0, 1].reshape(-1, 1))
                        occ_pred = self.occ_scaler.inverse_transform(prediction_scaled[0, 2].reshape(-1, 1))
                        time_pred = self.time_scaler.inverse_transform(prediction_scaled[0, 3].reshape(-1, 1))
                        
                        prediction = [
                            speed_pred[0, 0],
                            flow_pred[0, 0],
                            occ_pred[0, 0],
                            time_pred[0, 0]
                        ]
                        
                        predictions.append(prediction)

                
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
    def prepare_scalers(self):
        """
        Prepare scalers by fitting them on training data
        
        Returns:
            bool: True if scalers are successfully prepared, False otherwise
        """
        try:
            # Attempt to prepare scalers using training data
            X_train, _, y_train, _ = self.prepare_training_data()
            
            if X_train is not None and y_train is not None:
                # Reshape and fit X scaler
                n_samples, n_timesteps, n_features = X_train.shape
                X_reshaped = X_train.reshape(-1, n_features)
                self.scaler_X.fit(X_reshaped)
                
                # Fit y scaler
                self.scaler_y.fit(y_train)
                
                print("Scalers prepared successfully")
                return True
            
            print("Could not prepare scalers: insufficient training data")
            return False
        
        except Exception as e:
            print(f"Error preparing scalers: {e}")
            return False
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
                # Fix: Use the correct method to load the model
                success = system.load_trained_model()
                if not success:
                    print("Failed to load model.")
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
