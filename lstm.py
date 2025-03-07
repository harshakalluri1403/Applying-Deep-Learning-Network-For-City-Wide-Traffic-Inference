import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import datetime
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Weather API key
WEATHER_API_KEY = "d9812e87c02c43b5a9590308250703"
WEATHER_BASE_URL = "http://api.weatherapi.com/v1"

class TrafficPredictionSystem:
    def __init__(self, speed_data_path, detector_distances_path):
        self.speed_data_path = speed_data_path
        self.detector_distances_path = detector_distances_path
        self.speed_data = None
        self.detector_distances = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.graph = None
        self.sequence_length = 12  # 1 hour of data with 5-minute intervals
        
    def load_data(self):
        """Load and preprocess the speed and distance data"""
        print("Loading speed data...")
        self.speed_data = pd.read_excel(self.speed_data_path)
        
        # Convert day to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.speed_data['day']):
            self.speed_data['day'] = pd.to_datetime(self.speed_data['day'])
        
        # Create a timestamp column combining day and interval
        self.speed_data['timestamp'] = self.speed_data.apply(
            lambda row: row['day'] + pd.Timedelta(seconds=row['interval']), axis=1)
        
        print("Loading detector distances...")
        self.detector_distances = pd.read_excel(self.detector_distances_path)
        
        # Build graph for path finding
        self.build_graph()
        
        print("Data loaded successfully.")
        
    def build_graph(self):
        """Build a graph representation of the detector network for path finding"""
        self.graph = nx.Graph()
        
        # Add edges with distances
        for _, row in self.detector_distances.iterrows():
            self.graph.add_edge(
                row['Detector1'], 
                row['Detector2'], 
                weight=row['Distance (meters)']
            )
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        
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
    
    def prepare_training_data(self):
        """Prepare sequences for LSTM training"""
        print("Preparing training data...")
        
        # Group by detector and sort by timestamp
        grouped = self.speed_data.sort_values(['detid', 'timestamp'])
        
        X, y = [], []
        
        for detector, group in grouped.groupby('detid'):
            # Get speed values
            speeds = group['estimated_speed'].values
            
            # Create sequences
            for i in range(len(speeds) - self.sequence_length):
                X.append(speeds[i:i + self.sequence_length])
                y.append([
                    speeds[i + self.sequence_length],  # Speed prediction
                    i * 5  # Time offset (assuming 5-minute intervals)
                ])
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Normalize data
        X_scaled = np.zeros_like(X_array)
        for i in range(X_array.shape[0]):
            X_scaled[i] = self.scaler.fit_transform(X_array[i].reshape(-1, 1)).reshape(-1)
        
        y_scaled = self.scaler.fit_transform(y_array)
        
        # Reshape for LSTM [samples, time steps, features]
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y_scaled, test_size=0.2, random_state=42)
        
        print(f"Training data shape: {X_train.shape}, Target data shape: {y_train.shape}")
        return X_train, X_test, y_train, y_test
    
    def build_model(self):
        """Build and compile the LSTM model"""
        print("Building LSTM model...")
        
        # Using Input layer first to avoid the warning
        model = Sequential()
        model.add(Input(shape=(self.sequence_length, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(2))  # Output: [speed, time]
        
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, epochs=50, batch_size=32):
        """Train the LSTM model with early stopping"""
        if self.model is None:
            self.build_model()
            
        X_train, X_test, y_train, y_test = self.prepare_training_data()
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Model checkpoint callback - using .keras extension for TF 2.x
        checkpoint = ModelCheckpoint(
            'traffic_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, checkpoint]
        )
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('training_history.png')
        
        print("Model training completed.")
        return history
    
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
        """Predict ETA between two detectors considering weather conditions"""
        if not current_time:
            current_time = datetime.datetime.now()
        
        path, total_distance, path_details = self.find_shortest_path(start_detector, end_detector)
        
        if not path:
            return {
                "error": path_details
            }
        
        # Get weather data if not provided
        if not local_weather:
            weather_data = self.get_weather_data()
            weather_features = self.extract_weather_features(weather_data)
        else:
            weather_features = local_weather
        
        # Get recent speed data for detectors in the path
        recent_speeds = []
        for detector in path:
            detector_data = self.speed_data[self.speed_data['detid'] == detector]
            if not detector_data.empty:
                # Get the most recent speed
                detector_data = detector_data.sort_values('timestamp', ascending=False)
                recent_speeds.append(detector_data.iloc[0]['estimated_speed'])
        
        # If we have speed data for at least one detector
        if recent_speeds:
            avg_speed = sum(recent_speeds) / len(recent_speeds)
            
            # Adjust speed based on weather conditions
            if weather_features['is_rainy']:
                avg_speed *= 0.8  # Reduce speed in rainy conditions
            
            if weather_features['wind_speed'] > 30:
                avg_speed *= 0.9  # Reduce speed in windy conditions
                
            # Calculate ETA
            hours = total_distance / 1000 / avg_speed  # Convert to km
            eta_seconds = hours * 3600
            
            # Calculate arrival time
            arrival_time = current_time + datetime.timedelta(seconds=eta_seconds)
            
            # Calculate recommended speed to reach faster (slightly higher than average)
            recommended_speed = min(avg_speed * 1.1, 130)  # Cap at 130 km/h
            
            return {
                "start_detector": start_detector,
                "end_detector": end_detector,
                "path": path,
                "total_distance_meters": total_distance,
                "total_distance_km": total_distance / 1000,
                "average_speed_kmh": avg_speed,
                "recommended_speed_kmh": recommended_speed,
                "eta_seconds": eta_seconds,
                "eta_minutes": eta_seconds / 60,
                "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "estimated_arrival_time": arrival_time.strftime("%Y-%m-%d %H:%M:%S"),
                "weather_conditions": weather_features
            }
        else:
            # No recent speed data available, use a default speed
            default_speed = 60  # km/h
            hours = total_distance / 1000 / default_speed
            eta_seconds = hours * 3600
            arrival_time = current_time + datetime.timedelta(seconds=eta_seconds)
            
            return {
                "start_detector": start_detector,
                "end_detector": end_detector,
                "path": path,
                "total_distance_meters": total_distance,
                "total_distance_km": total_distance / 1000,
                "average_speed_kmh": default_speed,
                "recommended_speed_kmh": default_speed,
                "eta_seconds": eta_seconds,
                "eta_minutes": eta_seconds / 60,
                "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "estimated_arrival_time": arrival_time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "Using default speed due to lack of recent speed data",
                "weather_conditions": weather_features
            }
    
    def save_model(self, filename='traffic_lstm_model.keras'):
        """Save the trained model"""
        if self.model:
            self.model.save(filename)
            print(f"Model saved to {filename}")
        else:
            print("No model to save. Train a model first.")
    
    def load_trained_model(self, filename='traffic_lstm_model.keras'):
        """Load a previously trained model"""
        try:
            self.model = load_model(filename)
            print(f"Model loaded from {filename}")
        except Exception as e:
            print(f"Error loading model: {e}")


def main():
    # Create interactive CLI
    print("Traffic Prediction System")
    print("------------------------")
    
    # Get file paths from user (using default for now)
    speed_data_path = input("Enter path to speed data file (or press Enter for default): ")
    if not speed_data_path:
        speed_data_path = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\corrected_speed_data.xlsx"
        
    detector_distances_path = input("Enter path to detector distances file (or press Enter for default): ")
    if not detector_distances_path:
        detector_distances_path = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\detector_distances.xlsx"
    
    # Initialize system
    system = TrafficPredictionSystem(speed_data_path, detector_distances_path)
    
    try:
        # Load data
        system.load_data()
        
        # Check if model exists, otherwise train
        try:
            system.load_trained_model()
            print("Loaded existing model.")
        except:
            print("No existing model found. Training new model...")
            system.train_model(epochs=30)  # Reduced epochs for demonstration
            system.save_model()
        
        # Main interaction loop
        while True:
            print("\nOptions:")
            print("1. Find shortest path and predict ETA")
            print("2. Retrain model")
            print("3. Exit")
            
            choice = input("Enter your choice (1-3): ")
            
            if choice == '1':
                start_detector = input("Enter start detector ID: ")
                end_detector = input("Enter end detector ID: ")
                
                result = system.predict_eta(start_detector, end_detector)
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print("\nRoute Information:")
                    print(f"From {result['start_detector']} to {result['end_detector']}")
                    print(f"Path: {' -> '.join(result['path'])}")
                    print(f"Total distance: {result['total_distance_km']:.2f} km")
                    print(f"Average speed: {result['average_speed_kmh']:.2f} km/h")
                    print(f"Recommended speed: {result['recommended_speed_kmh']:.2f} km/h")
                    print(f"ETA: {result['eta_minutes']:.2f} minutes")
                    print(f"Current time: {result['current_time']}")
                    print(f"Estimated arrival time: {result['estimated_arrival_time']}")
                    
                    print("\nWeather Conditions:")
                    weather = result['weather_conditions']
                    print(f"Temperature: {weather['temperature']}°C")
                    print(f"Precipitation: {weather['precipitation']} mm")
                    print(f"Wind speed: {weather['wind_speed']} kph")
                    print(f"Humidity: {weather['humidity']}%")
                    print(f"Rainy: {'Yes' if weather['is_rainy'] else 'No'}")
            
            elif choice == '2':
                epochs = int(input("Enter number of epochs for training (default: 50): ") or "50")
                system.train_model(epochs=epochs)
                system.save_model()
            
            elif choice == '3':
                print("Exiting program.")
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
