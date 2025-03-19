import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import datetime
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import json
import warnings
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
WEATHER_API_KEY = "d9812e87c02c43b5a9590308250703"
WEATHER_BASE_URL = "http://api.weatherapi.com/v1"
MODEL_DIR = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\gat"
METRICS_DIR = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\gat\\metrics"
print(f"Model directory exists: {os.path.exists(MODEL_DIR)}")
print(f"Model directory is writable: {os.access(MODEL_DIR, os.W_OK)}")
print(f"Metrics directory exists: {os.path.exists(METRICS_DIR)}")
print(f"Metrics directory is writable: {os.access(METRICS_DIR, os.W_OK)}")


# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj=None):
        batch_size, num_nodes, _ = h.size()
        
        # Linear transformation
        Wh = torch.matmul(h, self.W)  # (batch_size, num_nodes, out_features)
        
        # Prepare attention mechanism input
        a_input = self._prepare_attentional_mechanism_input(Wh)
        
        # Compute attention coefficients
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        # Apply adjacency matrix if provided
        if adj is None:
            adj = torch.ones(batch_size, num_nodes, num_nodes, device=h.device)
        
        # Mask attention coefficients
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Normalize attention coefficients
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
            
    def _prepare_attentional_mechanism_input(self, Wh):
        batch_size, N, out_features = Wh.size()
        
        # Repeat features for all nodes
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # [batch, N, N, out_features]
        Wh2 = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # [batch, N, N, out_features]
        
        # Combine features for all possible node pairs
        combined = torch.cat([Wh1, Wh2], dim=3)  # [batch, N, N, 2*out_features]
        
        return combined
def train_gat_model(preprocessor):
    # Prepare training data
    X_train, X_test, y_train, y_test = preprocessor.prepare_training_data()
    
    # Get number of nodes from the preprocessor's graph
    num_nodes = preprocessor.graph.number_of_nodes()
    
    # Create model
    model = GATTrafficPredictionModel(
        nfeat=X_train.shape[2],  # Number of features
        nhid=64,                 # Hidden layer features
        nclass=y_train.shape[1], # Number of output classes
        num_nodes=num_nodes,     # Pass number of nodes
        dropout=0.6,
        nheads=8
    )
    
    # Create trainer
    trainer = GATTrainer(model, device)
    
    # Train model
    history, metrics = trainer.train_model(
        X_train, X_test, y_train, y_test,
        epochs=100,
        batch_size=32,
        patience=10
    )
    
    return model, history, metrics


class LearningRateScheduler:
    def __init__(self, optimizer, total_epochs):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_epochs, 
            eta_min=1e-6
        )

    def step(self):
        self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()
def quick_train(preprocessor, epochs=5, batch_size=256):
    """Quick training function for testing"""
    print("Preparing data for quick training...")
    
    # Prepare training data
    X_train, X_test, y_train, y_test = preprocessor.prepare_training_data()
    
    # Use complete dataset
    train_size = len(X_train)
    test_size = len(X_test)
    
    # Create simple model
    model = SimpleGATModel(
        input_dim=X_train.shape[2],
        hidden_dim=16,
        output_dim=y_train.shape[1],
        num_nodes=preprocessor.graph.number_of_nodes()
    )
    
    model.to(device)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Tracking variables
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    # Prediction storage for final analysis
    all_y_actual = []
    all_y_pred = []
    
    # Training loop
    print(f"Starting quick training for {epochs} epochs...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        batch_y_actual = []
        batch_y_pred = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
                
                # Store actual and predicted values
                batch_y_actual.append(batch_targets.cpu().numpy())
                batch_y_pred.append(outputs.cpu().numpy())
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        
        # Concatenate actual and predicted values
        epoch_y_actual = np.concatenate(batch_y_actual)
        epoch_y_pred = np.concatenate(batch_y_pred)
        
        # Store actual and predicted values
        all_y_actual.append(epoch_y_actual)
        all_y_pred.append(epoch_y_pred)
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['lr'].append(0.001)  # Constant learning rate
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")
    
    print("Quick training completed!")
    
    # Save the model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"quick_model_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Combine all actual and predicted values
    y_actual = np.concatenate(all_y_actual)
    y_pred = np.concatenate(all_y_pred)
    
    # Prepare metrics
    metrics = {
        "timestamp": timestamp,
        "train_loss": training_history['train_loss'],
        "val_loss": training_history['val_loss'],
        "lr": training_history['lr'],
        "final_train_loss": training_history['train_loss'][-1],
        "final_val_loss": training_history['val_loss'][-1],
        
        # Additional metrics calculation
        "speed_mse": float(mean_squared_error(y_actual[:, 0], y_pred[:, 0])),
        "flow_mse": float(mean_squared_error(y_actual[:, 1], y_pred[:, 1])),
        "occupancy_mse": float(mean_squared_error(y_actual[:, 2], y_pred[:, 2])),
        "eta_mse": float(mean_squared_error(y_actual[:, 3], y_pred[:, 3])),
        
        "speed_mae": float(mean_absolute_error(y_actual[:, 0], y_pred[:, 0])),
        "flow_mae": float(mean_absolute_error(y_actual[:, 1], y_pred[:, 1])),
        "occupancy_mae": float(mean_absolute_error(y_actual[:, 2], y_pred[:, 2])),
        "eta_mae": float(mean_absolute_error(y_actual[:, 3], y_pred[:, 3])),
        
        "speed_r2": float(r2_score(y_actual[:, 0], y_pred[:, 0])),
        "flow_r2": float(r2_score(y_actual[:, 1], y_pred[:, 1])),
        "occupancy_r2": float(r2_score(y_actual[:, 2], y_pred[:, 2])),
        "eta_r2": float(r2_score(y_actual[:, 3], y_pred[:, 3]))
    }
    
    # Save metrics
    try:
        # Ensure metrics directory exists
        os.makedirs(METRICS_DIR, exist_ok=True)
        
        # Save metrics to JSON
        metrics_filename = f"quick_metrics_{timestamp}.json"
        metrics_path = os.path.join(METRICS_DIR, metrics_filename)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
        
        print(f"Metrics saved to {metrics_path}")
        
        # Create comprehensive visualization
        plt.figure(figsize=(20, 15))

        # 1. Training History Plot
        plt.subplot(3, 2, 1)
        plt.plot(metrics['train_loss'], label='Training Loss')
        plt.plot(metrics['val_loss'], label='Validation Loss')
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
        plt.xlabel('Actual Speed')
        plt.ylabel('Predicted Speed')
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
        plt.xlabel('Actual Time')
        plt.ylabel('Predicted Time')
        plt.legend()
        plt.grid(True)
        
        # 5. Error Distribution Plot
        plt.subplot(3, 2, 5)
        speed_errors = y_pred[:, 0] - y_actual[:, 0]
        sns.histplot(speed_errors, kde=True)
        plt.title('Speed Prediction Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # 6. Time Series Plot
        plt.subplot(3, 2, 6)
        sample_size = min(100, len(y_actual))
        plt.plot(range(sample_size), y_actual[:sample_size, 0], label='Actual Speed', alpha=0.7)
        plt.plot(range(sample_size), y_pred[:sample_size, 0], label='Predicted Speed', alpha=0.7)
        plt.title('Speed Prediction Time Series (Sample)')
        plt.xlabel('Time Steps')
        plt.ylabel('Speed')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plots_path = os.path.join(METRICS_DIR, f"quick_training_analysis_{timestamp}.png")
        plt.savefig(plots_path)
        plt.close()

        print(f"Training plot saved to {plots_path}")
    except Exception as e:
        print(f"Error saving metrics and plots: {e}")
        import traceback
        traceback.print_exc()
    
    return model
class TrafficDataPreprocessor:
    def __init__(self, speed_data_path, detector_distances_path):
        self.speed_data_path = speed_data_path
        self.detector_distances_path = detector_distances_path
        
        # Data storage
        self.speed_data = None
        self.detector_distances = None
        
        # Scalers
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        # Graph representation
        self.graph = None
        self.mst_graph = None
        self.graph_metrics = {}
        
        # Sequence and feature parameters
        self.sequence_length = 12
        self.metrics = {}
        
        # Define base features
        self.feature_columns = ['interval', 'flow', 'occ', 'error', 'estimated_speed']
        
        # Additional features: distance, time_of_day, day_of_week
        self.additional_features = ['distance', 'time_of_day', 'day_of_week']
        
        # Total number of features
        self.n_features = len(self.feature_columns) + len(self.additional_features)
        
        # Number of output variables
        self.n_outputs = 4  # speed, flow, occupancy, time
    
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

    def build_graph(self):
        """Build optimized graph representations for path finding"""
        print("Building network graphs...")
        
        try:
            # Create regular graph
            self.graph = nx.Graph()
            
            # Get unique detector IDs
            unique_detectors = set()
            for _, row in self.detector_distances.iterrows():
                # Ensure detector IDs are strings
                det1 = str(row['Detector1'])
                det2 = str(row['Detector2'])
                unique_detectors.add(det1)
                unique_detectors.add(det2)
            
            # Create a mapping from detector ID to integer index
            self.detector_to_idx = {det: i for i, det in enumerate(sorted(unique_detectors))}
            self.idx_to_detector = {i: det for det, i in self.detector_to_idx.items()}
            
            # Add edges with weights from detector distances
            for _, row in self.detector_distances.iterrows():
                # Convert to string to ensure consistency
                det1 = str(row['Detector1'])
                det2 = str(row['Detector2'])
                self.graph.add_edge(
                    det1,
                    det2,
                    weight=row['Distance (meters)']
                )
            
            # Create MST graph using Kruskal's algorithm
            self.mst_graph = nx.minimum_spanning_tree(self.graph)
            
            print(f"Network graphs built successfully:")
            print(f"- Main graph: {self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges")
            print(f"- MST graph: {self.mst_graph.number_of_nodes()} nodes, "
                f"{self.mst_graph.number_of_edges()} edges")
            
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



    def prepare_training_data(self):
        """Prepare data for GAT training"""
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
            
            # Convert to PyTorch tensors
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
            
            print(f"Training data shape: {X_train.shape}, Target data shape: {y_train.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            return None, None, None, None

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
class SimpleGATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=4, dropout=0.3):
        super(SimpleGATModel, self).__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Simple attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Use only the last few timesteps for prediction
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Process each timestep
        features = self.feature_extractor(x.view(batch_size * seq_len, -1))
        features = features.view(batch_size, seq_len, -1)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(features).squeeze(-1), dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), features).squeeze(1)
        
        # Final prediction
        output = self.output_layer(context)
        
        return output
class GATTrainer:
    def __init__(self, model, device, preprocessor=None):
        self.model = model
        self.device = device
        self.preprocessor = preprocessor  # Store preprocessor reference
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        self.best_val_loss = float('inf')


    def create_optimizer(self, learning_rate=0.001, weight_decay=1e-5):
        """Create optimizer with weight decay"""
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        return optimizer

    def create_scheduler(self, optimizer, total_epochs):
        """Create learning rate scheduler"""
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_epochs, 
            eta_min=1e-6
        )
        return scheduler
    def train_model(
            self, 
            X_train, 
            X_test, 
            y_train, 
            y_test, 
            epochs=20, 
            batch_size=128, 
            patience=10,
            learning_rate=0.001,
            train_loader=None,
            test_loader=None,
            accumulation_steps=1,
            verbose=False,
            preprocessor=None  # Add preprocessor parameter
        ):
            """Train the GAT model with improved training process"""
            import datetime
            
            # Store preprocessor reference if provided
            if preprocessor is not None:
                self.preprocessor = preprocessor
            
            # Generate timestamp for scaler files
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Move model to device
            self.model.to(self.device)

            # Prepare data loaders if not provided
            if train_loader is None or test_loader is None:
                train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
                test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
                
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=batch_size, 
                    shuffle=True
                )
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, 
                    batch_size=batch_size, 
                    shuffle=False
                )

            # Optimizer and scheduler
            optimizer = self.create_optimizer(learning_rate)
            scheduler = self.create_scheduler(optimizer, epochs)

            # Loss function
            criterion = nn.MSELoss()

            # Early stopping variables
            patience_counter = 0
            best_model_path = os.path.join(MODEL_DIR, f"best_gat_model_{timestamp}.pth")

            # Training loop
            from tqdm import tqdm
            for epoch in tqdm(range(epochs)):
                # Training phase
                self.model.train()
                train_loss = 0.0
                optimizer.zero_grad()
                
                for batch_idx, (batch_features, batch_targets) in enumerate(train_loader):
                    # Move data to device
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    # Forward pass 
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    
                    # Gradient accumulation
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    # Step optimizer only after accumulation steps
                    if (batch_idx + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    train_loss += loss.item() * accumulation_steps
                
                # Ensure optimizer step for remaining batches
                if len(train_loader) % accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_features, batch_targets in test_loader:
                        batch_features = batch_features.to(self.device)
                        batch_targets = batch_targets.to(self.device)

                        outputs = self.model(batch_features)
                        loss = criterion(outputs, batch_targets)
                        val_loss += loss.item()

                # Calculate average losses
                train_loss /= len(train_loader)
                val_loss /= len(test_loader)

                # Learning rate scheduler step
                scheduler.step()

                # Store training history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['lr'].append(scheduler.get_last_lr()[0])

                # Verbose output
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}")
                    print(f"Train Loss: {train_loss:.4f}")
                    print(f"Val Loss: {val_loss:.4f}")
                    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

                # Early stopping and model saving
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), best_model_path)
                    if verbose:
                        print(f"Model saved to {best_model_path}")
                else:
                    patience_counter += 1

                # Check for early stopping
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Save the scalers if preprocessor exists
            if hasattr(self, 'preprocessor') and self.preprocessor is not None:
                scaler_path = os.path.join(MODEL_DIR, f"scalers_{timestamp}.pkl")
                scalers = {
                    'scaler_X': self.preprocessor.scaler_X,
                    'scaler_y': self.preprocessor.scaler_y
                }
                joblib.dump(scalers, scaler_path)
                print(f"Scalers saved to {scaler_path}")

            # Calculate final metrics
            try:
                print("Calculating final metrics...")
                metrics, y_actual, y_pred = self._calculate_metrics(X_test, y_test)
                print(f"Metrics calculated successfully: {metrics}")
                
                # Visualize training results
                print("Saving training results...")
                self._save_training_results(self.training_history, metrics, timestamp, y_actual, y_pred)
                print("Training results saved successfully")
            except Exception as e:
                print(f"Error in final metrics calculation or saving: {e}")
                import traceback
                traceback.print_exc()

            return self.training_history, metrics


    def _calculate_metrics(self, X_test, y_test, batch_size=1024):
        """Calculate detailed metrics and return predictions with batching for efficiency"""
        self.model.eval()
        
        # Create a DataLoader to process test data in batches
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        # Lists to store predictions and actual values
        all_y_pred = []
        all_y_test = []
        
        # Process in batches
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                # Move to device
                batch_features = batch_features.to(self.device)
                
                # Get predictions
                batch_pred = self.model(batch_features)
                
                # Store results
                all_y_pred.append(batch_pred.cpu().numpy())
                all_y_test.append(batch_targets.cpu().numpy())
        
        # Concatenate results
        y_pred = np.concatenate(all_y_pred)
        y_test_np = np.concatenate(all_y_test)
        
        # Calculate metrics
        metrics = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "final_train_loss": self.training_history['train_loss'][-1],
            "final_val_loss": self.training_history['val_loss'][-1],
            "mse": float(mean_squared_error(y_test_np, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test_np, y_pred))),
            "mae": float(mean_absolute_error(y_test_np, y_pred)),
            "r2_score": float(r2_score(y_test_np, y_pred))
        }
        
        return metrics, y_test_np, y_pred


    def _save_training_results(self, history, metrics, timestamp, y_actual, y_pred, max_points=5000):
        """Save training metrics and enhanced visualizations with limited data points"""
        try:
            # Ensure metrics directory exists
            os.makedirs(METRICS_DIR, exist_ok=True)
            
            # Prepare comprehensive metrics dictionary
            full_metrics = {
                "timestamp": timestamp,
                "train_loss": history['train_loss'],
                "val_loss": history['val_loss'],
                "learning_rate": history['lr'],
                "final_train_loss": history.get('train_loss', [])[-1] if history.get('train_loss') else None,
                "final_val_loss": history.get('val_loss', [])[-1] if history.get('val_loss') else None,
                "mse": metrics.get('mse'),
                "rmse": metrics.get('rmse'),
                "mae": metrics.get('mae'),
                "r2_score": metrics.get('r2_score'),
            }
            
            # Calculate additional metrics on a subset of data if needed
            if len(y_actual) > max_points:
                # Use random sampling for metrics calculation
                indices = np.random.choice(len(y_actual), max_points, replace=False)
                y_sample_actual = y_actual[indices]
                y_sample_pred = y_pred[indices]
                
                # Add additional metrics
                full_metrics.update({
                    "speed_mse": float(mean_squared_error(y_sample_actual[:, 0], y_sample_pred[:, 0])),
                    "flow_mse": float(mean_squared_error(y_sample_actual[:, 1], y_sample_pred[:, 1])),
                    "occupancy_mse": float(mean_squared_error(y_sample_actual[:, 2], y_sample_pred[:, 2])),
                    "eta_mse": float(mean_squared_error(y_sample_actual[:, 3], y_sample_pred[:, 3])),
                    
                    "speed_mae": float(mean_absolute_error(y_sample_actual[:, 0], y_sample_pred[:, 0])),
                    "flow_mae": float(mean_absolute_error(y_sample_actual[:, 1], y_sample_pred[:, 1])),
                    "occupancy_mae": float(mean_absolute_error(y_sample_actual[:, 2], y_sample_pred[:, 2])),
                    "eta_mae": float(mean_absolute_error(y_sample_actual[:, 3], y_sample_pred[:, 3])),
                    
                    "speed_r2": float(r2_score(y_sample_actual[:, 0], y_sample_pred[:, 0])),
                    "flow_r2": float(r2_score(y_sample_actual[:, 1], y_sample_pred[:, 1])),
                    "occupancy_r2": float(r2_score(y_sample_actual[:, 2], y_sample_pred[:, 2])),
                    "eta_r2": float(r2_score(y_sample_actual[:, 3], y_sample_pred[:, 3]))
                })
            else:
                # Use all data if it's small enough
                full_metrics.update({
                    "speed_mse": float(mean_squared_error(y_actual[:, 0], y_pred[:, 0])),
                    "flow_mse": float(mean_squared_error(y_actual[:, 1], y_pred[:, 1])),
                    "occupancy_mse": float(mean_squared_error(y_actual[:, 2], y_pred[:, 2])),
                    "eta_mse": float(mean_squared_error(y_actual[:, 3], y_pred[:, 3])),
                    
                    "speed_mae": float(mean_absolute_error(y_actual[:, 0], y_pred[:, 0])),
                    "flow_mae": float(mean_absolute_error(y_actual[:, 1], y_pred[:, 1])),
                    "occupancy_mae": float(mean_absolute_error(y_actual[:, 2], y_pred[:, 2])),
                    "eta_mae": float(mean_absolute_error(y_actual[:, 3], y_pred[:, 3])),
                    
                    "speed_r2": float(r2_score(y_actual[:, 0], y_pred[:, 0])),
                    "flow_r2": float(r2_score(y_actual[:, 1], y_pred[:, 1])),
                    "occupancy_r2": float(r2_score(y_actual[:, 2], y_pred[:, 2])),
                    "eta_r2": float(r2_score(y_actual[:, 3], y_pred[:, 3]))
                })
            
            # Save metrics to JSON
            metrics_filename = f"metrics_{timestamp}.json"
            metrics_path = os.path.join(METRICS_DIR, metrics_filename)
            
            print(f"Attempting to save metrics to: {metrics_path}")
            
            with open(metrics_path, 'w') as f:
                json.dump(full_metrics, f, indent=4, default=str)
            
            print(f"Successfully saved metrics to: {metrics_path}")

            # Sample data for visualization
            if len(y_actual) > max_points:
                indices = np.random.choice(len(y_actual), max_points, replace=False)
                y_sample_actual = y_actual[indices]
                y_sample_pred = y_pred[indices]
            else:
                y_sample_actual = y_actual
                y_sample_pred = y_pred

            # Create figure with subplots
            plt.figure(figsize=(20, 15))

            # 1. Training History Plot
            plt.subplot(3, 2, 1)
            plt.plot(history['train_loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Model Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            # 2. Speed Prediction Scatter Plot
            plt.subplot(3, 2, 2)
            plt.scatter(y_sample_actual[:, 0], y_sample_pred[:, 0], alpha=0.5)
            plt.plot([y_sample_actual[:, 0].min(), y_sample_actual[:, 0].max()],
                    [y_sample_actual[:, 0].min(), y_sample_actual[:, 0].max()],
                    'r--', label='Perfect Prediction')
            plt.title('Predicted vs Actual Speed')
            plt.xlabel('Actual Speed (km/h)')
            plt.ylabel('Predicted Speed (km/h)')
            plt.legend()
            plt.grid(True)
            
            # 3. Flow Prediction Plot
            plt.subplot(3, 2, 3)
            plt.scatter(y_sample_actual[:, 1], y_sample_pred[:, 1], alpha=0.5)
            plt.plot([y_sample_actual[:, 1].min(), y_sample_actual[:, 1].max()],
                    [y_sample_actual[:, 1].min(), y_sample_actual[:, 1].max()],
                    'r--', label='Perfect Prediction')
            plt.title('Predicted vs Actual Flow')
            plt.xlabel('Actual Flow')
            plt.ylabel('Predicted Flow')
            plt.legend()
            plt.grid(True)
            
            # 4. ETA Prediction Plot
            plt.subplot(3, 2, 4)
            actual_eta = y_sample_actual[:, 3]
            predicted_eta = y_sample_pred[:, 3]
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
            speed_errors = y_sample_pred[:, 0] - y_sample_actual[:, 0]
            sns.histplot(speed_errors, kde=True)
            plt.title('Speed Prediction Error Distribution')
            plt.xlabel('Prediction Error (km/h)')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            # 6. Time Series Plot
            plt.subplot(3, 2, 6)
            sample_size = min(100, len(y_sample_actual))
            plt.plot(range(sample_size), y_sample_actual[:sample_size, 0], label='Actual Speed', alpha=0.7)
            plt.plot(range(sample_size), y_sample_pred[:sample_size, 0], label='Predicted Speed', alpha=0.7)
            plt.title('Speed Prediction Time Series (Sample)')
            plt.xlabel('Time Steps')
            plt.ylabel('Speed (km/h)')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plots_path = os.path.join(METRICS_DIR, f"training_analysis_{timestamp}.png")
            plt.savefig(plots_path)
            plt.close()

            print(f"Successfully saved plots to: {plots_path}")

        except Exception as e:
            print(f"Error saving training results: {e}")
            import traceback
            traceback.print_exc()


# Example usage function
class GATTrafficPredictionModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_nodes, dropout=0.6, alpha=0.2, nheads=8):
        super(GATTrafficPredictionModel, self).__init__()
        self.dropout = dropout
        self.num_nodes = num_nodes
        
        # First layer with multiple attention heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True) 
            for _ in range(nheads)
        ])
        
        # Output layer (single attention head)
        self.out_att = GraphAttentionLayer(
            nhid * nheads,  # Input features are concatenated from all heads
            nclass, 
            dropout=dropout, 
            alpha=alpha, 
            concat=False
        )
        
        # Add a linear layer to handle the final transformation
        self.final_linear = nn.Linear(num_nodes * nclass, nclass)

    def forward(self, x):
        """
        x: input of shape (batch_size, seq_len, features)
        """
        batch_size = x.size(0)
        
        # Use only the last timestep for prediction
        x = x[:, -1, :]  # Shape: (batch_size, features)
        
        # Expand input to match graph structure
        x = x.unsqueeze(1).repeat(1, self.num_nodes, 1)
        
        # Create adjacency matrix
        adj = torch.ones(batch_size, self.num_nodes, self.num_nodes).to(x.device)
        
        # Apply attention layers (first layer with multiple heads)
        x_list = []
        for att in self.attentions:
            x_list.append(att(x, adj))
        
        # Concatenate outputs from all attention heads
        x = torch.cat(x_list, dim=-1)
        
        # Apply dropout
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Final attention layer
        x = self.out_att(x, adj)
        
        # Reshape and apply final linear layer
        x = x.view(batch_size, -1)
        x = self.final_linear(x)
        
        return x

def custom_loss(output, target):
    """
    Custom loss function similar to LSTM version
    Supports multiple objectives with weighted MSE
    """
    weights = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32).to(output.device)
    squared_errors = torch.square(output - target)
    weighted_errors = torch.multiply(squared_errors, weights)
    return torch.mean(weighted_errors)

class TrafficPredictor:
    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.scaler_X = preprocessor.scaler_X
        self.scaler_y = preprocessor.scaler_y
    def _extract_weather_features(self, weather_data):
        """
        Extract relevant weather features from the API response
        
        Args:
            weather_data (dict): Weather data from API
            
        Returns:
            dict: Extracted weather features
        """
        if not weather_data:
            return {
                'temperature': 15.0,  # Default values if weather data is unavailable
                'precipitation': 0.0,
                'wind_speed': 5.0,
                'humidity': 50.0,
                'is_rainy': 0,
                'description': 'Fair (Default)'
            }
        
        current = weather_data.get('current', {})
        
        # Extract basic weather features
        temperature = current.get('temp_c', 15.0)
        precipitation = current.get('precip_mm', 0.0)
        wind_speed = current.get('wind_kph', 5.0)
        humidity = current.get('humidity', 50.0)
        is_rainy = 1 if precipitation > 0 else 0
        
        # Get weather condition description
        condition = current.get('condition', {})
        description = condition.get('text', 'Unknown')
        
        return {
            'temperature': temperature,
            'precipitation': precipitation,
            'wind_speed': wind_speed,
            'humidity': humidity,
            'is_rainy': is_rainy,
            'description': description
        }

    def _get_weather_data(self):
        """Get current weather data"""
        try:
            return self.preprocessor.get_weather_data()
        except AttributeError:
            # If the preprocessor doesn't have the method, use a default implementation
            WEATHER_API_KEY = "d9812e87c02c43b5a9590308250703"
            WEATHER_BASE_URL = "http://api.weatherapi.com/v1"
            
            url = f"{WEATHER_BASE_URL}/forecast.json?key={WEATHER_API_KEY}&q=augsburg&days=1&aqi=no"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Weather API error: {response.status_code}")
                    return None
            except Exception as e:
                print(f"Error fetching data")

    def predict_eta(self, start_detector, end_detector, current_time=None, local_weather=None):
        """
        Predict traffic conditions between two detectors
        
        Args:
            start_detector (int): Starting detector ID
            end_detector (int): Ending detector ID
            current_time (datetime, optional): Current time
            local_weather (dict, optional): Local weather data
        
        Returns:
            dict: Prediction results
        """
        # Validate input detectors
        if start_detector not in self.preprocessor.speed_data['detid'].unique() or \
           end_detector not in self.preprocessor.speed_data['detid'].unique():
            return {"error": f"Invalid detectors: {start_detector} or {end_detector} not found"}
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Set current time if not provided
        if not current_time:
            current_time = datetime.datetime.now()
        
        # Get paths
        try:
            regular_path, regular_distance, regular_edges = self._find_shortest_path(start_detector, end_detector)
            kruskal_path, kruskal_distance = self.preprocessor.kruskal_shortest_path(start_detector, end_detector)
            
            if not regular_path and not kruskal_path:
                return {"error": "No path found between detectors"}
        
        except Exception as e:
            return {"error": f"Path finding error: {e}"}
        
        # Get weather data - enhanced with more details
        if local_weather:
            weather_features = local_weather
        else:
            weather_data = self._get_weather_data()
            weather_features = self._extract_weather_features(weather_data)
            
            # Add weather description for better user experience
            if weather_features['is_rainy']:
                if weather_features['precipitation'] > 10:
                    weather_features['description'] = "Heavy Rain"
                elif weather_features['precipitation'] > 5:
                    weather_features['description'] = "Moderate Rain"
                else:
                    weather_features['description'] = "Light Rain"
            elif weather_features['temperature'] > 30:
                weather_features['description'] = "Hot"
            elif weather_features['temperature'] < 5:
                weather_features['description'] = "Cold"
            elif weather_features['wind_speed'] > 30:
                weather_features['description'] = "Windy"
            else:
                weather_features['description'] = "Fair"
        
        # Calculate predictions for paths with enhanced weather consideration
        paths_info = []
        for path, distance, path_type in [
            (regular_path, regular_distance, "Regular"),
            (kruskal_path, kruskal_distance, "Kruskal")
        ]:
            if not path:
                continue
            
            try:
                # Prepare path features
                path_features = self._prepare_path_features(path, distance, current_time, weather_features)
                
                # Make prediction
                with torch.no_grad():
                    path_features = path_features.to(self.device)
                    predictions = self.model(path_features)
                    predictions = predictions.cpu().numpy()
                
                # Inverse transform predictions
                predictions = self.scaler_y.inverse_transform(predictions)
                
                # Apply weather adjustments with more sophisticated model
                predictions = self._apply_weather_adjustments(predictions, weather_features)
                
                # Calculate path metrics
                path_metrics = self._calculate_path_metrics(
                    path, 
                    predictions, 
                    distance, 
                    current_time,
                    weather_features
                )
                
                paths_info.append({
                    "path_type": path_type,
                    **path_metrics
                })
            
            except Exception as path_error:
                print(f"Error processing path {path_type}: {path_error}")
    
        
        
        # Choose best path
        if not paths_info:
            return {"error": "Could not calculate predictions for any path"}
        
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
    def _apply_weather_adjustments(self, predictions, weather_features):
        """Apply sophisticated weather adjustments to predictions"""
        # Base adjustment factors
        speed_adjustment = 1.0
        flow_adjustment = 1.0
        occupancy_adjustment = 1.0
        
        # Temperature effects
        temp = weather_features['temperature']
        if temp < 0:  # Cold weather
            speed_adjustment *= max(0.8, 1.0 - abs(temp) * 0.01)
        elif temp > 30:  # Hot weather
            speed_adjustment *= max(0.9, 1.0 - (temp - 30) * 0.005)
        
        # Precipitation effects
        precip = weather_features['precipitation']
        if precip > 0:  # Any precipitation
            # Non-linear effect: light rain has less impact than heavy rain
            rain_factor = min(0.3, precip * 0.1)  # Max 30% reduction for heavy rain
            speed_adjustment *= (1.0 - rain_factor)
            flow_adjustment *= (1.0 - rain_factor * 0.5)
            occupancy_adjustment *= (1.0 + rain_factor)
        
        # Wind effects
        wind = weather_features['wind_speed']
        if wind > 20:  # Strong wind
            wind_factor = min(0.2, (wind - 20) * 0.01)
            speed_adjustment *= (1.0 - wind_factor)
        
        # Humidity effects (high humidity can affect visibility)
        humidity = weather_features['humidity']
        if humidity > 85:  # High humidity
            humidity_factor = min(0.1, (humidity - 85) * 0.005)
            speed_adjustment *= (1.0 - humidity_factor)
        
        # Apply adjustments to predictions
        predictions[:, 0] *= speed_adjustment  # Adjust speed
        predictions[:, 1] *= flow_adjustment  # Adjust flow
        predictions[:, 2] *= occupancy_adjustment  # Adjust occupancy
        
        # Recalculate ETA based on adjusted speed if needed
        # This depends on how ETA is calculated in your model
        
        return predictions

    def _find_shortest_path(self, start_detector, end_detector):
        """Find the shortest path between two detectors"""
        if not self.preprocessor.graph:
            raise ValueError("Graph not initialized. Call load_data() first.")
        
        if start_detector not in self.preprocessor.graph or end_detector not in self.preprocessor.graph:
            return None, None, "One or both detectors not found in the network."
        
        try:
            path = nx.shortest_path(
                self.preprocessor.graph, 
                start_detector, 
                end_detector, 
                weight='weight'
            )
        
            # Calculate total distance
            total_distance = 0
            path_edges = []
        
            for i in range(len(path) - 1):
                distance = self.preprocessor.graph[path[i]][path[i+1]]['weight']
                total_distance += distance
                path_edges.append((path[i], path[i+1], distance))
        
            return path, total_distance, path_edges
        
        except nx.NetworkXNoPath:
            return None, None, "No path exists between these detectors."

    def _prepare_path_features(self, path, distance, current_time, weather_features=None):
        """Prepare features for a path of detectors"""
        path_features = []
        
        # Check how many features the scaler expects
        expected_features = self.scaler_X.n_features_in_
        
        for detector in path:
            try:
                # Get recent data for this detector
                detector_data = self.preprocessor.speed_data[
                    self.preprocessor.speed_data['detid'] == detector
                ].sort_values('timestamp')
                
                if len(detector_data) < self.preprocessor.sequence_length:
                    print(f"Warning: Not enough data for detector {detector}")
                    continue
                
                # Get the most recent sequence
                recent_data = detector_data.iloc[-self.preprocessor.sequence_length:]
                sequence = recent_data[self.preprocessor.feature_columns].values
                
                # Add contextual features
                time_of_day = current_time.hour / 24.0
                day_of_week = current_time.weekday() / 7.0
                normalized_distance = distance / 1000  # km
                
                # Create sequence with all features
                sequence_with_features = np.column_stack((
                    sequence,  # Base features
                    np.full((self.preprocessor.sequence_length, 1), normalized_distance),  # Distance
                    np.full((self.preprocessor.sequence_length, 1), time_of_day),  # Time of day
                    np.full((self.preprocessor.sequence_length, 1), day_of_week)  # Day of week
                ))
                
                # Check if we need to add weather features based on what the scaler expects
                if expected_features > sequence_with_features.shape[1] and weather_features is not None:
                    # Add weather features to the sequence
                    temp = weather_features.get('temperature', 15.0)
                    precip = weather_features.get('precipitation', 0.0)
                    wind = weather_features.get('wind_speed', 5.0)
                    is_rainy = weather_features.get('is_rainy', 0)
                    
                    weather_column = np.column_stack((
                        np.full((self.preprocessor.sequence_length, 1), temp / 40.0),  # Normalize temperature
                        np.full((self.preprocessor.sequence_length, 1), precip / 20.0),  # Normalize precipitation
                        np.full((self.preprocessor.sequence_length, 1), wind / 100.0),  # Normalize wind speed
                        np.full((self.preprocessor.sequence_length, 1), is_rainy)  # Binary rain indicator
                    ))
                    
                    sequence_with_features = np.column_stack((sequence_with_features, weather_column))
                elif expected_features < sequence_with_features.shape[1]:
                    # If scaler expects fewer features, trim the sequence
                    sequence_with_features = sequence_with_features[:, :expected_features]
                
                # Scale the features
                sequence_scaled = self.scaler_X.transform(sequence_with_features)
                path_features.append(sequence_scaled)
                
            except Exception as e:
                print(f"Error preparing features for detector {detector}: {e}")
        
        if not path_features:
            raise ValueError("Could not prepare features for any detector in the path")
        
        # Convert to tensor
        path_features_tensor = torch.tensor(np.array(path_features), dtype=torch.float32)
        return path_features_tensor



    def _apply_weather_adjustments(self, predictions, weather_features):
        """Adjust predictions based on weather conditions"""
        speed_adjustment = 1.0
        
        if weather_features['is_rainy']:
            speed_adjustment *= 0.8
        
        if weather_features['wind_speed'] > 30:
            speed_adjustment *= 0.9
        
        predictions[:, 0] *= speed_adjustment  # Adjust speed
        return predictions

    def _calculate_path_metrics(self, path, predictions, distance, current_time, weather_features=None):
        """Calculate path-level metrics with weather consideration"""
        # Average predictions
        avg_prediction = np.mean(predictions, axis=0)
        
        # Calculate ETA with weather consideration
        predicted_speed = max(1, avg_prediction[0])
        
        # Apply additional weather-based ETA adjustments if needed
        weather_eta_factor = 1.0
        if weather_features:
            if weather_features['is_rainy'] and weather_features['precipitation'] > 5:
                # Heavy rain adds extra delay
                weather_eta_factor += 0.15
            elif weather_features['wind_speed'] > 30:
                # Strong wind adds some delay
                weather_eta_factor += 0.1
        
        # Calculate ETA with weather factor
        hours = (distance / 1000 / predicted_speed) * weather_eta_factor
        eta_seconds = hours * 3600
        arrival_time = current_time + datetime.timedelta(seconds=eta_seconds)
        
        # Enhanced metrics
        metrics = {
            "path": path,
            "distance": distance,
            "predicted_speed": predicted_speed,
            "predicted_flow": avg_prediction[1],
            "predicted_occupancy": min(1, max(0, avg_prediction[2])),
            "eta_minutes": eta_seconds / 60,
            "arrival_time": arrival_time
        }
        
        # Add weather impact assessment
        if weather_features:
            # Calculate weather impact score (0-100%)
            impact_score = 0
            if weather_features['is_rainy']:
                impact_score += min(50, weather_features['precipitation'] * 5)
            if weather_features['wind_speed'] > 20:
                impact_score += min(30, (weather_features['wind_speed'] - 20) * 1.5)
            if weather_features['temperature'] < 0 or weather_features['temperature'] > 30:
                impact_score += min(20, abs(weather_features['temperature'] - 15) * 0.8)
            
            metrics["weather_impact_score"] = min(100, impact_score)
            metrics["weather_description"] = weather_features.get('description', 'Unknown')
        
        return metrics

# Example usage function
    def predict_traffic_path(preprocessor, model, start_detector, end_detector):
        """
        Convenience function to predict traffic path
        
        Args:
            preprocessor (TrafficDataPreprocessor): Data preprocessor
            model (GATTrafficPredictionModel): Trained GAT model
            start_detector (int): Starting detector ID
            end_detector (int): Ending detector ID
        
        Returns:
            dict: Prediction results
        """
        predictor = TrafficPredictor(model, preprocessor, device)
        return predictor.predict_eta(start_detector, end_detector)
def train_new_model(preprocessor):
    print("\nModel Training Configuration")
    print("-" * 40)
    try:
        # Get training parameters
        epochs = int(input("Enter maximum epochs (default 20): ") or "20")
        batch_size = int(input("Enter batch size (default 128): ") or "128")
        patience = int(input("Enter early stopping patience (default 10): ") or "10")
        
        print("\nStarting model training...")
        print("This may take several minutes. Please wait...")
        
        start_time = time.time()
        
        # Prepare training data
        X_train, X_test, y_train, y_test = preprocessor.prepare_training_data()
        
        if X_train is None or X_test is None or y_train is None or y_test is None:
            print("Failed to prepare training data. Check your dataset.")
            return None
        
        # Get number of nodes in the graph
        num_nodes = preprocessor.graph.number_of_nodes()
        
        # Create model with optimized architecture
        model = GATTrafficPredictionModel(
            nfeat=X_train.shape[2],  # Number of features
            nhid=64,                 # Hidden layer features
            nclass=y_train.shape[1], # Number of output classes
            dropout=0.3,             # Moderate dropout
            nheads=4,                # Reduced number of attention heads
            num_nodes=min(num_nodes, 20)  # Limit number of nodes
        )
        
        # Move model to GPU if available
        model = model.to(device)
        
        # Create trainer
        trainer = GATTrainer(model, device, preprocessor)
        
        # Optimize data loading with efficient parameters
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        
        # Use gradient accumulation for memory efficiency
        accumulation_steps = max(1, len(X_train) // (batch_size * 10000))
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing overhead
            pin_memory=True  # Enable pinned memory for faster GPU transfer
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Modify the trainer to use more efficient training
        history, metrics = trainer.train_model(
            X_train, X_test, y_train, y_test,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            train_loader=train_loader,
            test_loader=test_loader,
            accumulation_steps=accumulation_steps,
            verbose=True
        )
        
        # Generate timestamp for model and scaler files
        # Generate timestamp for model and scaler files
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if metrics and 'timestamp' in metrics:
            timestamp = metrics['timestamp']

        # Save the model
        model_path = os.path.join(MODEL_DIR, f"gat_model_{timestamp}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Final model saved to {model_path}")

        # Save the scalers
        scaler_path = os.path.join(MODEL_DIR, f"scalers_{timestamp}.pkl")
        scalers = {
            'scaler_X': preprocessor.scaler_X,
            'scaler_y': preprocessor.scaler_y
        }
        joblib.dump(scalers, scaler_path)
        print(f"Scalers saved to {scaler_path}")

        
        training_time = time.time() - start_time
        
        if metrics:
            print("\nTraining Summary:")
            print("-" * 40)
            print(f"Training timestamp: {timestamp}")
            print(f"Training time: {training_time:.2f} seconds")
            print(f"Final training loss: {metrics.get('final_train_loss', 'N/A'):.6f}")
            print(f"Final validation loss: {metrics.get('final_val_loss', 'N/A'):.6f}")

            print("\nPrediction Metrics:")
            print(f"RMSE: {metrics.get('rmse', 'N/A'):.2f}")
            print(f"MAE: {metrics.get('mae', 'N/A'):.2f}")
            print(f"R² Score: {metrics.get('r2_score', 'N/A'):.3f}")
            
            print("\nTraining visualizations have been saved to:")
            print(f"- {METRICS_DIR}")
            
            # Get predictions for visualization
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test.to(device)).cpu().numpy()
                y_actual = y_test.numpy()
            
            # Create comprehensive visualization
            plt.figure(figsize=(20, 15))

            # 1. Training History Plot
            plt.subplot(3, 2, 1)
            plt.plot(history['train_loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
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
            plt.xlabel('Actual Speed')
            plt.ylabel('Predicted Speed')
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
            plt.xlabel('Actual Time')
            plt.ylabel('Predicted Time')
            plt.legend()
            plt.grid(True)
            
            # 5. Error Distribution Plot
            plt.subplot(3, 2, 5)
            speed_errors = y_pred[:, 0] - y_actual[:, 0]
            sns.histplot(speed_errors, kde=True)
            plt.title('Speed Prediction Error Distribution')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            # 6. Time Series Plot
            plt.subplot(3, 2, 6)
            sample_size = min(100, len(y_actual))
            plt.plot(range(sample_size), y_actual[:sample_size, 0], label='Actual Speed', alpha=0.7)
            plt.plot(range(sample_size), y_pred[:sample_size, 0], label='Predicted Speed', alpha=0.7)
            plt.title('Speed Prediction Time Series (Sample)')
            plt.xlabel('Time Steps')
            plt.ylabel('Speed')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plots_path = os.path.join(METRICS_DIR, f"training_analysis_{timestamp}.png")
            plt.savefig(plots_path)
            plt.close()
            
            print(f"Training plot saved to {plots_path}")
        else:
            print("\nTraining failed. Please check the error messages above.")
        
        return model
    
    except ValueError as e:
        print(f"\nError: Invalid input - {e}")
        return None
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def make_prediction(preprocessor, model):
    """Make a traffic prediction with weather integration"""
    if model is None:
        print("\nNo model loaded. Please train or load a model first.")
        return
    
    try:
        print("\nTraffic Prediction")
        print("-" * 40)
        
        # Get detector IDs
        detectors = sorted(list(preprocessor.graph.nodes()))
        
        print("\nAvailable detectors:")
        for i, det in enumerate(detectors[:10]):
            print(f"{i+1}. {det}", end="\t")
            if (i+1) % 5 == 0:
                print()
        print("...")
        
        start_id = input("\nEnter starting detector ID: ")
        end_id = input("Enter destination detector ID: ")
        
        if start_id not in preprocessor.graph or end_id not in preprocessor.graph:
            print("Invalid detector IDs. Please try again.")
            return
        
        # Get weather data
        use_weather = input("\nInclude current weather data? (y/n): ").lower() == 'y'
        
        # Create predictor
        predictor = TrafficPredictor(model, preprocessor, device)
        
        # Make prediction
        print("\nCalculating prediction...")
        if use_weather:
            # Fetch current weather
            weather_data = predictor._get_weather_data()
            result = predictor.predict_eta(start_id, end_id, local_weather=None)  # Weather fetched inside
        else:
            # Use default weather
            result = predictor.predict_eta(start_id, end_id, local_weather={
                'temperature': 15.0,
                'precipitation': 0.0,
                'wind_speed': 5.0,
                'humidity': 50.0,
                'is_rainy': 0,
                'description': 'Fair (Default)'
            })
        
        # Display results with weather visualization
        display_prediction_with_weather(result)
        
    except Exception as e:
        print(f"\nError making prediction: {e}")
        import traceback
        traceback.print_exc()
def display_prediction_with_weather(result):
    """Display prediction results with weather visualization"""
    if "error" in result:
        print(f"\nError: {result['error']}")
        return
    
    # Display basic results
    print("\nPrediction Results:")
    print("-" * 40)
    print(f"Path: {' -> '.join(map(str, result['path']))}")
    print(f"Distance: {result['total_distance_km']:.2f} km")
    print(f"Predicted Speed: {result['predicted_speed_kmh']:.2f} km/h")
    print(f"ETA: {result['eta_minutes']:.1f} minutes")
    print(f"Arrival Time: {result['estimated_arrival_time']}")
    
    # Display weather information
    weather = result['weather_conditions']
    print("\nWeather Conditions:")
    print("-" * 40)
    print(f"Temperature: {weather['temperature']:.1f}°C")
    print(f"Precipitation: {weather['precipitation']:.1f} mm")
    print(f"Wind Speed: {weather['wind_speed']:.1f} km/h")
    print(f"Humidity: {weather['humidity']:.1f}%")
    
    # Weather impact
    if 'weather_impact_score' in result:
        impact = result['weather_impact_score']
        print(f"Weather Impact: {impact:.1f}% ({result.get('weather_description', 'Unknown')})")
    else:
        impact = 0
        if weather['is_rainy']:
            impact = min(100, weather['precipitation'] * 10)
        print(f"Weather Impact: {impact:.1f}%")
    
    # Create visualization
    try:
        plt.figure(figsize=(15, 10))
        
        # 1. Path visualization
        plt.subplot(2, 2, 1)
        path_length = len(result['path'])
        plt.plot(range(path_length), [1] * path_length, 'bo-')
        for i, node in enumerate(result['path']):
            plt.text(i, 1.1, str(node), ha='center')
        plt.title('Route Path')
        plt.xticks([])
        plt.yticks([])
        
        # 2. Speed with weather impact
        plt.subplot(2, 2, 2)
        base_speed = result['predicted_speed_kmh']
        plt.bar(['Predicted Speed'], [base_speed], color='skyblue')
        
        # Add weather impact indicator
        if impact > 0:
            plt.bar(['Weather Impact'], [impact/100 * base_speed], color='salmon', alpha=0.7)
            plt.axhline(y=base_speed, color='red', linestyle='--')
        
        plt.title('Speed Prediction with Weather Impact')
        plt.ylabel('Speed (km/h)')
        
        # 3. Weather conditions
        plt.subplot(2, 2, 3)
        weather_labels = ['Temperature', 'Precipitation', 'Wind', 'Humidity']
        weather_values = [
            weather['temperature'],
            weather['precipitation'],
            weather['wind_speed'],
            weather['humidity']
        ]
        colors = ['red', 'blue', 'green', 'purple']
        plt.bar(weather_labels, weather_values, color=colors)
        plt.title('Weather Conditions')
        
        # 4. ETA visualization
        plt.subplot(2, 2, 4)
        eta_minutes = result['eta_minutes']
        
        # Create a simple clock visualization
        clock = plt.Circle((0.5, 0.5), 0.4, fill=False)
        plt.gca().add_patch(clock)
        
        # Add hour and minute hands
        hours = eta_minutes / 60
        hour_angle = (hours % 12) * 30 * np.pi/180
        min_angle = (eta_minutes % 60) * 6 * np.pi/180
        
        plt.plot([0.5, 0.5 + 0.3 * np.sin(hour_angle)], 
                [0.5, 0.5 + 0.3 * np.cos(hour_angle)], 'k-', linewidth=3)
        plt.plot([0.5, 0.5 + 0.35 * np.sin(min_angle)], 
                [0.5, 0.5 + 0.35 * np.cos(min_angle)], 'r-', linewidth=2)
        
        plt.text(0.5, 0.2, f"{int(eta_minutes // 60)}h {int(eta_minutes % 60)}m", 
                ha='center', fontsize=12)
        plt.title('Estimated Travel Time')
        plt.xticks([])
        plt.yticks([])
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not create visualization: {e}")

def view_metrics(preprocessor):
    """View model metrics"""
    # Ensure metrics directory exists
    if not os.path.exists(METRICS_DIR):
        print(f"\nMetrics directory not found: {METRICS_DIR}")
        return
        
    metric_files = [f for f in os.listdir(METRICS_DIR) if f.endswith('.json')]
    
    if not metric_files:
        print("\nNo metrics files found in directory:", METRICS_DIR)
        return

    
    print("\nAvailable metrics:")
    for i, metric_file in enumerate(metric_files):
        print(f"{i+1}. {metric_file}")
    
    try:
        choice = int(input("\nSelect metrics to view (number): ")) - 1
        if choice < 0 or choice >= len(metric_files):
            print("Invalid selection.")
            return
        
        metric_path = os.path.join(METRICS_DIR, metric_files[choice])
        
        with open(metric_path, 'r') as f:
            metrics = json.load(f)
        
        print("\nTraining Metrics:")
        print("-" * 40)
        
        # Plot metrics
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(metrics['train_loss'], label='Training Loss')
        plt.plot(metrics['val_loss'], label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(metrics['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"\nError viewing metrics: {e}")

def view_system_status(preprocessor):
    """View system status"""
    print("\nSystem Status")
    print("-" * 40)
    
    # Data statistics
    print("\nData Statistics:")
    print(f"Speed data shape: {preprocessor.speed_data.shape}")
    print(f"Number of detectors: {preprocessor.speed_data['detid'].nunique()}")
    print(f"Date range: {preprocessor.speed_data['day'].min()} to {preprocessor.speed_data['day'].max()}")
    
    # Graph statistics
    print("\nGraph Statistics:")
    print(f"Number of nodes: {preprocessor.graph.number_of_nodes()}")
    print(f"Number of edges: {preprocessor.graph.number_of_edges()}")
    print(f"Network diameter: {preprocessor.graph_metrics['diameter']:.2f} meters")
    print(f"Average path length: {preprocessor.graph_metrics['average_shortest_path']:.2f} meters")
    
    # Model statistics
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
    print(f"\nNumber of saved models: {len(model_files)}")
    
    # System resources
    try:
        import psutil
        print("\nSystem Resources:")
        print(f"CPU usage: {psutil.cpu_percent()}%")
        print(f"Memory usage: {psutil.virtual_memory().percent}%")
        print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    except ImportError:
        print("\nSystem resource information not available (psutil not installed)")
    
    # Device info
    print(f"\nCompute device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
def load_model(preprocessor):
    """Load an existing model from disk with improved architecture detection"""
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
    
    if not model_files:
        print("\nNo saved models found.")
        return None
    
    print("\nAvailable models:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    try:
        choice = int(input("\nSelect a model to load (number): ")) - 1
        if choice < 0 or choice >= len(model_files):
            print("Invalid selection.")
            return None
        
        model_path = os.path.join(MODEL_DIR, model_files[choice])
        model_filename = model_files[choice]
        
        # Extract timestamp from filename
        timestamp = model_filename.split('_')[-1].split('.')[0]
        
        # Check for corresponding scalers file
        scaler_path = os.path.join(MODEL_DIR, f"scalers_{timestamp}.pkl")
        if os.path.exists(scaler_path):
            print(f"Loading scalers from {scaler_path}")
            scalers = joblib.load(scaler_path)
            preprocessor.scaler_X = scalers['scaler_X']
            preprocessor.scaler_y = scalers['scaler_y']
        else:
            print("Warning: No matching scalers found. Predictions may not work correctly.")
            # Try to find any scalers file
            scaler_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
            if scaler_files:
                print(f"Loading most recent scalers instead: {scaler_files[-1]}")
                scalers = joblib.load(os.path.join(MODEL_DIR, scaler_files[-1]))
                preprocessor.scaler_X = scalers['scaler_X']
                preprocessor.scaler_y = scalers['scaler_y']
        
        # Load the state dict
        state_dict = torch.load(model_path)
        
        # Determine model type and architecture from filename and state dict
        is_simple_model = 'simple' in model_filename.lower() or 'quick' in model_filename.lower() or any('feature_extractor' in key for key in state_dict.keys())
        
        if is_simple_model:
            print("Loading SimpleGATModel...")
            # For SimpleGATModel, determine input dimension from weights
            if 'feature_extractor.0.weight' in state_dict:
                input_dim = state_dict['feature_extractor.0.weight'].shape[1]
                hidden_dim = state_dict['feature_extractor.0.weight'].shape[0]
                
                model = SimpleGATModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=4,
                    dropout=0.3
                )
                model.load_state_dict(state_dict)
                print(f"Model loaded with input_dim={input_dim}, hidden_dim={hidden_dim}")
            else:
                print("Error: Cannot determine model dimensions from state dict")
                return None
        else:
            print("Loading GATTrafficPredictionModel...")
            
            # Analyze state dict to determine architecture parameters
            if 'attentions.0.W' in state_dict:
                # Get dimensions from the state dict
                nfeat = state_dict['attentions.0.W'].shape[0]
                nhid = state_dict['attentions.0.W'].shape[1]
                
                # Count number of attention heads
                nheads = 0
                while f'attentions.{nheads}.W' in state_dict:
                    nheads += 1
                
                # Get output dimension
                if 'out_att.W' in state_dict:
                    out_features = state_dict['out_att.W'].shape[1]
                else:
                    out_features = 4  # Default
                
                # Estimate number of nodes from final_linear layer if available
                if 'final_linear.weight' in state_dict:
                    num_nodes = state_dict['final_linear.weight'].shape[1] // out_features
                else:
                    num_nodes = 20  # Default
                
                print(f"Creating model with: nfeat={nfeat}, nhid={nhid}, nclass={out_features}, num_nodes={num_nodes}, nheads={nheads}")
                
                # Create model with exact same architecture as saved model
                model = GATTrafficPredictionModel(
                    nfeat=nfeat,
                    nhid=nhid,
                    nclass=out_features,
                    num_nodes=num_nodes,
                    dropout=0.3,
                    nheads=nheads
                )
                
                # Load state dict
                model.load_state_dict(state_dict)
            else:
                print("Error: Cannot determine model architecture from state dict")
                return None
        
        # Move model to device and set to evaluation mode
        model.to(device)
        model.eval()
        
        print(f"\nModel loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        print(f"\nError loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Enhanced main execution function with improved UI"""
    def print_header():
        print("\n" + "="*60)
        print("GAT Traffic Prediction System".center(60))
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
        print("7. Quick train (5 min)")
        print("8. Exit")
        print("-" * 40)




    # Main execution
    print_header()
    
    # Default file paths
    default_speed_path = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\corrected_speed_data.xlsx"
    default_distance_path = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\detector_distances.xlsx"
    
    try:
        # Initialize preprocessor
        preprocessor = TrafficDataPreprocessor(default_speed_path, default_distance_path)
        
        # Load data
        print("\nInitializing system and loading data...")
        preprocessor.load_data()
        
        # Initialize model
        current_model = None
        
        while True:
            print_menu()
            choice = input("\nEnter your choice (1-7): ")
            
            if choice == '1':
                current_model = train_new_model(preprocessor)  # Pass preprocessor directly
            elif choice == '2':
                current_model = load_model(preprocessor)
            elif choice == '3':
                make_prediction(preprocessor, current_model)
            elif choice == '4':
                view_metrics(preprocessor)
            elif choice == '5':
                make_prediction(preprocessor, current_model)
            elif choice == '6':
                view_system_status(preprocessor)
            elif choice == '7':
                print("\nStarting quick training (simplified model)...")
                current_model = quick_train(preprocessor)
            elif choice == '8':
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
