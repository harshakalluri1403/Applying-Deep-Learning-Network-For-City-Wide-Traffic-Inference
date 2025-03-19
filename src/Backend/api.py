# api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime
import os
import sys
import json
import torch
import joblib
import numpy as np

# Import functionality from gat.py
from gat import TrafficDataPreprocessor, SimpleGATModel, GATTrafficPredictionModel, TrafficPredictor, device

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants from gat.py
MODEL_DIR = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\gat"
METRICS_DIR = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\gat\\metrics"

# Default file paths
default_speed_path = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\corrected_speed_data.xlsx"
default_distance_path = "C:\\Users\\harsh\\Desktop\\utd\\utd\\extract\\detector_distances.xlsx"

# Initialize preprocessor and model globally
preprocessor = None
model = None

def initialize():
    global preprocessor, model
    
    # Initialize preprocessor
    preprocessor = TrafficDataPreprocessor(default_speed_path, default_distance_path)
    
    # Load data
    print("Initializing system and loading data...")
    preprocessor.load_data()
    
    # Load model
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
    if model_files:
        # Get the most recent model
        model_path = os.path.join(MODEL_DIR, model_files[-1])
        model_filename = model_files[-1]
        
        print(f"Loading model from {model_path}")
        
        # Extract timestamp from filename
        timestamp = model_filename.split('_')[-1].split('.')[0]
        
        # Check for corresponding scalers file
        scaler_path = os.path.join(MODEL_DIR, f"scalers_{timestamp}.pkl")
        if os.path.exists(scaler_path):
            print(f"Loading scalers from {scaler_path}")
            scalers = joblib.load(scaler_path)
            preprocessor.scaler_X = scalers['scaler_X']
            preprocessor.scaler_y = scalers['scaler_y']
        
        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Analyze state dict to determine model architecture
        print("Analyzing model architecture from state dict...")
        for key in state_dict.keys():
            print(f"Key: {key}, Shape: {state_dict[key].shape}")
        
        # Check if it's a SimpleGATModel with feature_extractor layers
        if any('feature_extractor' in key for key in state_dict.keys()):
            print("Detected SimpleGATModel with feature_extractor layers")
            
            # Get input and hidden dimensions from the state dict
            input_dim = state_dict['feature_extractor.0.weight'].shape[1]
            hidden_dim = state_dict['feature_extractor.0.weight'].shape[0]
            
            # Create a custom SimpleGATModel that matches the saved architecture
            class CustomSimpleGATModel(torch.nn.Module):
                def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
                    super(CustomSimpleGATModel, self).__init__()
                    self.feature_extractor = torch.nn.Sequential(
                        torch.nn.Linear(input_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim, output_dim)
                    )
                    self.dropout = torch.nn.Dropout(dropout)
                
                def forward(self, x, adj=None):
                    x = self.dropout(x)
                    x = self.feature_extractor(x)
                    return x
            
            # Create model with matching architecture
            model = CustomSimpleGATModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=4,
                dropout=0.3
            )
            
            # Load state dict
            model.load_state_dict(state_dict)
            print(f"CustomSimpleGATModel loaded with input_dim={input_dim}, hidden_dim={hidden_dim}")
            
        # Check if it's a SimpleGATModel with attention layers
        elif any('attention' in key for key in state_dict.keys()):
            print("Detected SimpleGATModel with attention layers")
            
            # Get dimensions from the state dict
            if 'attention.0.weight' in state_dict:
                in_features = state_dict['attention.0.weight'].shape[1]
                out_features = state_dict['attention.0.weight'].shape[0]
                
                # Create a custom SimpleGATModel that matches the saved architecture
                class CustomAttentionModel(torch.nn.Module):
                    def __init__(self, in_features, out_features, dropout=0.3):
                        super(CustomAttentionModel, self).__init__()
                        self.attention = torch.nn.Sequential(
                            torch.nn.Linear(in_features, out_features),
                            torch.nn.ReLU(),
                            torch.nn.Linear(out_features, 4)
                        )
                        self.dropout = torch.nn.Dropout(dropout)
                    
                    def forward(self, x, adj=None):
                        x = self.dropout(x)
                        x = self.attention(x)
                        return x
                
                # Create model with matching architecture
                model = CustomAttentionModel(
                    in_features=in_features,
                    out_features=out_features,
                    dropout=0.3
                )
                
                # Load state dict
                model.load_state_dict(state_dict)
                print(f"CustomAttentionModel loaded with in_features={in_features}, out_features={out_features}")
                
        # Check if it's a GATTrafficPredictionModel
        elif any('attentions.0.W' in key for key in state_dict.keys()):
            print("Detected GATTrafficPredictionModel")
            
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
            print(f"GATTrafficPredictionModel loaded with nfeat={nfeat}, nhid={nhid}, nheads={nheads}")
        
        else:
            print("Unknown model architecture. Creating a default SimpleGATModel.")
            model = SimpleGATModel(input_dim=16, hidden_dim=32, output_dim=4)
        
        # Move model to device and set to evaluation mode
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully from {model_path}")
    else:
        print("No saved models found. Please train a model first.")

# Initialize on startup
initialize()

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        start_detector = data.get('from')
        end_detector = data.get('to')
        
        if not start_detector or not end_detector:
            return jsonify({"error": "Missing 'from' or 'to' parameters"}), 400
        
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Create predictor
        predictor = TrafficPredictor(model, preprocessor, device)
        
        # Make prediction
        result = predictor.predict_eta(start_detector, end_detector)
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/detectors', methods=['GET'])
def get_detectors():
    try:
        if preprocessor and preprocessor.graph:
            detectors = sorted(list(preprocessor.graph.nodes()))
            return jsonify({"detectors": detectors})
        else:
            return jsonify({"error": "Preprocessor not initialized"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
