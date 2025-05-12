from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import joblib
import numpy as np

# Import functionality from gat.py
from gat import TrafficDataPreprocessor, SimpleGATModel, GATTrafficPredictionModel, TrafficPredictor, device

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
MODEL_PATH = r"D:\vscode\Applying-Deep-Learning-Network-For-City-Wide-Traffic-Inference\Data\gat\best_gat_model_20250512_100216.pth"
SCALER_PATH = r"D:\vscode\Applying-Deep-Learning-Network-For-City-Wide-Traffic-Inference\Data\gat\scalers_20250512_100216.pkl"
METRICS_DIR = r"D:\vscode\Applying-Deep-Learning-Network-For-City-Wide-Traffic-Inference\Data\gat\metrics"

# Default file paths
default_speed_path = r"D:\vscode\Applying-Deep-Learning-Network-For-City-Wide-Traffic-Inference\Data\corrected_speed_data.xlsx"
default_distance_path = r"D:\vscode\Applying-Deep-Learning-Network-For-City-Wide-Traffic-Inference\Data\detector_distances.xlsx"

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
    
    # Load scalers
    if os.path.exists(SCALER_PATH):
        print(f"Loading scalers from {SCALER_PATH}")
        scalers = joblib.load(SCALER_PATH)
        preprocessor.scaler_X = scalers['scaler_X']
        preprocessor.scaler_y = scalers['scaler_y']
    else:
        print(f"Scaler file not found at {SCALER_PATH}")
        return
    
    # Load model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        
        # Load the state dict
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
        # Analyze state dict to determine model architecture
        print("Analyzing model architecture from state dict...")
        for key in state_dict.keys():
            print(f"Key: {key}, Shape: {state_dict[key].shape}")
        
        # Check if it's a SimpleGATModel
        if any('feature_extractor' in key for key in state_dict.keys()):
            print("Detected SimpleGATModel")
            
            # Get input and hidden dimensions from the state dict
            input_dim = state_dict['feature_extractor.0.weight'].shape[1]
            hidden_dim = state_dict['feature_extractor.0.weight'].shape[0]
            
            # Create SimpleGATModel matching the saved architecture
            model = SimpleGATModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=4,
                dropout=0.3
            )
            
            # Load state dict
            model.load_state_dict(state_dict)
            print(f"SimpleGATModel loaded with input_dim={input_dim}, hidden_dim={hidden_dim}")
            
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
            
            # Estimate number of nodes from final_linear layer
            if 'final_linear.weight' in state_dict:
                num_nodes = state_dict['final_linear.weight'].shape[1] // out_features
            else:
                num_nodes = preprocessor.graph.number_of_nodes()
            
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
            print("Unknown model architecture. Cannot load model.")
            return
        
        # Move model to device and set to evaluation mode
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"No model found at {MODEL_PATH}. Please train a model first.")

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
