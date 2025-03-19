import { useState, useEffect } from 'react';

export default function PredictionOutputComponent({ from, to, submitted }) {
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchPrediction = async () => {
            if (!submitted || !from || !to) return;
            
            setLoading(true);
            setError(null);
            
            try {
                // Get the selected model type
                const modelSelect = document.getElementById('model');
                const modelType = modelSelect ? 
                    (modelSelect.options[modelSelect.selectedIndex].value === 'traffic-aware' ? 'gat' : 'lstm') 
                    : 'gat';
                
                const response = await fetch('http://localhost:5000/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        from: from,
                        to: to,
                        model: modelType
                    }),
                });
                
                const data = await response.json();
                
                if (data.error) {
                    setError(data.error);
                } else {
                    setPrediction(data);
                }
            } catch (err) {
                setError('Failed to connect to the server. Please make sure the backend is running.');
                console.error('Error:', err);
            } finally {
                setLoading(false);
            }
        };
        
        fetchPrediction();
    }, [from, to, submitted]);

    if (loading) {
        return (
            <div className="prediction-output loading">
                <p>Loading prediction...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="prediction-output error">
                <p>Error: {error}</p>
            </div>
        );
    }

    if (!submitted) {
        return (
            <div className="prediction-output">
                <p>Distance: <strong>Output 5</strong></p>
                <p>ETA: <strong>Output 1</strong></p>
                <p>Vol: <strong>Output 2</strong></p>
                <p>Speed Req: <strong>Output 3</strong></p>
                <p className="path-info">Path: <strong>Output 4</strong></p>
            </div>
        );
    }

    if (!prediction) {
        return (
            <div className="prediction-output">
                <p>Enter source and destination to see prediction</p>
            </div>
        );
    }

    return (
        <div className="prediction-output">
            <p>Distance: <strong>{(prediction.total_distance_km || 0).toFixed(2)} Km</strong></p>
            <p>ETA: <strong>{(prediction.eta_minutes || 0).toFixed(2)} Minutes</strong></p>
            <p>Vol: <strong>{(prediction.predicted_flow || 0).toFixed(2)} Vehicles</strong></p>
            <p>Speed Req: <strong>{(prediction.predicted_speed_kmh || 0).toFixed(2)} km/h</strong></p>
            <p className="path-info">Path: <strong>{prediction.path ? prediction.path.join(' → ') : `${from} to ${to}`}</strong></p>
        </div>
    );
}
