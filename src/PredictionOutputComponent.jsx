

export default function PredictionOutputComponent({ from, to, submitted }) {
    // Mock data for demonstration
    const dist = "0.19 Km";
    const eta = "0.99 Minutes";
    const volume = "86.06 Vehicles";
    const speedReq = "11.54 km/h";
    const path = submitted ? `${from} to ${to}` : "From-To";
    
    return (
        <div className="prediction-output">
            <p>Distance: <strong>{submitted ? dist : "Output 5"}</strong></p>
            <p>ETA: <strong>{submitted ? eta : "Output 1"}</strong></p>
            <p>Speed Req: <strong>{submitted ? speedReq : "Output 3"}</strong></p>
            <p className="path-info">Path: <strong>{submitted ? path : "Output 4"}</strong></p>
        </div>
    );
}