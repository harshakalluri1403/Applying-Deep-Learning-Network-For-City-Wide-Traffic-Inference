// import { useState, useEffect } from 'react';

// export default function FormComponent({ fromValue, setFromValue, toValue, setToValue, onSubmit }) {
//   const [detectors, setDetectors] = useState([]);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState(null);

//   useEffect(() => {
//     const fetchDetectors = async () => {
//       try {
//         setLoading(true);
//         const response = await fetch('http://localhost:5000/api/detectors');
//         const data = await response.json();
        
//         if (data.error) {
//           setError(data.error);
//         } else {
//           setDetectors(data.detectors || []);
//         }
//       } catch (err) {
//         setError('Failed to fetch detectors. Please make sure the backend is running.');
//         console.error('Error fetching detectors:', err);
//       } finally {
//         setLoading(false);
//       }
//     };

//     fetchDetectors();
//   }, []);

//   return (
//     <form onSubmit={onSubmit}>
//       <label htmlFor="from">From:</label>
//       {loading ? (
//         <input
//           type="text"
//           placeholder="Loading detectors..."
//           className="from"
//           id="from"
//           disabled
//         />
//       ) : error ? (
//         <input
//           type="text"
//           placeholder="Enter the start place..."
//           className="from"
//           id="from"
//           value={fromValue}
//           onChange={(e) => setFromValue(e.target.value)}
//           required
//         />
//       ) : detectors.length > 0 ? (
//         <select
//           id="from"
//           className="from"
//           value={fromValue}
//           onChange={(e) => setFromValue(e.target.value)}
//           required
//         >
//           <option value="">Select start detector</option>
//           {detectors.map((detector) => (
//             <option key={detector} value={detector}>
//               {detector}
//             </option>
//           ))}
//         </select>
//       ) : (
//         <input
//           type="text"
//           placeholder="Enter the start place..."
//           className="from"
//           id="from"
//           value={fromValue}
//           onChange={(e) => setFromValue(e.target.value)}
//           required
//         />
//       )}
      
//       <label htmlFor="to">To:</label>
//       {loading ? (
//         <input
//           type="text"
//           placeholder="Loading detectors..."
//           className="to"
//           id="to"
//           disabled
//         />
//       ) : error ? (
//         <input
//           type="text"
//           placeholder="Enter the destination..."
//           className="to"
//           id="to"
//           value={toValue}
//           onChange={(e) => setToValue(e.target.value)}
//           required
//         />
//       ) : detectors.length > 0 ? (
//         <select
//           id="to"
//           className="to"
//           value={toValue}
//           onChange={(e) => setToValue(e.target.value)}
//           required
//         >
//           <option value="">Select destination detector</option>
//           {detectors.map((detector) => (
//             <option key={detector} value={detector}>
//               {detector}
//             </option>
//           ))}
//         </select>
//       ) : (
//         <input
//           type="text"
//           placeholder="Enter the destination..."
//           className="to"
//           id="to"
//           value={toValue}
//           onChange={(e) => setToValue(e.target.value)}
//           required
//         />
//       )}
      
//       <label htmlFor="model">Model:</label>
//       <select id="model" className="model-select">
//         <option value="standard">LSTM</option>
//         <option value="traffic-aware">GAT</option>
//       </select>
      
//       <button type="submit" className="submitBtn">Submit</button>
//     </form>
//   );
// }
import { useState, useEffect } from "react";
import axios from "axios";

export default function FormComponent({
  fromValue,
  setFromValue,
  toValue,
  setToValue,
  onSubmit,
}) {
  const [detectors, setDetectors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [outputText, setOutputText] = useState(""); // Store API response

  useEffect(() => {
    const fetchDetectors = async () => {
      try {
        setLoading(true);
        const response = await axios.get("http://localhost:5000/api/detectors");
        setDetectors(response.data.detectors || []);
      } catch (err) {
        setError("Failed to fetch detectors. Please make sure the backend is running.");
        console.error("Error fetching detectors:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchDetectors();
  }, []);

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      const response = await axios.post("http://127.0.0.1:5000/process", {
        from: fromValue,
        to: toValue,
      });

      setOutputText(response.data.output); // Set processed output
      onSubmit(fromValue, toValue);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="from">From:</label>
      {loading ? (
        <input type="text" placeholder="Loading detectors..." id="from" disabled />
      ) : error ? (
        <input
          type="text"
          placeholder="Enter the start place..."
          id="from"
          value={fromValue}
          onChange={(e) => setFromValue(e.target.value)}
          required
        />
      ) : (
        <select id="from" value={fromValue} onChange={(e) => setFromValue(e.target.value)} required>
          <option value="">Select start detector</option>
          {detectors.map((detector) => (
            <option key={detector} value={detector}>
              {detector}
            </option>
          ))}
        </select>
      )}

      <label htmlFor="to">To:</label>
      {loading ? (
        <input type="text" placeholder="Loading detectors..." id="to" disabled />
      ) : error ? (
        <input
          type="text"
          placeholder="Enter the destination..."
          id="to"
          value={toValue}
          onChange={(e) => setToValue(e.target.value)}
          required
        />
      ) : (
        <select id="to" value={toValue} onChange={(e) => setToValue(e.target.value)} required>
          <option value="">Select destination detector</option>
          {detectors.map((detector) => (
            <option key={detector} value={detector}>
              {detector}
            </option>
          ))}
        </select>
      )}

      <label htmlFor="model">Model:</label>
      <select id="model">
        <option value="standard">LSTM</option>
        <option value="traffic-aware">GAT</option>
      </select>

      <button type="submit">Submit</button>

      {outputText && <p>Processed Output: {outputText}</p>}
    </form>
  );
}
