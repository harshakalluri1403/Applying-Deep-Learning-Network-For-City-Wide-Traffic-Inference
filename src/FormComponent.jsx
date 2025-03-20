

export default function FormComponent({ fromValue, setFromValue, toValue, setToValue, onSubmit }) {
  return (
    <form onSubmit={onSubmit}>
      <label htmlFor="from">From:</label>
      <input
        type="text"
        placeholder="Enter the start place..."
        className="from"
        id="from"
        value={fromValue}
        onChange={(e) => setFromValue(e.target.value)}
        required
      />
      
      <label htmlFor="to">To:</label>
      <input
        type="text"
        placeholder="Enter the destination..."
        className="to"
        id="to"
        value={toValue}
        onChange={(e) => setToValue(e.target.value)}
        required
      />
      
      <label htmlFor="model">Model:</label>
      <select id="model" className="model-select">
        <option value="standard">LSTM</option>
        <option value="traffic-aware">GAT</option>
      </select>
      
      <button type="submit" className="submitBtn">Submit</button>
    </form>
  );
}
