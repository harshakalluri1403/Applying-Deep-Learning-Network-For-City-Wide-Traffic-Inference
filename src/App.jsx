
import { useState } from 'react';
import './App.css';
import MainHeading from './MainHeading';
import FirstContainer from './FirstContainer';
import PredictionOutputComponent from './PredictionOutputComponent';
import PredictionModelMetrics from './PredictionModelMetrics';
import MapComponent from './MapComponent';
import WeatherComponent from './WeatherComponent';

function App() {
  const [from, setFrom] = useState('');
  const [to, setTo] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (fromValue, toValue) => {
    setFrom(fromValue);
    setTo(toValue);
    setSubmitted(true); 
  };

  return (
    <>
      <MainHeading />
      <div className="container">
        <FirstContainer onSubmit={handleSubmit} />
      </div>
      
      <div className="m-container">
        <div className="container" style={{maxWidth:'50%'}}>
          <PredictionOutputComponent from={from} to={to} submitted={submitted} />
          <PredictionModelMetrics />
          {submitted && <WeatherComponent from={from} to={to} />}
        </div>

        <div className="map-container">
            {submitted && <MapComponent from={from} to={to} />}
          </div>
      </div>
    </>
  );
}

export default App;