


// MapComponent.jsx
import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

export default function MapComponent({ from, to }) {
  const [fromCoords, setFromCoords] = useState(null); // Null initially until geocoded
  const [toCoords, setToCoords] = useState(null);     // Null initially until geocoded
  const [error, setError] = useState(null);           // To handle geocoding errors

  // Geocode function to convert address to coordinates
  const geocode = async (address, setCoords) => {
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(address)}&format=json&limit=1`
      );
      const data = await response.json();
      if (data.length > 0) {
        setCoords([parseFloat(data[0].lat), parseFloat(data[0].lon)]);
      } else {
        setError(`Could not find coordinates for "${address}"`);
      }
    } catch (err) {
      setError(`Geocoding error: ${err.message}`);
    }
  };

  // Fetch coordinates when "from" or "to" changes
  useEffect(() => {
    setFromCoords(null); // Reset coords when new input comes
    setToCoords(null);
    setError(null);      // Reset error
    if (from) geocode(from, setFromCoords);
    if (to) geocode(to, setToCoords);
  }, [from, to]);

  // Only render map if both coordinates are available
  if (!fromCoords || !toCoords) {
    return (
      <div
        style={{
          height: '600px',
          width: '500px',
          border: '1px solid #ccc',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        {error ? <p>{error}</p> : <p>Loading map...</p>}
      </div>
    );
  }

  // Calculate center of the map
  const center = [
    (fromCoords[0] + toCoords[0]) / 2,
    (fromCoords[1] + toCoords[1]) / 2,
  ];

  // Polyline positions (straight line between points)
  const polylinePositions = [fromCoords, toCoords];

  return (
    <div
      style={{
        height: '70vh',
        border: '1px solid #ccc',
      }}
    >
      <MapContainer
        center={center}
        zoom={6} // Adjust zoom level as needed
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        <Marker position={fromCoords}>
          <Popup>{from}</Popup>
        </Marker>
        <Marker position={toCoords}>
          <Popup>{to}</Popup>
        </Marker>
        <Polyline positions={polylinePositions} color="blue" weight={5} opacity={0.8} />
      </MapContainer>
    </div>
  );
}





