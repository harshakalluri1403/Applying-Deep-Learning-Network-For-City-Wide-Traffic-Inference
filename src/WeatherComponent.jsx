// import { useState, useEffect } from 'react';

// export default function WeatherComponent({ from }) {
//   const [fromWeather, setFromWeather] = useState(null);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState(null);
  
//   const API_KEY = import.meta.env.VITE_API_KEY;
  
//   useEffect(() => {
//     const fetchWeather = async (location) => {
//       try {
//         const response = await fetch(
//           `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(location)}&units=metric&appid=${API_KEY}`
//         );
        
//         if (!response.ok) {
//           throw new Error(`Weather for ${location} not found`);
//         }
        
//         const data = await response.json();
//         setFromWeather(data);
//       } catch (err) {
//         console.error(`Error fetching weather for ${location}:`, err);
//         setError(`Couldn't fetch weather for ${location}: ${err.message}`);
//       } finally {
//         setLoading(false);
//       }
//     };
    
//     if (from) {
//       setLoading(true);
//       setError(null);
//       fetchWeather(from);
//     }
//   }, [from, API_KEY]);
  
//   const getWeatherEmoji = (weather) => {
//     if (!weather || !weather.weather || !weather.weather[0]) return '🌡️';
    
//     const condition = weather.weather[0].main.toLowerCase();
    
//     switch (condition) {
//       case 'clear':
//         return '☀️';
//       case 'clouds':
//         return '☁️';
//       case 'rain':
//         return '🌧️';
//       case 'drizzle':
//         return '🌦️';
//       case 'thunderstorm':
//         return '⛈️';
//       case 'snow':
//         return '❄️';
//       case 'mist':
//       case 'fog':
//       case 'haze':
//         return '🌫️';
//       default:
//         return '🌡️';
//     }
//   };
  
//   const getWeatherInfo = (weather) => {
//     if (!weather) return null;
    
//     const emoji = getWeatherEmoji(weather);
//     const temp = Math.round(weather.main.temp);
//     const description = weather.weather[0].description;
//     const humidity = weather.main.humidity;
//     const windSpeed = weather.wind.speed;
    
//     return (
//       <div className="weather-info">
//         <div className="weather-main">
//           <span className="weather-emoji">{emoji}</span>
//           <span className="weather-temp">{temp}°C</span>
//         </div>
//         <div className="weather-description">
//           {description.charAt(0).toUpperCase() + description.slice(1)}
//         </div>
//         <div className="weather-details">
//           <div>💧 Humidity: {humidity}%</div>
//           <div>💨 Wind: {windSpeed} m/s</div>
//         </div>
//       </div>
//     );
//   };

//   if (!from) {
//     return <div className="weather-container">Enter a location to see weather information</div>;
//   }

//   if (loading) {
//     return <div className="weather-container">Loading weather data... ⏳</div>;
//   }

//   if (error) {
//     return <div className="weather-container">Error: {error}</div>;
//   }

//   return (
//     <div className="weather-container">
//       <div className="weather-card">
//         <h3>Weather at {from} {getWeatherEmoji(fromWeather)}</h3>
//         {getWeatherInfo(fromWeather)}
//       </div>
//     </div>
//   );
// }

import { useState, useEffect } from 'react';

export default function WeatherComponent({ from = 'Augsburg' }) {
  const [fromWeather, setFromWeather] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const API_KEY = import.meta.env.VITE_API_KEY;
  
  useEffect(() => {
    const fetchWeather = async (location) => {
      try {
        const response = await fetch(
          `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(location)}&units=metric&appid=${API_KEY}`
        );
        
        if (!response.ok) {
          throw new Error(`Weather for ${location} not found`);
        }
        
        const data = await response.json();
        setFromWeather(data);
      } catch (err) {
        console.error(`Error fetching weather for ${location}:`, err);
        setError(`Couldn't fetch weather for ${location}: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };
    
    if (from) {
      setLoading(true);
      setError(null);
      fetchWeather(from);
    }
  }, [from, API_KEY]);
  
  const getWeatherEmoji = (weather) => {
    if (!weather || !weather.weather || !weather.weather[0]) return '🌡️';
    
    const condition = weather.weather[0].main.toLowerCase();
    
    switch (condition) {
      case 'clear':
        return '☀️';
      case 'clouds':
        return '☁️';
      case 'rain':
        return '🌧️';
      case 'drizzle':
        return '🌦️';
      case 'thunderstorm':
        return '⛈️';
      case 'snow':
        return '❄️';
      case 'mist':
      case 'fog':
      case 'haze':
        return '🌫️';
      default:
        return '🌡️';
    }
  };
  
  const getWeatherInfo = (weather) => {
    if (!weather) return null;
    
    const emoji = getWeatherEmoji(weather);
    const temp = Math.round(weather.main.temp);
    const feelsLike = Math.round(weather.main.feels_like);
    const description = weather.weather[0].description;
    const humidity = weather.main.humidity;
    const windSpeed = weather.wind.speed;
    const pressure = weather.main.pressure;
    const visibility = weather.visibility / 1000;
    
    return (
      <div className="weather-info">
        <div className="weather-main">
          <span className="weather-emoji">{emoji}</span>
          <span className="weather-temp">{temp}°C</span>
        </div>
        <div className="weather-description">
          {description.charAt(0).toUpperCase() + description.slice(1)}
        </div>
        <div className="weather-details">
          <div>💧 Humidity: {humidity}%</div>
          <div>💨 Wind: {windSpeed} m/s</div>
          <div>🌡️ Feels like: {feelsLike}°C</div>
          <div>🌀 Pressure: {pressure} hPa</div>
          <div>🔭 Visibility: {visibility} km</div>
        </div>
      </div>
    );
  };

  if (loading) {
    return <div className="weather-container">Loading weather data... ⏳</div>;
  }

  if (error) {
    return <div className="weather-container">Error: {error}</div>;
  }

  return (
    <div className="weather-container">
      <div className="weather-card">
        <h3>Weather at {from} {getWeatherEmoji(fromWeather)}</h3>
        {getWeatherInfo(fromWeather)}
      </div>
    </div>
  );
}
