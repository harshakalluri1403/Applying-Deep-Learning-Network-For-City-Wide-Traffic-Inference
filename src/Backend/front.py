import streamlit as st
import datetime
import requests
import folium
from streamlit_folium import folium_static
import pandas as pd
import time
import urllib.parse

# Set page configuration
st.set_page_config(
    page_title="Traffic Prediction System",
    layout="wide"
)

# Custom CSS to match the styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .container {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
    }
    .m-container {
        display: flex;
        flex-direction: column;
    }
    .weather-container {
        margin-top: 1rem;
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f8ff;
        border: 1px solid #cce5ff;
    }
    .weather-card h3 {
        margin-bottom: 0.5rem;
    }
    .weather-main {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .weather-emoji {
        font-size: 2rem;
        margin-right: 0.5rem;
    }
    .weather-temp {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .weather-description {
        font-style: italic;
        margin-bottom: 0.5rem;
    }
    .weather-details {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.5rem;
    }
    .prediction-output {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 8px;
        background-color: #e9ecef;
        border: 1px solid #ced4da;
    }
    .model-metrics {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 8px;
        background-color: #e2e3e5;
        border: 1px solid #d6d8db;
    }
</style>
""", unsafe_allow_html=True)

# Main heading
st.markdown("<h1 class='main-header'>TRAFFIC PREDICTION SYSTEM</h1>", unsafe_allow_html=True)

# Initialize session state variables
if 'from_value' not in st.session_state:
    st.session_state.from_value = ""
if 'to_value' not in st.session_state:
    st.session_state.to_value = ""
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'model' not in st.session_state:
    st.session_state.model = "standard"

# Create two columns for the top section
col1, col2 = st.columns([3, 1])

# Form Component in the first column
with col1:
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    with st.form(key='route_form'):
        from_input = st.text_input("From:", value=st.session_state.from_value)
        to_input = st.text_input("To:", value=st.session_state.to_value)
        model_options = ["Standard", "Traffic-Aware", "Eco-Route"]
        model_selection = st.selectbox("Model:", options=model_options, index=0)
        
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            st.session_state.from_value = from_input
            st.session_state.to_value = to_input
            st.session_state.model = model_selection.lower()
            st.session_state.submitted = True
            st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Date and Time Component in the second column
with col2:
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    def get_current_datetime():
        now = datetime.datetime.now()
        formatted_date = now.strftime("%d-%m-%y")
        formatted_time = now.strftime("%H:%M:%S")
        return formatted_time, formatted_date
    
    time_placeholder = st.empty()
    date_placeholder = st.empty()
    
    # Function to update time (will be called periodically)
    def update_time():
        current_time, current_date = get_current_datetime()
        time_placeholder.markdown(f"<p>Time:</p><p>{current_time}</p>", unsafe_allow_html=True)
        date_placeholder.markdown(f"<p>Date:</p><p>{current_date}</p>", unsafe_allow_html=True)
    
    # Initial update
    update_time()
    st.markdown("</div>", unsafe_allow_html=True)

# Improved geocoding function with proper error handling and URL encoding
def geocode(address):
    if not address:
        return None
        
    try:
        # Properly encode the address for URL
        encoded_address = urllib.parse.quote(address)
        
        # Add user-agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request
        response = requests.get(
            f"https://nominatim.openstreetmap.org/search?q={encoded_address}&format=json&limit=1",
            headers=headers
        )
        
        # Check if request was successful
        if response.status_code != 200:
            st.error(f"Geocoding API returned status code {response.status_code}")
            return None
            
        # Parse the JSON response
        data = response.json()
        
        # Check if we got any results
        if not data or len(data) == 0:
            st.warning(f"No geocoding results found for '{address}'")
            return None
            
        # Return the coordinates
        return [float(data[0]['lat']), float(data[0]['lon'])]
        
    except Exception as e:
        st.error(f"Geocoding error: {str(e)}")
        return None

# Fallback geocoding using hardcoded coordinates for common cities
def get_fallback_coordinates(city_name):
    # Dictionary of common cities and their approximate coordinates
    cities = {
        'new york': [40.7128, -74.0060],
        'london': [51.5074, -0.1278],
        'paris': [48.8566, 2.3522],
        'tokyo': [35.6762, 139.6503],
        'sydney': [-33.8688, 151.2093],
        'berlin': [52.5200, 13.4050],
        'rome': [41.9028, 12.4964],
        'beijing': [39.9042, 116.4074],
        'delhi': [28.6139, 77.2090],
        'mumbai': [19.0760, 72.8777],
        'los angeles': [34.0522, -118.2437],
        'chicago': [41.8781, -87.6298],
        'houston': [29.7604, -95.3698],
        'madrid': [40.4168, -3.7038],
        'dubai': [25.2048, 55.2708],
        'san francisco': [37.7749, -122.4194],
        'seattle': [47.6062, -122.3321],
        'miami': [25.7617, -80.1918],
        'amsterdam': [52.3676, 4.9041],
        'toronto': [43.6532, -79.3832],
        'singapore': [1.3521, 103.8198],
        'mexico city': [19.4326, -99.1332],
        'washington': [38.9072, -77.0369],
        'washington dc': [38.9072, -77.0369],
        'boston': [42.3601, -71.0589],
        'barcelona': [41.3851, 2.1734],
        'munich': [48.1351, 11.5820],
        'bangkok': [13.7563, 100.5018],
        'moscow': [55.7558, 37.6173],
        'vienna': [48.2082, 16.3738],
    }
    
    # Normalize the city name (lowercase and strip)
    normalized_name = city_name.lower().strip()
    
    # Try to match the city name
    for city, coords in cities.items():
        if city in normalized_name or normalized_name in city:
            return coords
            
    # Return None if no match
    return None

# Function to get weather data
def get_weather(location):
    API_KEY = "66587ee2b0a5031f6d4a5012ed5b004e"  # Using the provided API key
    try:
        # Properly encode the location
        encoded_location = urllib.parse.quote(location)
        
        response = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={encoded_location}&units=metric&appid={API_KEY}"
        )
        
        # Check if request was successful
        if response.status_code != 200:
            st.warning(f"Weather API returned status code {response.status_code} for '{location}'")
            return None
            
        return response.json()
    except Exception as e:
        st.error(f"Weather API error: {str(e)}")
        return None

# Function to render weather emoji
def get_weather_emoji(weather):
    if not weather or 'weather' not in weather or not weather['weather']:
        return '🌡️'
    
    condition = weather['weather'][0]['main'].lower()
    
    emojis = {
        'clear': '☀️',
        'clouds': '☁️',
        'rain': '🌧️',
        'drizzle': '🌦️',
        'thunderstorm': '⛈️',
        'snow': '❄️',
        'mist': '🌫️',
        'fog': '🌫️',
        'haze': '🌫️'
    }
    
    return emojis.get(condition, '🌡️')

# Function to render weather information
def render_weather_info(weather):
    if not weather:
        return st.error("Could not retrieve weather data")
    
    emoji = get_weather_emoji(weather)
    temp = round(weather['main']['temp'])
    feels_like = round(weather['main']['feels_like'])
    description = weather['weather'][0]['description']
    humidity = weather['main']['humidity']
    wind_speed = weather['wind']['speed']
    pressure = weather['main']['pressure']
    visibility = weather.get('visibility', 0) / 1000
    
    st.markdown(f"""
    <div class='weather-main'>
        <span class='weather-emoji'>{emoji}</span>
        <span class='weather-temp'>{temp}°C</span>
    </div>
    <div class='weather-description'>
        {description.capitalize()}
    </div>
    <div class='weather-details'>
        <div>💧 Humidity: {humidity}%</div>
        <div>💨 Wind: {wind_speed} m/s</div>
        <div>🌡️ Feels like: {feels_like}°C</div>
        <div>🌀 Pressure: {pressure} hPa</div>
        <div>🔭 Visibility: {visibility} km</div>
    </div>
    """, unsafe_allow_html=True)

# Create columns for the lower section
if st.session_state.submitted:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Prediction Output Component
        st.markdown("<div class='prediction-output'>", unsafe_allow_html=True)
        # Mock data similar to React app
        eta = "25 min"
        volume = "Medium"
        speed_req = "30 km/h"
        path = f"{st.session_state.from_value} to {st.session_state.to_value}"
        
        st.markdown(f"""
        <p>ETA: <strong>{eta}</strong></p>
        <p>Vol: <strong>{volume}</strong></p>
        <p>Speed Req: <strong>{speed_req}</strong></p>
        <p class="path-info">Path: <strong>{path}</strong></p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Prediction Model Metrics
        st.markdown("<div class='model-metrics'>", unsafe_allow_html=True)
        st.markdown("""
        <p>Prediction Model Metrics:</p>
        <p>INFO BOX</p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Weather Component
        st.markdown("<div class='weather-container'>", unsafe_allow_html=True)
        st.markdown(f"<h3>Weather at {st.session_state.from_value}</h3>", unsafe_allow_html=True)
        
        with st.spinner("Loading weather data..."):
            weather_data = get_weather(st.session_state.from_value)
            if weather_data:
                render_weather_info(weather_data)
            else:
                st.error(f"Could not fetch weather for {st.session_state.from_value}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_right:
        # Map Component
        st.markdown("<div style='height: 600px; border: 1px solid #ccc;'>", unsafe_allow_html=True)
        
        with st.spinner("Loading map..."):
            # Try to get coordinates using the geocoding API
            from_coords = geocode(st.session_state.from_value)
            to_coords = geocode(st.session_state.to_value)
            
            # If geocoding fails, try to use fallback coordinates
            if not from_coords:
                from_coords = get_fallback_coordinates(st.session_state.from_value)
                if from_coords:
                    st.info(f"Using approximate coordinates for {st.session_state.from_value}")
                    
            if not to_coords:
                to_coords = get_fallback_coordinates(st.session_state.to_value)
                if to_coords:
                    st.info(f"Using approximate coordinates for {st.session_state.to_value}")
            
            # If we have coordinates for both locations, create the map
            if from_coords and to_coords:
                # Calculate center
                center = [(from_coords[0] + to_coords[0]) / 2, (from_coords[1] + to_coords[1]) / 2]
                
                # Create a map
                m = folium.Map(location=center, zoom_start=6)
                
                # Add markers for 'from' and 'to' locations
                folium.Marker(
                    location=from_coords,
                    popup=st.session_state.from_value,
                    icon=folium.Icon(color='blue')
                ).add_to(m)
                
                folium.Marker(
                    location=to_coords,
                    popup=st.session_state.to_value,
                    icon=folium.Icon(color='red')
                ).add_to(m)
                
                # Add a line between the two points
                folium.PolyLine(
                    locations=[from_coords, to_coords],
                    color='blue',
                    weight=5,
                    opacity=0.8
                ).add_to(m)
                
                # Display the map
                folium_static(m, height=600)
            else:
                st.error("Could not geocode one or both locations. Please check the input addresses.")
                
                # Show a default map if we can't get coordinates
                default_map = folium.Map(location=[0, 0], zoom_start=2)
                folium_static(default_map, height=600)
        
        st.markdown("</div>", unsafe_allow_html=True)
else:
    # Display empty containers when not submitted
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("<div class='prediction-output'>", unsafe_allow_html=True)
        st.markdown("""
        <p>ETA: <strong>Output 1</strong></p>
        <p>Vol: <strong>Output 2</strong></p>
        <p>Speed Req: <strong>Output 3</strong></p>
        <p class="path-info">Path: <strong>Output 4</strong></p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='model-metrics'>", unsafe_allow_html=True)
        st.markdown("""
        <p>Prediction Model Metrics:</p>
        <p>INFO BOX</p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_right:
        st.markdown("""
        <div style='height: 600px; width: 100%; border: 1px solid #ccc; display: flex; justify-content: center; align-items: center;'>
            <p>Enter locations and submit to view the map</p>
        </div>
        """, unsafe_allow_html=True)

# For clock updates (this is a simplified approach - in production, you'd want a better method)
# Using a smaller sleep time to make the app more responsive
if st.session_state.get('run_clock', True):
    update_time()
    time.sleep(0.5)  # Reduced sleep time
    st.experimental_rerun()  # Using experimental_rerun to maintain compatibility with older Streamlit versions