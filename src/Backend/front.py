import streamlit as st
import requests
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

def get_weather(api_key, location):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None

def get_coordinates(location):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location_data = geolocator.geocode(location)
    if location_data:
        return location_data.latitude, location_data.longitude
    return None, None

def create_map(from_location, to_location):
    from_lat, from_lon = get_coordinates(from_location)
    to_lat, to_lon = get_coordinates(to_location)
    
    if from_lat and to_lat:
        map_obj = folium.Map(location=[(from_lat + to_lat) / 2, (from_lon + to_lon) / 2], zoom_start=12)
        folium.Marker([from_lat, from_lon], popup=f"From: {from_location}", icon=folium.Icon(color="blue")).add_to(map_obj)
        folium.Marker([to_lat, to_lon], popup=f"To: {to_location}", icon=folium.Icon(color="red")).add_to(map_obj)
        folium.PolyLine([(from_lat, from_lon), (to_lat, to_lon)], color="blue", weight=2.5).add_to(map_obj)
        return map_obj
    return None

st.set_page_config(layout="wide")
st.title("Traffic Prediction System")

col1, col2 = st.columns([3, 1])

with col1:
    from_location = st.text_input("From:")
    to_location = st.text_input("To:")
    submit = st.button("Submit")

with col2:
    st.write("### Date & Time")
    st.write("**Date:**", st.session_state.get("date", "--"))
    st.write("**Time:**", st.session_state.get("time", "--"))

if submit:
    if from_location and to_location:
        st.session_state["date"] = st.date_input("Select Date").strftime("%Y-%m-%d")
        st.session_state["time"] = st.time_input("Select Time").strftime("%H:%M:%S")
        
        col_left, col_right = st.columns([1, 3])
        with col_left:
            st.write("### ETA, Volume, Speed & Path")
            st.write("ETA: -- Vol: -- Speed Required: --")
            st.write(f"Path: {from_location} → {to_location}")
            
            st.write("### Weather")
            api_key = "YOUR_OPENWEATHERMAP_API_KEY"  # Replace with your API key
            weather_data = get_weather(api_key, from_location)
            if weather_data:
                st.write(f"Temperature: {weather_data['main']['temp']}°C")
                st.write(f"Weather: {weather_data['weather'][0]['description']}")
            else:
                st.write("Weather data not found")
            
        with col_right:
            st.write("### Map")
            map_obj = create_map(from_location, to_location)
            if map_obj:
                folium_static(map_obj)
            else:
                st.write("Invalid locations provided.")
