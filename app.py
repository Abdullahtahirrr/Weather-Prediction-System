import numpy as np
import pickle
import streamlit as st
import sklearn
from PIL import Image

# loading the saved model
mlModel = pickle.load(open("C:/Users/Dipen Patel/Desktop/Project/mlModel.sav", 'rb'))

def predictor(input_data):    
    input_array = np.asarray(input_data)
    reshaped_input = input_array.reshape(1, -1)
    predicted_value = mlModel.predict(reshaped_input)
    return predicted_value

def main():
    st.title('Cloud Sense: Weather Temperature Predictor')
    
    # Load weather-related images
    weather_icons = {
        "sun": Image.open("C:/Users/Dipen Patel/Desktop/Project/sun.png"),
        "cloud": Image.open("C:/Users/Dipen Patel/Desktop/Project/cloud.png"),
        # Add more icons for different weather conditions
    }
    
    st.sidebar.title("Input Parameters")
    
    st.sidebar.write("**Temperature**")
    MaxTemp = st.sidebar.number_input('Max Temp (°C)', help='Maximum Temperature')
    MinTemp = st.sidebar.number_input('Min Temp (°C)', help='Minimum Temperature')
    
    st.sidebar.write("**Weather Conditions**")
    cloudCover = st.sidebar.number_input('Cloud Cover (%)', help='Cloud Cover')
    humidity = st.sidebar.number_input('Humidity (%)', help='Humidity')
    
    st.sidebar.write("**Other Factors**")
    sunHour = st.sidebar.number_input('Sun Hours', help='Sun Hours')
    HeatIndex = st.sidebar.number_input('Heat Index', help='Heat Index')
    
    st.sidebar.write("**Precipitation and Pressure**")
    precip = st.sidebar.number_input('Precipitation', help='Precipitation Rate')
    pressure = st.sidebar.number_input('Pressure (hPa)', help='Pressure')
    
    windSpeed = st.sidebar.number_input('Wind Speed (km/h)', help='Wind Speed')

    predict_button = st.sidebar.button('Predict')
  
    weather = ''

    if predict_button:
        weather = predictor([MaxTemp, MinTemp, cloudCover, humidity, sunHour, HeatIndex, precip, pressure, windSpeed])
    
    st.markdown("---")
    
    if weather:
        weather_condition = "sun" if weather[0] > 20 else "cloud"  # Example logic, you can modify this
        st.image(weather_icons[weather_condition], width=100)
        st.success(f"The forecasted Temperature for the day is : {weather[0]:.2f} °C")

if __name__ == '__main__':
    main()
