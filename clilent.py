import pandas as pd
import numpy as np
import streamlit as st
import requests

# Establecer una semilla para reproducibilidad
np.random.seed(42)

# Función para asignar colores basados en cuadrantes
def asignar_color(df, lat_media, lon_media):
    condiciones = [
        (df['lat'] >= lat_media) & (df['lon'] >= lon_media),
        (df['lat'] >= lat_media) & (df['lon'] < lon_media),
        (df['lat'] < lat_media) & (df['lon'] >= lon_media),
        (df['lat'] < lat_media) & (df['lon'] < lon_media),
    ]
    colores = ['blue', 'green', 'red', 'purple']
    df['color'] = np.select(condiciones, colores)

# Crear datos aleatorios para Saskatoon, Canadá y Riyadh, Arabia Saudita
datos_saskatoon = np.random.randn(500, 2) / [50, 50] + [52.13, -106.67]  # Saskatoon
datos_riyadh = np.random.randn(500, 2) / [50, 50] + [24.71, 46.68]  # Riyadh

# Convertir a DataFrame de pandas
map_data_combined = pd.concat([
    pd.DataFrame(datos_saskatoon, columns=['lat', 'lon']),
    pd.DataFrame(datos_riyadh, columns=['lat', 'lon'])
])

lat_media = map_data_combined['lat'].mean()
lon_media = map_data_combined['lon'].mean()

# Asignar colores basados en cuadrantes
asignar_color(map_data_combined, lat_media, lon_media)

# Función para obtener predicciones de la API
def get_predictions(inputs):
    SERVER_URL = 'https://custom-model-service-nearr27.cloud.okteto.net/v1/models/custom-model:predict'
    predict_request = {'instances': inputs}
    response = requests.post(SERVER_URL, json=predict_request)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error al obtener predicciones. Por favor, verifica tus entradas e intenta de nuevo.")
        return None

# Interfaz de usuario en Streamlit
def main():
    st.title("Mapa combinado de Saskatoon y Riyadh con Cuadrantes")
    st.header("Usando Streamlit y Mapbox")

    # Mostrar el mapa
    st.map(map_data_combined.assign(color=map_data_combined['color']))

    st.title('Predictor de Ubicaciones Geográficas')

    # Coordenadas para Saskatoon
    st.header('Coordenadas para Saskatoon, Canadá')
    saskatoon_lat = st.number_input('Ingrese la latitud de Saskatoon:', value=52.13)
    saskatoon_lon = st.number_input('Ingrese la longitud de Saskatoon:', value=-106.67)

    # Coordenadas para Riyadh
    st.header('Coordenadas para Riyadh, Arabia Saudita')
    riyadh_lat = st.number_input('Ingrese la latitud de Riyadh:', value=24.71)
    riyadh_lon = st.number_input('Ingrese la longitud de Riyadh:', value=46.68)

    # Botón para realizar predicciones
    if st.button('Predecir'):
        inputs = [
            [saskatoon_lon, saskatoon_lat],
            [riyadh_lon, riyadh_lat]
        ]
        predictions = get_predictions(inputs)

        if predictions:
            st.write("\nPredicciones para Saskatoon:")
            st.write(predictions['predictions'][0])

            st.write("\nPredicciones para Riyadh:")
            st.write(predictions['predictions'][1])

if __name__ == '__main__':
    main()
