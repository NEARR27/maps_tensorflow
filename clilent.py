import streamlit as st
import requests

SERVER_URL = 'http://localhost:8501/v1/models/custom-model:predict'

def get_predictions(inputs):
    predict_request = {'instances': inputs}
    response = requests.post(SERVER_URL, json=predict_request)
    
    if response.status_code == 200:
        prediction = response.json()
        return prediction
    else:
        st.error("Error al obtener predicciones. Por favor, verifica tus entradas e intenta de nuevo.")
        return None

def main():
    st.title('Predictor de Ubicaciones Geográficas')

    st.header('Coordenadas para Saskatoon, Canadá')
    saskatoon_lat = st.number_input('Ingrese la latitud de Saskatoon:', value=52.13)
    saskatoon_lon = st.number_input('Ingrese la longitud de Saskatoon:', value=-106.67)

    st.header('Coordenadas para Riyadh, Arabia Saudita')
    riyadh_lat = st.number_input('Ingrese la latitud de Riyadh:', value=24.71)
    riyadh_lon = st.number_input('Ingrese la longitud de Riyadh:', value=46.68)

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
