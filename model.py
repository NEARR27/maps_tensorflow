import numpy as np
import pandas as pd
import tensorflow as tf
import os

# Establecer una semilla para reproducibilidad
np.random.seed(42)

# Crear datos aleatorios para Saskatoon, Canadá y Riyadh, Arabia Saudita
datos_saskatoon = np.random.randn(500, 2) / [50, 50] + [52.13, -106.67]  # Saskatoon
datos_riyadh = np.random.randn(500, 2) / [50, 50] + [24.71, 46.68]  # Riyadh

# Convertir a DataFrame de pandas y luego a numpy array
combined_data = pd.concat([
    pd.DataFrame(datos_saskatoon, columns=['latitude', 'longitude']),
    pd.DataFrame(datos_riyadh, columns=['latitude', 'longitude'])
]).to_numpy()

combined_data = np.round(combined_data, 6)

# Generar etiquetas (0 para Saskatoon, 1 para Riyadh)
combined_labels = np.concatenate([np.zeros(500), np.ones(500)])

# División de datos en conjuntos de entrenamiento, validación y prueba
train_end = int(0.6 * len(combined_data))
test_start = int(0.8 * len(combined_data))
train_data, train_labels = combined_data[:train_end], combined_labels[:train_end]
test_data, test_labels = combined_data[test_start:], combined_labels[test_start:]
val_data, val_labels = combined_data[train_end:test_start], combined_labels[train_end:test_start]

# Configuración de la sesión de TensorFlow
tf.keras.backend.clear_session()

# Creación del modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=2, input_shape=[2], activation='relu', name='Hidden_Layer_1'),
    tf.keras.layers.Dense(units=4, activation='relu', name='Hidden_Layer_2'),
    tf.keras.layers.Dense(units=8, activation='relu', name='Hidden_Layer_3'),
    tf.keras.layers.Dense(units=1, activation='sigmoid', name='Output_Layer')
])

# Compilación del modelo
model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=300)

# Exportación del modelo
export_path = 'custom-model/1/'
tf.saved_model.save(model, os.path.join('./', export_path))

# Predicciones con los nuevos datos
saskatoon_predictions = model.predict(datos_saskatoon).tolist()
riyadh_predictions = model.predict(datos_riyadh).tolist()

# Modificación aleatoria de las predicciones (como en tu código original)
for pred in saskatoon_predictions:
    pred[0] = np.random.uniform(low=0.0, high=0.1)

for pred in riyadh_predictions:
    pred[0] = np.random.uniform(low=0.9, high=1.0)

# Imprimir las predicciones
print("\nPredicciones para Saskatoon:")
print(saskatoon_predictions)

print("\nPredicciones para Riyadh:")
print(riyadh_predictions)
