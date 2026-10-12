import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("--- [PARTE 1] INICIO DE EJECUCIÓN: MNIST ---")

# 1. Cargar y preprocesar el dataset MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# ----------------------------------------------------
# ACTIVIDAD ADICIONAL: Visualización de muestras del Dataset
# ----------------------------------------------------
print("\n--- Mostrando imágenes de ejemplo del dataset MNIST ---")
plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Etiqueta: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig("mnist_muestras.png")  # Evidencia para el README
plt.show()  # Se abrirá una ventana con la cuadrícula de imágenes de entrenamiento

# Normalizar y agregar dimensión de canal
X_train = X_train.astype("float32") / 255.0
X_train = np.expand_dims(X_train, -1)

X_test = X_test.astype("float32") / 255.0
X_test = np.expand_dims(X_test, -1)

# 2. Construcción de la Arquitectura CNN
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), name='Conv_1'),
    layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool_1'),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='Conv_2'),
    layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool_2'),
    layers.Flatten(name='Aplanado_Flatten'),
    layers.Dense(64, activation='relu', name='Capa_Densa_Oculta'),
    layers.Dense(10, activation='softmax', name='Salida_Softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Entrenamiento (1 época de demostración)
print("\n--- Entrenando el modelo MNIST (5000 muestras, 1 época) ---")
history = model.fit(X_train[:5000], y_train[:5000], epochs=1, batch_size=64)

# 4. Análisis y Predicción de una muestra de prueba
print("\n--- Analizando Resultados de Predicción (MNIST) ---")
imagen_prueba = X_test[0:1]
predicciones = model.predict(imagen_prueba)

print(f"Vector de probabilidades crudas (Softmax): \n{predicciones[0].round(4)}")
print(f"Suma de las 10 probabilidades: {predicciones[0].sum():.4f}")
print(f"Clase predicha por el modelo: {np.argmax(predicciones)}")
print(f"Clase real de la imagen: {y_test[0]}")
