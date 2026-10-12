import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("--- [PARTE 2] INICIO DE EJECUCIÓN: FASHION-MNIST ---")

# 1. Cargar y preprocesar Fashion-MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

class_names = ['Camiseta/Top', 'Pantalón', 'Pulóver', 'Vestido', 'Abrigo',
               'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']

# ----------------------------------------------------
# ACTIVIDAD ADICIONAL: Visualización de muestras del Dataset
# ----------------------------------------------------
print("\n--- Mostrando imágenes de ejemplo del dataset Fashion-MNIST ---")
plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"{class_names[y_train[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig("fashion_muestras.png")  # Evidencia para el README
plt.show()  # Se abrirá una ventana con la cuadrícula de prendas de vestir

X_train = X_train.astype("float32") / 255.0
X_train = np.expand_dims(X_train, -1)

X_test = X_test.astype("float32") / 255.0
X_test = np.expand_dims(X_test, -1)

# 2. Construcción de la Arquitectura CNN (con capa densa ampliada)
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), name='Conv_1'),
    layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool_1'),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='Conv_2'),
    layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool_2'),
    layers.Flatten(name='Aplanado_Flatten'),
    layers.Dense(128, activation='relu', name='Capa_Densa_Oculta'),  # Capa ampliada a 128
    layers.Dense(10, activation='softmax', name='Salida_Softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Entrenamiento (1 época de demostración)
print("\n--- Entrenando el modelo Fashion-MNIST (5000 muestras, 1 época) ---")
history = model.fit(X_train[:5000], y_train[:5000], epochs=1, batch_size=64)

# 4. Análisis y Predicción de una muestra de prueba
print("\n--- Analizando Resultados de Predicción (Fashion-MNIST) ---")
imagen_prueba = X_test[0:1]
predicciones = model.predict(imagen_prueba)
indice_predicho = np.argmax(predicciones)

print(f"Vector de probabilidades crudas (Softmax): \n{predicciones[0].round(4)}")
print(f"Suma de las 10 probabilidades: {predicciones[0].sum():.4f}")
print(f"Clase predicha: {class_names[indice_predicho]} (Índice {indice_predicho})")
# Nota: la guía tiene una llave '}' de más en esta línea, lo que da SyntaxError; aquí está corregida.
print(f"Clase real: {class_names[y_test[0]]} (Índice {y_test[0]})")
