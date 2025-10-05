# convertir_a_gz.py
import pandas as pd
import numpy as np
import gzip
import os

def csv_a_mnist_gz(archivo_csv, directorio_salida):
    """
    Convierte un archivo CSV limpio al formato de archivos .gz de MNIST.
    """
    try:
        # Crear el directorio de salida si no existe
        if not os.path.exists(directorio_salida):
            os.makedirs(directorio_salida)
            print(f"Directorio '{directorio_salida}' creado.")

        # Cargar el dataset limpio
        df = pd.read_csv(archivo_csv)

        # La columna 'label' se usará para el archivo de etiquetas
        labels = df['label'].values.astype(np.uint8)
        sub_categorias = df['cluster'].values.astype(np.uint8)
        
        # Los datos de imagen son todas las columnas excepto 'label' y 'cluster'
        images = df.drop(columns=['label', 'cluster']).values.astype(np.uint8)

        # --- Crear el archivo de etiquetas .gz ---
        ruta_labels = os.path.join(directorio_salida, 'escuela-labels-idx1-ubyte.gz')
        with gzip.open(ruta_labels, 'wb') as lbpath:
            # Escribir la cabecera mágica de MNIST
            lbpath.write((2049).to_bytes(4, 'big'))
            lbpath.write(len(labels).to_bytes(4, 'big'))
            lbpath.write(labels.tobytes())
            lbpath.write(len(sub_categorias).to_bytes(4, 'big'))
            lbpath.write(sub_categorias.tobytes())
        print(f"Archivo de etiquetas guardado en: {ruta_labels}")

        # --- Crear el archivo de imágenes .gz ---
        ruta_images = os.path.join(directorio_salida, 'escuela-images-idx3-ubyte.gz')
        with gzip.open(ruta_images, 'wb') as imgpath:
            # Escribir la cabecera mágica de MNIST
            imgpath.write((2051).to_bytes(4, 'big'))
            imgpath.write(len(images).to_bytes(4, 'big'))
            imgpath.write((28).to_bytes(4, 'big')) # Filas de la imagen
            imgpath.write((28).to_bytes(4, 'big')) # Columnas de la imagen
            imgpath.write(images.tobytes())
        print(f"Archivo de imágenes guardado en: {ruta_images}")

    except FileNotFoundError:
        print(f"Error: El archivo '{archivo_csv}' no fue encontrado.")
    except Exception as e:
        print(f"Ha ocurrido un error: {e}")

# --- CONFIGURACIÓN ---
# 1. Ruta a tu archivo CSV ya sin duplicados.
archivo_csv_limpio = '../data/fashion_mnist_limpio.csv'

# 2. Directorio donde se guardarán los nuevos archivos .gz.
#    Debe ser la ruta que usa tu `main.py` en la función `load_mnist`.
directorio_final = '../data'

# --- Ejecutar la conversión ---
csv_a_mnist_gz(archivo_csv_limpio, directorio_final)