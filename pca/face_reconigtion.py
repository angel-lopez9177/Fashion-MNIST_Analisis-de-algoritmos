import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

GALLERY = r"C:\Users\alems\Downloads\ADA\pca\Gallery\Gallery"
PROBE = r"C:\Users\alems\Downloads\ADA\pca\Probe\Probe"

def load_images(path):
    images = []
    labels = []
    for folder in sorted(os.listdir(path)):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for file in sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0])):
                img_path = os.path.join(folder_path, file)
                img = np.array(Image.open(img_path), dtype='float32')
                images.append(img.flatten())
                labels.append(int(folder[1:])) 
    return np.array(images), np.array(labels)

print("Cargando imágenes...")
X_gallery, y_gallery = load_images(GALLERY)
X_probe, y_probe = load_images(PROBE)
print(f"✅ Gallery: {X_gallery.shape}, Probe: {X_probe.shape}")

print("Calculando Eigenfaces...")

mean_face = np.mean(X_gallery, axis=0)
X_centered = X_gallery - mean_face

C = np.dot(X_centered, X_centered.T) / X_centered.shape[0]

eigenvalues, eigenvectors = np.linalg.eigh(C)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

eigenfaces = np.dot(X_centered.T, eigenvectors)
eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)

print(f"Eigenfaces calculadas: {eigenfaces.shape[1]}")

def show_eigenfaces(eigenfaces, h, w, n=6):
    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(2, 3, i+1)
        plt.imshow(eigenfaces[:, i].reshape(h, w), cmap='gray')
        plt.title(f"Eigenface {i+1}")
        plt.axis('off')
    plt.suptitle("Top 6 Eigenfaces", fontsize=14)
    plt.show()

h, w = 112, 92 
show_eigenfaces(eigenfaces, h, w)

total_var = np.sum(eigenvalues)
var_ratio = np.cumsum(eigenvalues) / total_var

plt.figure()
plt.plot(np.arange(1, len(var_ratio)+1), var_ratio * 100)
plt.axhline(y=85, color='r', linestyle='--', label='85%')
plt.axhline(y=95, color='g', linestyle='--', label='95%')
plt.xlabel("Número de componentes")
plt.ylabel("% de varianza explicada")
plt.title("Varianza acumulada")
plt.legend()
plt.show()

d85 = np.argmax(var_ratio >= 0.85) + 1
d95 = np.argmax(var_ratio >= 0.95) + 1
print(f"Para 85% de varianza → {d85} componentes")
print(f"Para 95% de varianza → {d95} componentes")

def reconstruct_face(img, mean_face, eigenfaces, k):
    weights = np.dot(eigenfaces[:, :k].T, (img - mean_face))
    recon = mean_face + np.dot(eigenfaces[:, :k], weights)
    return recon

img = X_gallery[0]
ks = [1, 4, 15, 150, eigenfaces.shape[1]]

errors = []
plt.figure(figsize=(12, 6))
for i, k in enumerate(ks):
    recon = reconstruct_face(img, mean_face, eigenfaces, k)
    mse = np.mean((img - recon) ** 2)
    errors.append(mse)
    plt.subplot(2, 3, i+1)
    plt.imshow(recon.reshape(h, w), cmap='gray')
    plt.title(f"{k} Eigenfaces\nMSE={mse:.2f}")
    plt.axis('off')

plt.suptitle("Reconstrucción con diferentes Eigenfaces")
plt.show()

plt.figure()
plt.plot(ks, errors, marker='o')
plt.xlabel("Número de Eigenfaces")
plt.ylabel("MSE")
plt.title("Error de reconstrucción vs Eigenfaces")
plt.show()
