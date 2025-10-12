import os
os.environ["OMP_NUM_THREADS"] = "4"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("fashion-mnist_test.csv")

label_focus = 8
category_names = {
    0: "Camiseta", 1: "Pantalon", 2: "Pullover", 3: "Vestido",
    4: "Abrigo", 5: "Sandalia", 6: "Camisa", 7: "Zapatilla deportiva",
    8: "Bolsa", 9: "Bota"
}

subset = data[data['label'] == label_focus].drop('label', axis=1)
print(f"Se seleccionaron {len(subset)} ejemplos de la categoría '{category_names[label_focus]}'")

scaler = StandardScaler()
subset_scaled = scaler.fit_transform(subset)

pca = PCA(n_components=50, random_state=42)
subset_pca = pca.fit_transform(subset_scaled)

num_clusters = 4 
print(f"Agrupando en {num_clusters} clusters...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
subset['cluster'] = kmeans.fit_predict(subset_pca)


for cluster_id in range(num_clusters):
    cluster_data = subset[subset['cluster'] == cluster_id]
    n_samples = min(9, len(cluster_data))  # evita pedir más de lo que hay
    cluster_imgs = cluster_data.sample(n_samples, random_state=42)
    
    imgs = cluster_imgs.drop('cluster', axis=1).to_numpy().reshape(-1, 28, 28)

    plt.figure(figsize=(6, 6))
    for i, img in enumerate(imgs):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Cluster {cluster_id}', fontsize=14)
    plt.show()

subset['sub_label'] = subset['cluster'].map({
    0: 'Camiseta tipo 1',
    1: 'Camiseta tipo 2',
    2: 'Camiseta tipo 3',
    3: 'Camiseta tipo 4'
})

output = f"fashionmnist_sub_{category_names[label_focus].replace(' ', '_')}.csv"
subset.to_csv(output, index=False)
print(f"\nArchivo guardado como: {output}")
