def load_mnist(path, kind='escuela'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                                '%s-labels-idx1-ubyte.gz'
                                % kind)
    images_path = os.path.join(path,
                                '%s-images-idx3-ubyte.gz'
                                % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
    # Saltar el número mágico (primeros 4 bytes)
        lbpath.read(4)
        
        # Leer el número de etiquetas principales y luego los datos de las etiquetas
        num_labels = int.from_bytes(lbpath.read(4), 'big')
        labels = np.frombuffer(lbpath.read(num_labels), dtype=np.uint8)
        
        # Leer el número de clusters y luego los datos de los clusters
        num_clusters = int.from_bytes(lbpath.read(4), 'big')
        clusters = np.frombuffer(lbpath.read(num_clusters), dtype=np.uint8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                offset=16).reshape(len(labels), 784)

    return images, labels, clusters
