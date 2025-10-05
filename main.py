"""
Visualization of the Fashion MNIST image data using tmap.
"""

import base64
from io import BytesIO
from timeit import default_timer as timer

import numpy as np
import tmap as tm
from faerun import Faerun
from PIL import Image

from utils import mnist_reader

# Coniguration for the tmap layout
CFG = tm.LayoutConfiguration()
CFG.node_size = 1 / 55

LABELS_MAP = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}

# Load fashion mnist data
IMAGES, LABELS, CLUSTERS = mnist_reader.load_mnist(
    "data", kind="escuela")

IMAGE_LABELS = []


def main():
    """ Main function """

    # Initialize and configure tmap
    dims = 1024
    enc = tm.Minhash(28 * 28, 42, dims)
    lf = tm.LSHForest(dims * 2, 128)
    
    unique_pairs = sorted(list(set(zip(LABELS, CLUSTERS))))
    pair_to_id = {pair: i for i, pair in enumerate(unique_pairs)}
    color_values = np.array([pair_to_id[(l, c)] for l, c in zip(LABELS, CLUSTERS)])
    hover_labels = [f"{LABELS_MAP[label]} tipo {cluster}" for label, cluster in zip(LABELS, CLUSTERS)]
    legend_tuples = []
    for pair, num_id in pair_to_id.items():
        label_id, cluster_id = pair
        text_label = f"{LABELS_MAP[label_id]} tipo {cluster_id}"
        legend_tuples.append((num_id, text_label))


    print("Converting images ...")
    for image in IMAGES:
        img = Image.fromarray(np.uint8(np.split(np.array(image), 28)))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        IMAGE_LABELS.append(
            "data:image/bmp;base64," + str(img_str).replace("b'", "").replace("'", "")
        )
    tmp = []
    for _, image in enumerate(IMAGES):
        tmp.append(tm.VectorFloat(image / 255))

    print("Running tmap ...")
    start = timer()
    lf.batch_add(enc.batch_from_weight_array(tmp))
    lf.index()
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, CFG)
    print("tmap: " + str(timer() - start))

    faerun = Faerun(clear_color="#222222", view="front", coords=False, legend_title="Fashion MNIST Subtipos")
    faerun.add_scatter(
        "FMNIST",
        {"x": x, "y": y, "c": color_values, "labels": IMAGE_LABELS},
        colormap="turbo",
        shader="smoothCircle",
        point_scale=2.5,
        max_point_size=10,
        has_legend=True,
        categorical=False,
        legend_labels=legend_tuples,
    )
    faerun.add_tree(
        "FMNIST_tree", {"from": s, "to": t}, point_helper="FMNIST", color="#666666"
    )
    faerun.plot("fmnist", path="output", template="url_image")


if __name__ == "__main__":
    main()