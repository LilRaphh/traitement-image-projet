import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from PIL import Image


def plot_histograms(img: np.ndarray) -> np.ndarray:
    """Génère un histogramme RGB et retourne l'image en numpy."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ("red", "green", "blue")
    labels = ("Rouge", "Vert", "Bleu")

    for i, (color, label) in enumerate(zip(colors, labels)):
        axes[i].hist(img[:, :, i].ravel(), bins=256, color=color, alpha=0.7)
        axes[i].set_title(f"Canal {label}")
        axes[i].set_xlim([0, 256])

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf))


def plot_deforestation_map(mask_t0: np.ndarray, mask_t1: np.ndarray) -> np.ndarray:
    """
    Génère une carte de déforestation colorée :
    - Vert  : végétation conservée
    - Rouge : végétation perdue (déforestation)
    - Gris  : non-végétation
    """
    h, w = mask_t0.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    vegetation_conserved = (mask_t0 == 1) & (mask_t1 == 1)
    deforested = (mask_t0 == 1) & (mask_t1 == 0)
    non_vegetation = mask_t0 == 0

    output[vegetation_conserved] = [34, 139, 34]   # vert forêt
    output[deforested] = [220, 50, 50]              # rouge
    output[non_vegetation] = [180, 180, 180]        # gris

    return output
