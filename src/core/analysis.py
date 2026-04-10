import numpy as np
import cv2
from src.utils.image_io import numpy_to_base64
from src.utils.visualization import plot_histograms


def analyze(img: np.ndarray) -> dict:
    """
    Analyse initiale d'une image satellite RGB.

    Retourne :
    - histogrammes (base64)
    - statistiques par canal
    - ratio vert moyen
    - observations qualité
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("L'image doit être un tableau RGB de forme (H, W, 3).")

    stats = {}
    for i, canal in enumerate(["R", "G", "B"]):
        channel = img[:, :, i].astype(np.float32)
        stats[canal] = {
            "mean": float(channel.mean()),
            "std": float(channel.std()),
            "min": float(channel.min()),
            "max": float(channel.max()),
        }

    # Ratio vert
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    denom = R.astype(float) + G.astype(float) + B.astype(float)
    denom[denom == 0] = 1
    green_ratio = float((G.astype(float) / denom).mean())

    # Histogramme
    hist_img = plot_histograms(img)

    observations = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    brightness_mean = float(gray.mean())
    contrast_std = float(gray.std())
    p05, p95 = np.percentile(gray, [5, 95])
    dynamic_range = float(p95 - p05)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_std = float((gray - blurred).std())

    dark_ratio = float((gray < 40).mean() * 100)
    bright_ratio = float((gray > 215).mean() * 100)

    if brightness_mean < 75 or dark_ratio > 20:
        observations.append(
            "Image globalement sombre : certaines zones végétalisées peuvent être moins discriminantes."
        )
    elif brightness_mean > 180 or bright_ratio > 20:
        observations.append(
            "Image très lumineuse : risque de surexposition sur les surfaces les plus claires."
        )
    else:
        observations.append("Luminosité globalement équilibrée pour l'analyse.")

    if dynamic_range < 45 or contrast_std < 30:
        observations.append(
            "Contraste faible : les classes proches visuellement peuvent être plus difficiles à séparer."
        )
    elif dynamic_range > 170 and contrast_std > 60:
        observations.append("Contraste marqué : les grandes structures sont bien différenciées.")
    else:
        observations.append("Contraste intermédiaire, compatible avec une segmentation standard.")

    if noise_std > 18:
        observations.append(
            "Bruit visible détecté : un lissage peut améliorer la stabilité du clustering."
        )
    elif noise_std > 10:
        observations.append("Bruit modéré détecté : le prétraitement apportera un léger gain de robustesse.")
    else:
        observations.append("Niveau de bruit faible à modéré.")

    if bright_ratio > 5:
        observations.append(
            f"Présence de hautes lumières sur environ {bright_ratio:.1f}% des pixels."
        )
    if dark_ratio > 5:
        observations.append(
            f"Présence d'ombres marquées sur environ {dark_ratio:.1f}% des pixels."
        )

    return {
        "stats": stats,
        "green_ratio_mean": green_ratio,
        "histogram_base64": numpy_to_base64(hist_img),
        "observations": observations,
    }
