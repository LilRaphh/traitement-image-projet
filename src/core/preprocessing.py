import numpy as np
import cv2
from skimage.exposure import match_histograms


def harmonize_colorimetry(img_ref: np.ndarray, img_target: np.ndarray) -> np.ndarray:
    """
    Aligne la colorimétrie de img_target sur img_ref par correspondance
    d'histogrammes canal par canal (R, G, B).

    Corrige les différences d'éclairage, de saison ou de capteur entre t0 et t1
    avant segmentation, pour que le clustering soit fait dans le même espace colorimétrique.

    Retourne img_target recalée (uint8).
    """
    matched = match_histograms(img_target, img_ref, channel_axis=-1)
    return matched.astype(np.uint8)


def preprocess(img: np.ndarray, apply_gamma: bool = True) -> tuple[np.ndarray, list[dict]]:
    """
    Prétraitement de l'image en fonction des difficultés observées.

    Paramètres :
    - apply_gamma : applique la correction gamma (γ=2.0) pour compresser les hautes lumières.
      Utile pour t0 (souvent surexposée). À désactiver pour t1 qui a déjà été harmonisée
      sur t0, afin d'éviter un assombrissement inutile.

    Retourne :
    - image prétraitée (numpy uint8)
    - liste des étapes appliquées avec justification
    """
    steps = []
    result = img.copy()

    if apply_gamma:
        # Correction gamma (γ = 2.0) : assombrit les zones surexposées en compressant
        # les hautes lumières sans écrêter (contrairement à une soustraction linéaire).
        # Formule : out = (in / 255)^γ × 255
        gamma = 2.0
        lut = (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        result = cv2.LUT(result, lut)
        steps.append({
            "step": f"gamma_correction_gamma={gamma}",
            "justification": "correction des images surexposées : compression des hautes lumières sans écrêtage"
        })


    # Flou gaussien 5×5 : réduit le bruit pixel à pixel avant clustering
    result = cv2.GaussianBlur(result, (5, 5), 0)
    steps.append({
        "step": "gaussian_blur_5x5",
        "justification": "lissage du bruit pour homogénéiser les régions avant extraction de features"
    })

    return result, steps
