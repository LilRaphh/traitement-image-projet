import numpy as np
import cv2


def compute_green_ratio(img: np.ndarray) -> np.ndarray:
    """Ratio vert : G / (R + G + B) par pixel."""
    R = img[:, :, 0].astype(float)
    G = img[:, :, 1].astype(float)
    B = img[:, :, 2].astype(float)
    denom = R + G + B
    denom[denom == 0] = 1
    return G / denom


def compute_hsv(img: np.ndarray) -> np.ndarray:
    """Convertit l'image RGB en HSV."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def compute_local_mean(img_gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Moyenne locale par convolution."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    return cv2.filter2D(img_gray.astype(np.float32), -1, kernel)


def compute_local_variance(img_gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Variance locale : E[X²] - E[X]²."""
    img_f = img_gray.astype(np.float32)
    mean = compute_local_mean(img_f, kernel_size)
    mean_sq = compute_local_mean(img_f ** 2, kernel_size)
    variance = mean_sq - mean ** 2
    return np.clip(variance, 0, None)


def compute_exg(img: np.ndarray) -> np.ndarray:
    """Excess Green index : 2G - R - B, normalisé dans [-1, 1]."""
    R = img[:, :, 0].astype(np.float32) / 255.0
    G = img[:, :, 1].astype(np.float32) / 255.0
    B = img[:, :, 2].astype(np.float32) / 255.0
    exg = 2 * G - R - B          # range [-2, 2]
    return exg / 2.0              # normalise dans [-1, 1]


def compute_vari(img: np.ndarray) -> np.ndarray:
    """VARI (Visible Atmospherically Resistant Index) : (G-R)/(G+R-B)."""
    R = img[:, :, 0].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    B = img[:, :, 2].astype(np.float32)
    denom = G + R - B
    denom[np.abs(denom) < 1e-3] = 1e-3   # évite la division par zéro
    vari = (G - R) / denom
    return np.clip(vari, -1.0, 1.0)


def build_feature_matrix(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Construit la matrice de features pour chaque pixel.

    Features incluses :
    - R, G, B normalisés
    - H, S, V normalisés
    - Ratio vert  (G / R+G+B)
    - ExG         (2G - R - B) — discriminant végétation vs sol/bâti
    - VARI        ((G-R)/(G+R-B)) — robuste aux variations d'éclairage
    - Moyenne locale (canal vert)
    - Variance locale (canal vert)

    Retourne un tableau (H*W, n_features).
    """
    H, W = img.shape[:2]

    # RGB normalisé
    rgb = img.astype(np.float32) / 255.0

    # HSV normalisé
    hsv = compute_hsv(img).astype(np.float32)
    hsv[:, :, 0] /= 179.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0

    # Indices spectraux
    green_ratio = compute_green_ratio(img)
    exg         = compute_exg(img)
    vari        = compute_vari(img)

    # Features locales sur le canal vert
    green_channel = img[:, :, 1].astype(np.float32)
    local_mean = compute_local_mean(green_channel, kernel_size) / 255.0
    local_var  = compute_local_variance(green_channel, kernel_size)
    local_var  = local_var / (local_var.max() + 1e-8)

    features = np.stack([
        rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2],
        hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2],
        green_ratio,
        exg,
        vari,
        local_mean,
        local_var,
    ], axis=-1)

    return features.reshape(H * W, -1)
