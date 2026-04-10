import numpy as np
import cv2


def postprocess_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Nettoyage morphologique du masque de végétation.

    Étapes :
    1. Ouverture  : supprime les petits bruits isolés
    2. Fermeture  : comble les petits trous dans les zones végétalisées
    3. Filtrage des composantes connexes : supprime les îlots < 0.1 % de l'image

    Retourne le masque nettoyé (uint8 : 0 ou 1).
    """
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = mask.astype(np.uint8)

    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,  kernel)
    closed = cv2.morphologyEx(opened,  cv2.MORPH_CLOSE, kernel)

    # Suppression des petites composantes connexes (bruit résiduel)
    n_pixels = closed.size
    min_area = max(50, int(n_pixels * 0.001))   # seuil : 0.1 % de l'image, min 50 px

    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    filtered = np.zeros_like(closed)
    for lbl in range(1, num_labels):            # 0 = fond
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels_cc == lbl] = 1

    return filtered
