import numpy as np


def quantify(mask_t0: np.ndarray, mask_t1: np.ndarray) -> dict:
    """
    Quantifie la déforestation entre t0 et t1.

    Formule :
        perte = (surface_t0 - surface_t1) / surface_t0

    Retourne :
    - surface végétalisée t0 (pixels + %)
    - surface végétalisée t1 (pixels + %)
    - perte relative (%)
    """
    total_pixels = mask_t0.size

    area_t0_px = int(mask_t0.sum())
    area_t1_px = int(mask_t1.sum())

    area_t0_pct = round(area_t0_px / total_pixels * 100, 2)
    area_t1_pct = round(area_t1_px / total_pixels * 100, 2)

    if area_t0_px > 0:
        loss = round((area_t0_px - area_t1_px) / area_t0_px * 100, 2)
    else:
        loss = 0.0

    return {
        "area_t0": {"pixels": area_t0_px, "percentage": area_t0_pct},
        "area_t1": {"pixels": area_t1_px, "percentage": area_t1_pct},
        "loss_percentage": loss,
    }
