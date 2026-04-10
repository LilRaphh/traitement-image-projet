import numpy as np
from scipy import ndimage
from src.core.classification import classify_deforestation
from src.core.segmentation import segment
from src.core.quantification import quantify
from src.utils.visualization import plot_deforestation_map


def compare(img_t0: np.ndarray, img_t1: np.ndarray) -> dict:
    """Compare deux images brutes en passant par le pipeline de segmentation."""
    seg_t0 = segment(img_t0)
    seg_t1 = segment(img_t1)
    return compare_masks(seg_t0["mask"], seg_t1["mask"])


def compare_masks(mask_t0: np.ndarray, mask_t1: np.ndarray) -> dict:
    """
    Compare deux masques de végétation et produit la carte de déforestation.

    Retourne :
    - map : carte colorée (numpy H, W, 3)
    - area_t0, area_t1 : surfaces
    - loss_percentage : perte relative
    - interpretation : texte automatique
    """
    if mask_t0.shape != mask_t1.shape:
        raise ValueError("Les deux masques doivent avoir la même taille.")

    quant = quantify(mask_t0, mask_t1)
    deforestation_map = plot_deforestation_map(mask_t0, mask_t1)

    loss = quant["loss_percentage"]
    deforested = (mask_t0 == 1) & (mask_t1 == 0)
    regrowth = (mask_t0 == 0) & (mask_t1 == 1)

    if loss > 20:
        interpretation_parts = [
            f"Déforestation significative détectée : {loss:.1f}% de perte de couverture végétale."
        ]
    elif loss > 5:
        interpretation_parts = [f"Déforestation modérée détectée : {loss:.1f}% de perte."]
    elif loss > 0:
        interpretation_parts = [f"Légère réduction de la végétation : {loss:.1f}% de perte."]
    else:
        interpretation_parts = ["Aucune déforestation significative détectée."]

    focus_mask = deforested if deforested.any() else regrowth
    if focus_mask.any():
        coords = np.argwhere(focus_mask)
        y_mean, x_mean = coords.mean(axis=0)
        h, w = focus_mask.shape

        vertical = "nord" if y_mean < h / 3 else "centre" if y_mean < 2 * h / 3 else "sud"
        horizontal = "ouest" if x_mean < w / 3 else "centre" if x_mean < 2 * w / 3 else "est"

        labeled, n_components = ndimage.label(focus_mask.astype(np.uint8))
        if n_components > 0:
            component_sizes = ndimage.sum(
                focus_mask.astype(np.uint8), labeled, index=np.arange(1, n_components + 1)
            )
            largest_share = float(component_sizes.max() / focus_mask.sum() * 100)
        else:
            largest_share = 0.0

        phenomenon = "Les pertes" if deforested.any() else "Les gains"
        interpretation_parts.append(
            f"{phenomenon} se concentrent principalement dans la zone {vertical}-{horizontal} de l'image."
        )
        if n_components >= 4 and largest_share < 50:
            interpretation_parts.append("Le phénomène est réparti en plusieurs foyers diffus.")
        elif largest_share > 60:
            interpretation_parts.append(
                f"Un foyer principal domine, avec environ {largest_share:.1f}% de la zone concernée."
            )

    interpretation_parts.append(
        "Cette interprétation repose sur des masques issus d'une segmentation RGB et peut être sensible "
        "aux différences d'éclairage, de saison, d'alignement ou de résolution entre les deux images."
    )

    interpretation = " ".join(interpretation_parts)

    classification = classify_deforestation(
        area_t0=quant["area_t0"],
        area_t1=quant["area_t1"],
        loss_percentage=loss,
    )

    return {
        "map": deforestation_map,
        "area_t0": quant["area_t0"],
        "area_t1": quant["area_t1"],
        "loss_percentage": loss,
        "interpretation": interpretation,
        "classification": classification,
    }
