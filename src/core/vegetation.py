import numpy as np
import cv2
from src.core.features import compute_green_ratio, compute_exg, compute_vari


def identify_vegetation_cluster(img: np.ndarray, labels: np.ndarray) -> tuple[int, list[dict]]:
    """
    Identifie automatiquement le cluster correspondant à la végétation.

    Score multi-critères (moyenne normalisée de 3 indices) :
    - Ratio vert  : G / (R+G+B)
    - ExG         : 2G - R - B  (discriminant végétation vs sol/bâti)
    - VARI        : (G-R)/(G+R-B)  (robuste aux variations d'éclairage)

    Le cluster avec le score composite le plus élevé est retenu.

    Retourne :
    - index du cluster végétation
    - liste d'infos par cluster
    """
    green_ratio = compute_green_ratio(img).reshape(-1)
    exg = compute_exg(img).reshape(-1)
    vari = compute_vari(img).reshape(-1)
    rgb = img.reshape(-1, 3).astype(np.float32)

    # Teinte HSV : végétation ~ H ∈ [35°, 85°] → [25, 60] dans OpenCV (0-179)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].reshape(-1).astype(np.float32)
    sat = hsv[:, :, 1].reshape(-1).astype(np.float32) / 255.0
    val = hsv[:, :, 2].reshape(-1).astype(np.float32) / 255.0
    # Score de teinte : 1 si H ∈ [25, 60], décroissant hors de cette plage
    hue_score = np.clip(1.0 - np.abs(hue - 42.5) / 30.0, 0.0, 1.0)

    k = int(labels.max()) + 1
    clusters_info = []
    scores = []

    for i in range(k):
        mask = labels == i
        n = int(mask.sum())
        if n == 0:
            clusters_info.append({
                "cluster_id": i,
                "size": 0,
                "green_ratio_mean": 0.0,
                "exg_mean": 0.0,
                "vari_mean": 0.0,
                "hue_mean": 0.0,
                "saturation_mean": 0.0,
                "value_mean": 0.0,
                "green_dominance_mean": 0.0,
                "veg_score": 0.0,
            })
            scores.append(float("-inf"))
            continue

        cluster_rgb = rgb[mask]
        mean_gr = float(green_ratio[mask].mean())
        mean_exg = float(exg[mask].mean())
        mean_vari = float(vari[mask].mean())
        mean_hue = float(hue_score[mask].mean())
        mean_sat = float(sat[mask].mean())
        mean_val = float(val[mask].mean())
        green_dominance = float(
            ((cluster_rgb[:, 1] > cluster_rgb[:, 0]) & (cluster_rgb[:, 1] > cluster_rgb[:, 2])).mean()
        )
        base_score = 0.30 * mean_gr + 0.35 * mean_exg + 0.25 * mean_vari + 0.10 * mean_hue

        penalty = 0.0
        if mean_sat < 0.12:
            penalty += 0.10
        if mean_val < 0.20:
            penalty += 0.12
        if green_dominance < 0.35:
            penalty += 0.08
        if mean_hue < 0.12:
            penalty += 0.08

        score = base_score - penalty

        clusters_info.append({
            "cluster_id": i,
            "size": n,
            "green_ratio_mean": round(mean_gr, 4),
            "exg_mean": round(mean_exg, 4),
            "vari_mean": round(mean_vari, 4),
            "hue_mean": round(mean_hue, 4),
            "saturation_mean": round(mean_sat, 4),
            "value_mean": round(mean_val, 4),
            "green_dominance_mean": round(green_dominance, 4),
            "veg_score": round(float(score), 4),
        })
        scores.append(score)

    vegetation_cluster = int(np.argmax(scores))
    return vegetation_cluster, clusters_info
