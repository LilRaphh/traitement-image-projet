import numpy as np
from src.core.features import build_feature_matrix
from src.core.clustering import choose_k_with_cah, apply_kmeans
from src.core.vegetation import identify_vegetation_cluster
from src.core.postprocessing import postprocess_mask


def segment(img: np.ndarray, k: int = None, img_raw: np.ndarray = None) -> dict:
    """
    Pipeline de segmentation complet :
    1. Extraction des features
    2. Choix de k (CAH) si non fourni
    3. K-means
    4. Identification automatique végétation
    5. Post-traitement morphologique

    Paramètres :
    - img     : image prétraitée utilisée pour le clustering (features)
    - k       : nombre de clusters (estimé par CAH si None)
    - img_raw : image originale (non prétraitée) pour le calcul des indices de végétation.
                Si None, on utilise img. À fournir quand img a subi une correction gamma
                qui distord les ratios spectraux et peut inverser la détection végétation.

    Retourne :
    - segmented : image segmentée colorée (H, W, 3)
    - mask : masque végétation binaire (H, W)
    - k : nombre de clusters utilisé
    - clusters_info : infos par cluster
    - vegetation_cluster : index du cluster végétation
    """
    H, W = img.shape[:2]

    # 1. Features
    features = build_feature_matrix(img)

    # 2. Choix de k
    if k is None:
        k = choose_k_with_cah(features)

    # 3. K-means
    labels = apply_kmeans(features, k)

    # 4. Identification végétation — sur l'image brute si disponible pour éviter
    #    que le gamma ne distorde les indices spectraux (ExG, VARI, green_ratio)
    img_score = img_raw if img_raw is not None else img
    vegetation_cluster, clusters_info = identify_vegetation_cluster(img_score, labels)

    # 5. Image segmentée (couleurs aléatoires reproductibles par cluster)
    rng = np.random.default_rng(42)
    colors = rng.integers(0, 255, size=(k, 3), dtype=np.uint8)
    segmented = colors[labels].reshape(H, W, 3)

    # 6. Masque végétation + post-traitement
    raw_mask = (labels == vegetation_cluster).reshape(H, W).astype(np.uint8)
    mask = postprocess_mask(raw_mask)

    return {
        "segmented": segmented,
        "mask": mask,
        "k": k,
        "clusters_info": clusters_info,
        "vegetation_cluster": vegetation_cluster,
    }
