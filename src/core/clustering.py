import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage


def choose_k_with_cah(features: np.ndarray, max_k: int = 8, sample_size: int = 2000) -> int:
    """
    Choisit k via CAH (classification hiérarchique ascendante).

    Stratégie : on observe les hauteurs de fusion (distances) dans le dendrogramme
    et on choisit k à l'endroit où le saut est le plus grand.

    Retourne k recommandé.
    """
    # Échantillonnage
    if len(features) > sample_size:
        idx = np.random.choice(len(features), sample_size, replace=False)
        sample = features[idx]
    else:
        sample = features

    sample = np.nan_to_num(sample.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    upper_k = min(max_k, max(1, len(sample) - 1))
    if upper_k < 2:
        return 1

    try:
        # Les hauteurs de fusion du linkage Ward correspondent aux niveaux du dendrogramme.
        # Le meilleur k est estimé au plus grand saut de fusion entre k et k-1 clusters.
        z = linkage(sample, method="ward")
        heights = z[:, 2]

        jumps = []
        for k in range(2, upper_k + 1):
            jump = heights[-(k - 1)] - heights[-k]
            jumps.append(jump)

        k_opt = int(np.argmax(jumps)) + 2
        return k_opt
    except Exception:
        # Repli robuste si la CAH échoue sur un échantillon dégénéré.
        pass

    inertias = []
    for k in range(2, upper_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(sample)
        inertias.append(km.inertia_)

    if len(inertias) == 1:
        return 2

    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    if len(diffs2) == 0:
        return 2

    k_opt = int(np.argmax(diffs2)) + 3
    k_opt = max(2, min(k_opt, upper_k))

    return k_opt


def apply_kmeans(features: np.ndarray, k: int) -> np.ndarray:
    """
    Applique K-means sur la matrice de features.

    Retourne les labels (H*W,).
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(features)
    return labels
