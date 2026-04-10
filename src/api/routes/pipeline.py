from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from src.utils.image_io import upload_to_numpy, numpy_to_base64
from src.core import analysis, preprocessing, segmentation, comparison

router = APIRouter()


@router.post("/")
async def run_pipeline(
    image_t0: UploadFile = File(..., description="Image avant (t0)"),
    image_t1: UploadFile = File(..., description="Image après (t1)"),
):
    """
    Pipeline complet : analyse → prétraitement → segmentation → comparaison.

    Retourne l'ensemble des résultats intermédiaires et finaux.
    """
    img_t0 = await upload_to_numpy(image_t0)
    img_t1 = await upload_to_numpy(image_t1)

    # 1. Analyse initiale
    stats_t0 = analysis.analyze(img_t0)
    stats_t1 = analysis.analyze(img_t1)

    # 2. Harmonisation colorimétrique : aligne t1 sur t0 avant tout traitement
    # (corrige les différences d'éclairage, de saison, de capteur)
    img_t1_harmonized = preprocessing.harmonize_colorimetry(img_t0, img_t1)

    # 3. Prétraitement individuel sur images déjà harmonisées.
    # t0 : gamma activé (compense la surexposition fréquente de t0).
    # t1 : gamma désactivé (l'harmonisation colorimétrique a déjà aligné les niveaux sur t0 ;
    #      appliquer en plus un gamma assombrirait inutilement t1).
    proc_t0, steps_t0 = preprocessing.preprocess(img_t0, apply_gamma=True)
    proc_t1, steps_t1 = preprocessing.preprocess(img_t1_harmonized, apply_gamma=False)
    steps_t1 = [{"step": "histogram_matching_to_t0",
                  "justification": "alignement colorimétrique de t1 sur t0 pour comparaison cohérente"}] + steps_t1

    # 4. Segmentation — on passe l'image brute (img_raw) pour le scoring végétation
    #    afin que les indices spectraux (ExG, VARI, green_ratio) ne soient pas
    #    distordus par la correction gamma appliquée à proc_t0/proc_t1.
    seg_t0 = segmentation.segment(proc_t0, img_raw=img_t0)
    seg_t1 = segmentation.segment(proc_t1, img_raw=img_t1)

    # 5. Comparaison
    result = comparison.compare_masks(seg_t0["mask"], seg_t1["mask"])

    return JSONResponse(content={
        "analysis": {"t0": stats_t0, "t1": stats_t1},
        "preprocessing": {"steps_t0": steps_t0, "steps_t1": steps_t1},
        "segmentation": {
            "t0": {
                "segmented_image_base64": numpy_to_base64(seg_t0["segmented"]),
                "vegetation_mask_base64": numpy_to_base64(seg_t0["mask"]),
                "k": seg_t0["k"],
                "clusters_info": seg_t0["clusters_info"],
                "vegetation_cluster_index": seg_t0["vegetation_cluster"],
            },
            "t1": {
                "segmented_image_base64": numpy_to_base64(seg_t1["segmented"]),
                "vegetation_mask_base64": numpy_to_base64(seg_t1["mask"]),
                "k": seg_t1["k"],
                "clusters_info": seg_t1["clusters_info"],
                "vegetation_cluster_index": seg_t1["vegetation_cluster"],
            },
        },
        "comparison": {
            "deforestation_map_base64": numpy_to_base64(result["map"]),
            "area_t0": result["area_t0"],
            "area_t1": result["area_t1"],
            "loss_percentage": result["loss_percentage"],
            "classification": result["classification"],
            "interpretation": result["interpretation"],
        },
    })
