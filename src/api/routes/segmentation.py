from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from src.utils.image_io import upload_to_numpy, numpy_to_base64
from src.core import segmentation

router = APIRouter()


@router.post("/")
async def segment_image(
    image: UploadFile = File(...),
    k: int = Form(default=None, description="Nombre de clusters (None = auto via CAH)"),
):
    """
    Segmentation K-means + identification automatique de la végétation.

    Retourne :
    - image segmentée (base64)
    - masque végétation binaire (base64)
    - k retenu
    - informations par cluster (ratio vert moyen, taille)
    - index du cluster végétation identifié automatiquement
    """
    img = await upload_to_numpy(image)

    result = segmentation.segment(img, k=k)

    return JSONResponse(content={
        "segmented_image_base64": numpy_to_base64(result["segmented"]),
        "vegetation_mask_base64": numpy_to_base64(result["mask"]),
        "k": result["k"],
        "clusters_info": result["clusters_info"],
        "vegetation_cluster_index": result["vegetation_cluster"],
    })
