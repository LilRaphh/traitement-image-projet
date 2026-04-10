from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from src.utils.image_io import upload_to_numpy, numpy_to_base64
from src.core import comparison

router = APIRouter()


@router.post("/")
async def compare_images(
    image_t0: UploadFile = File(..., description="Image avant (t0)"),
    image_t1: UploadFile = File(..., description="Image après (t1)"),
):
    """
    Comparaison de deux images et quantification de la déforestation.

    Retourne :
    - carte de déforestation (base64)
    - surface végétalisée t0 (pixels + %)
    - surface végétalisée t1 (pixels + %)
    - perte relative (%)
    - classification métier de l'évolution
    - interprétation textuelle
    """
    img_t0 = await upload_to_numpy(image_t0)
    img_t1 = await upload_to_numpy(image_t1)

    result = comparison.compare(img_t0, img_t1)

    return JSONResponse(content={
        "deforestation_map_base64": numpy_to_base64(result["map"]),
        "area_t0": result["area_t0"],
        "area_t1": result["area_t1"],
        "loss_percentage": result["loss_percentage"],
        "classification": result["classification"],
        "interpretation": result["interpretation"],
    })
