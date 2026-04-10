from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from src.utils.image_io import upload_to_numpy
from src.core import analysis

router = APIRouter()


@router.post("/")
async def analyze_image(image: UploadFile = File(...)):
    """
    Analyse initiale d'une image satellite.

    Retourne :
    - histogrammes par canal (R, G, B)
    - statistiques de base (mean, std, min, max)
    - ratio vert moyen
    - observations sur la qualité (bruit, contraste, luminosité)
    """
    img = await upload_to_numpy(image)

    result = analysis.analyze(img)

    return JSONResponse(content=result)
