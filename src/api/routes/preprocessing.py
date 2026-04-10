from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from src.utils.image_io import upload_to_numpy, numpy_to_base64
from src.core import preprocessing

router = APIRouter()


@router.post("/")
async def preprocess_image(image: UploadFile = File(...), apply_gamma: bool = True):
    """
    Prétraitement de l'image.

    Paramètres :
    - apply_gamma : active la correction gamma (γ=2.0). Mettre à false pour t1
      (déjà harmonisée) afin d'éviter un assombrissement inutile.

    Retourne :
    - image prétraitée encodée en base64 (PNG)
    - liste des étapes appliquées et leur justification
    """
    img = await upload_to_numpy(image)

    processed, steps = preprocessing.preprocess(img, apply_gamma=apply_gamma)

    return JSONResponse(content={
        "image_base64": numpy_to_base64(processed),
        "steps_applied": steps,
    })
