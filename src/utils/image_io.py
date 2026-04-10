import base64
import io
import numpy as np
from PIL import Image
from fastapi import UploadFile


async def upload_to_numpy(upload: UploadFile) -> np.ndarray:
    """Convertit un UploadFile en tableau numpy RGB."""
    contents = await upload.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return np.array(image)


def numpy_to_base64(img: np.ndarray, fmt: str = "PNG") -> str:
    """Encode un tableau numpy en image base64."""
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    elif img.ndim == 2 and img.max() <= 1:
        # Masque binaire 0/1 — scale vers 0/255 pour un affichage correct
        img = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def numpy_to_bytes(img: np.ndarray, fmt: str = "PNG") -> bytes:
    """Encode un tableau numpy en bytes."""
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    elif img.ndim == 2 and img.max() <= 1:
        img = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt)
    return buffer.getvalue()
