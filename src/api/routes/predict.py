"""
Prediction routes.
"""

from certifi import contents
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from src.schemas.prediction import PredictionResponse
from src.services.inference import get_inference_service

from PIL import Image, UnidentifiedImageError
import io

router = APIRouter(tags=["predict"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Run single-image inference on an uploaded pathology patch.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an image.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    service = get_inference_service()

    try:
        result = service.predict_from_bytes(
            image_bytes=image_bytes,
            filename=file.filename or "uploaded_image",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image."
        )

    return result
