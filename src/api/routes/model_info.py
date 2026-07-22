"""
Model metadata routes.
"""

from fastapi import APIRouter

from src.schemas.prediction import ModelInfoResponse
from src.services.inference import get_inference_service
from src.training.constants import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES

router = APIRouter(tags=["model-info"])


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """
    Return metadata about the currently configured inference model.
    """
    service = get_inference_service()

    return ModelInfoResponse(
        model_name=service.model_name,
        num_classes=NUM_CLASSES,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        device=service.device.type,
        checkpoint_path=str(service.checkpoint_path),
    )
