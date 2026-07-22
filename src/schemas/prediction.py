"""
Pydantic schemas for inference responses.
"""

from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    """
    Per-class probability entry.
    """

    class_id: int = Field(..., ge=0)
    label: str
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """
    Response returned by the prediction endpoint.
    """

    filename: str
    predicted_class: int = Field(..., ge=0)
    predicted_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: list[ClassProbability]


class ModelInfoResponse(BaseModel):
    """
    Response returned by the model info endpoint.
    """

    model_name: str
    num_classes: int = Field(..., ge=1)
    image_height: int = Field(..., ge=1)
    image_width: int = Field(..., ge=1)
    device: str
    checkpoint_path: str
