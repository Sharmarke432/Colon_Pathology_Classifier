"""
Inference service for loading the trained model and running predictions.
"""

import os
from pathlib import Path
from typing import cast

import torch

from src.models import create_model
from src.schemas.prediction import ClassProbability, PredictionResponse
from src.services.preprocessing import load_image_from_bytes, preprocess_image
from src.core.config import get_default_config, get_torch_device
from src.training.constants import NUM_CLASSES
from src.training.utils import load_checkpoint


CLASS_LABELS: dict[int, str] = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth_muscle",
    6: "normal_colon_mucosa",
    7: "cancer_associated_stroma",
    8: "colorectal_adenocarcinoma_epithelium",
}


class InferenceService:
    """
    Service object responsible for model loading and prediction.
    """

    def __init__(self) -> None:
        self.model_name = os.getenv("MODEL_NAME", "resnet18")
        self.data_dir = Path("data")
        self.output_dir = Path("artifacts")

        self.config = get_default_config(
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            model_name=cast(str, self.model_name),
        )
        self.device = get_torch_device(self.config)
        self.checkpoint_path = self.config.checkpoint_path()

        self.model = create_model(
            backbone_name=self.config.model_name,
            num_classes=NUM_CLASSES,
            pretrained=False,
            dropout_p=0.0,
        ).to(self.device)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at '{self.checkpoint_path}'. "
                "Train the model before starting the API."
            )

        load_checkpoint(
            model=self.model,
            path=self.checkpoint_path,
            map_location=self.device,
        )
        self.model.eval()

    def predict_from_bytes(
        self,
        image_bytes: bytes,
        filename: str,
    ) -> PredictionResponse:
        """
        Run inference on an uploaded image represented as bytes.
        """
        image = load_image_from_bytes(image_bytes)
        inputs = preprocess_image(image).to(self.device)

        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())

        probability_items = [
            ClassProbability(
                class_id=class_id,
                label=CLASS_LABELS.get(class_id, f"class_{class_id}"),
                probability=float(prob.item()),
            )
            for class_id, prob in enumerate(probabilities)
        ]

        return PredictionResponse(
            filename=filename,
            predicted_class=predicted_class,
            predicted_label=CLASS_LABELS.get(
                predicted_class, f"class_{predicted_class}"
            ),
            confidence=confidence,
            probabilities=probability_items,
        )


_inference_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    """
    Return a singleton inference service instance.
    """
    global _inference_service

    if _inference_service is None:
        _inference_service = InferenceService()

    return _inference_service
