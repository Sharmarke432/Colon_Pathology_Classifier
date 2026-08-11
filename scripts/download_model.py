# scripts/download_model.py
from pathlib import Path
import os
from huggingface_hub import hf_hub_download

REPO_ID = os.getenv("HF_MODEL_REPO", "your-username/pathmnist-resnet18")
FILENAME = os.getenv("HF_MODEL_FILENAME", "pathmnist_resnet18.pt")
REVISION = os.getenv("HF_MODEL_REVISION")  # optional, recommended to pin commit
OUTPUT_PATH = Path("models") / FILENAME
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

downloaded_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    revision=REVISION,
    token=os.getenv("HF_TOKEN"),  # only needed if private
)

source = Path(downloaded_path)
if source.resolve() != OUTPUT_PATH.resolve():
    OUTPUT_PATH.write_bytes(source.read_bytes())

print(f"Model saved to {OUTPUT_PATH}")
