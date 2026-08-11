# scripts/download_model.py
from pathlib import Path
import os
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

REPO_ID = os.getenv("HF_MODEL_REPO") or "SharmarkeO/Resnet18-Colon"
FILENAME = os.getenv("HF_MODEL_FILENAME") or "best_model.pt"
HF_TOKEN = os.getenv("HF_TOKEN")  # only needed if private

OUTPUT_PATH = Path("models") / FILENAME
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

downloaded_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    token=HF_TOKEN,
)

source = Path(downloaded_path)
if source.resolve() != OUTPUT_PATH.resolve():
    OUTPUT_PATH.write_bytes(source.read_bytes())

print(f"Model saved to {OUTPUT_PATH}")
