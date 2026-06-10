import urllib.request
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

url = "https://zenodo.org/records/10519652/files/pathmnist_224.npz?download=1"
dest = DATA_DIR / "pathmnist_224.npz"
if dest.exists():
    print("Already downloaded.")
else:
    print("Downloading PathMNIST 224x224...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")