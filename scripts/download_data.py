import shutil
import time
import urllib.request
from pathlib import Path
from urllib.error import ContentTooShortError, URLError

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://zenodo.org/records/10519652/files/pathmnist_224.npz?download=1"
DEST = DATA_DIR / "pathmnist_224.npz"


def download_file(url: str, dest: Path, max_retries: int = 5, timeout: int = 60) -> None:
    """Download a file with retries and partial-file cleanup."""
    for attempt in range(1, max_retries + 1):
        try:
            if dest.exists():
                print(f"{dest} already exists.")
                return

            temp_dest = dest.with_suffix(dest.suffix + ".part")
            if temp_dest.exists():
                temp_dest.unlink()

            print(f"Downloading PathMNIST 224x224... (attempt {attempt}/{max_retries})")

            with urllib.request.urlopen(url, timeout=timeout) as response, open(temp_dest, "wb") as out_file:
                shutil.copyfileobj(response, out_file)

            temp_dest.rename(dest)
            print(f"Saved to {dest}")
            return

        except (ContentTooShortError, URLError, TimeoutError, OSError) as e:
            print(f"Download failed on attempt {attempt}: {e}")

            temp_dest = dest.with_suffix(dest.suffix + ".part")
            if temp_dest.exists():
                temp_dest.unlink()

            if attempt == max_retries:
                raise RuntimeError(f"Failed to download after {max_retries} attempts") from e

            time.sleep(3)


if __name__ == "__main__":
    download_file(URL, DEST)

    if not DEST.exists():
        raise FileNotFoundError(f"Failed to download {DEST}")