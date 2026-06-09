# PathMNIST FastAPI Service

Inference-only FastAPI service for a PathMNIST tissue patch classifier trained in Google Colab.

## Workflow

1. Train and evaluate the model in Google Colab.
2. Export:
   - `model.pt`
   - `id2label.json`
   - `config.json`
3. Put those files into `artifacts/`.
4. Run the FastAPI service locally or in Docker.

## Structure

```text
pathmnist-fastapi/
├── app/
├── artifacts/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Docker

```bash
docker build -t pathmnist-fastapi .
docker run -p 8000:8000 pathmnist-fastapi
```

## Endpoints

- `GET /health`
- `GET /metadata`
- `POST /predict`

## Example curl

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_patch.png"
```

## Important

The current scaffold uses a fallback `resnet18` so the app can boot. Replace the model-building logic in `app/models/loader.py` with the exact architecture you trained in Colab, otherwise your checkpoint may not load correctly.