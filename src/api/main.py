"""
FastAPI application entrypoint.

This module creates the FastAPI app instance and registers the API
routers for health checks, model metadata, and prediction.
"""

from fastapi import FastAPI

from src.api.routes.health import router as health_router
from src.api.routes.model_info import router as model_info_router
from src.api.routes.predict import router as predict_router

from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

app = FastAPI(
    title="Colon Pathology Classifier API",
    version="0.1.0",
    description="Inference API for PathMNIST colorectal histology classification.",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


app.include_router(health_router)
app.include_router(model_info_router)
app.include_router(predict_router)
