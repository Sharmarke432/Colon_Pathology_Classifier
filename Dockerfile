FROM python:3.11-slim


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


WORKDIR /app


# System deps for Pillow / OpenCV-style image ops
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*


# Install Python deps
COPY requirements.txt .


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision torchaudio && \
    pip install --no-cache-dir -r requirements.txt


# Copy code (but NOT the model checkpoint)
COPY src ./src
COPY scripts ./scripts

# Build args
ARG HF_TOKEN
ARG HF_MODEL_REPO
ARG HF_MODEL_FILENAME

# Set as env for the download step
ENV HF_TOKEN=${HF_TOKEN}
ENV HF_MODEL_REPO=${HF_MODEL_REPO}
ENV HF_MODEL_FILENAME=${HF_MODEL_FILENAME}

# Create models directory and download the checkpoint at build time
RUN mkdir -p /app/models
RUN python scripts/download_model.py


ENV MODEL_NAME=resnet18
ENV MODEL_PATH=/app/models/pathmnist_resnet18.pt


# Copy remaining project files (tests, etc.) if needed
COPY . .


EXPOSE 8000


# Start FastAPI with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
