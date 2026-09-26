# linux/arm64 target (Oracle Ampere A1). Build with:
#   docker buildx build --platform linux/arm64 -t yt-performance-predictor:latest .
FROM python:3.13-slim

WORKDIR /app

# System deps for pillow/torch on arm64 (libjpeg/zlib headers; torch's arm64
# wheel is manylinux and doesn't need a compiler, so no build-essential).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .

RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements-serve.txt

# Only what serving/ actually needs at import time -- no data/, notebooks,
# ingestion/, Airflow DAGs, or training-only requirements.
COPY serving/ serving/
COPY models/late_fusion_model.py models/precompute_embeddings.py models/
COPY features/ features/

RUN python -c "from transformers import CLIPModel, AutoTokenizer; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
AutoTokenizer.from_pretrained('openai/clip-vit-base-patch32')"

ENV HF_HUB_OFFLINE=1

# Bundle is NOT baked into the image (retrains shouldn't require a rebuild) --
# it's mounted as a volume at runtime, see docker-compose.yml.
ENV MODEL_BUNDLE_PATH=/app/models/bundles/latest_clip_b32_clip.pt

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]