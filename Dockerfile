# =============================================================================
# MeetingMind — production Dockerfile
# Base: python:3.11-slim  (Debian-based, small footprint, glibc for prebuilt
# wheels like ctranslate2/faster-whisper so NOTHING compiles from source)
# =============================================================================
FROM python:3.11-slim

# --- Environment hygiene -----------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# --- System dependencies ------------------------------------------------------
# tesseract-ocr   -> OCR for scanned PDFs / images (pytesseract)
# poppler-utils   -> PDF -> image rendering (pdf2image)
# ffmpeg          -> audio/video decoding for faster-whisper
# libgl1          -> required by some Pillow/OCR image codecs on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        ffmpeg \
        libgl1 \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies -------------------------------------------------------
# Copied and installed BEFORE the rest of the source so Docker layer caching
# skips this (slowest) step on every subsequent rebuild unless requirements
# actually change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application source ---------------------------------------------------------
COPY . .

# Non-root user (Cloud Run / GKE best practice)
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Cloud Run injects $PORT at runtime; default to 8080 for local `docker run`
ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/_stcore/health || exit 1

# shell form so ${PORT} expands correctly at container start
CMD streamlit run app/main.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
