# 🧠 MeetingMind

AI-powered meeting summarizer and action-item tracker. Upload notes, documents,
scanned PDFs, or audio recordings and get an instant executive summary, key
points, and a trackable action-item list — backed by the Cerebras LLM API,
with SQLite persistence and a graceful **mock mode** that works even without
an API key.

## Features

- 📄 Document ingestion: `.txt`, `.pdf` (native text + OCR fallback for scans), `.docx`
- 🖼️ Image OCR (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`) via Tesseract
- 🎙️ Audio transcription (`.mp3`, `.wav`, `.m4a`, `.mp4`, `.ogg`, `.flac`, `.webm`) via `faster-whisper`
- 🤖 Cerebras LLM summarization + structured action-item extraction (JSON mode)
- 🟡 Automatic **mock mode** fallback — the app is fully demoable with zero API key
- 🗄️ SQLite persistence (SQLAlchemy ORM) with a searchable meeting history
- ✅ Editable action items with status tracking (Open / In Progress / Done)
- ⬇️ Export to CSV or Markdown

## Project Structure

```
MeetingMind/
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
├── README.md
└── app/
    ├── __init__.py
    └── main.py
```

## 1. Local Setup (no Docker)

**Requirements:** Python 3.11+, and the system binaries `tesseract-ocr`,
`poppler-utils` (for `pdftoppm`), and `ffmpeg` for full functionality. If
these aren't installed, the app still runs — OCR/PDF-scan/audio features
degrade gracefully with a clear in-app message instead of crashing.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# then edit .env and add your CEREBRAS_API_KEY (optional — mock mode works without it)

# 4. Run
streamlit run app/main.py
```

Open the app at **http://localhost:8501**.

### Installing system dependencies (optional, for OCR/audio)

- **macOS:** `brew install tesseract poppler ffmpeg`
- **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr poppler-utils ffmpeg`
- **Windows:** install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki),
  [poppler for Windows](https://github.com/oschwartz10612/poppler-windows), and
  [ffmpeg](https://ffmpeg.org/download.html), then add each `bin/` folder to your PATH.

If you'd rather not manage these on Windows at all, **skip straight to Docker**
below — the container already includes all three, precompiled for Linux, so
you avoid Windows compiler/PATH issues entirely.

## 2. Run with Docker

```bash
# Build (force linux/amd64 so the image matches Cloud Run's architecture)
docker build --platform linux/amd64 -t meetingmind:local .

# Run, mapping port 8080 and loading your .env
docker run --platform linux/amd64 -p 8080:8080 --env-file .env meetingmind:local
```

Open **http://localhost:8080**.

## 3. Deploy to Google Cloud Run

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

gcloud artifacts repositories create meetingmind-repo \
  --repository-format=docker --location=asia-south1

gcloud builds submit --tag asia-south1-docker.pkg.dev/YOUR_PROJECT_ID/meetingmind-repo/meetingmind:v1

echo -n "your-cerebras-key" | gcloud secrets create cerebras-api-key --data-file=-

gcloud run deploy meetingmind \
  --image=asia-south1-docker.pkg.dev/YOUR_PROJECT_ID/meetingmind-repo/meetingmind:v1 \
  --region=asia-south1 --allow-unauthenticated \
  --memory=2Gi --cpu=2 --timeout=300 \
  --set-env-vars="LLM_PROVIDER=cerebras,CEREBRAS_API_BASE=https://api.cerebras.ai/v1,DATABASE_PATH=/tmp/meetingmind.db" \
  --set-secrets="CEREBRAS_API_KEY=cerebras-api-key:latest"
```

> **Note on persistence:** Cloud Run's filesystem is ephemeral — the SQLite
> file resets on every new container instance/revision. For durable history
> across restarts, mount a [Cloud Run volume backed by Cloud Storage](https://cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts)
> or migrate to Cloud SQL. Fine for demos/single-session use as-is.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | LLM backend identifier | `cerebras` |
| `CEREBRAS_API_BASE` | Cerebras API base URL | `https://api.cerebras.ai/v1` |
| `CEREBRAS_API_KEY` | Your Cerebras API key (blank = mock mode) | *(empty)* |
| `CEREBRAS_MODEL` | Model name | `llama3.1-8b` |
| `WHISPER_MODEL_SIZE` | `tiny`/`base`/`small`/`medium`/`large-v3` | `base` |
| `WHISPER_DEVICE` | `cpu` or `cuda` | `cpu` |
| `WHISPER_COMPUTE_TYPE` | `int8`/`float16`/`float32` | `int8` |
| `DATABASE_PATH` | SQLite file path | `meetingmind.db` |

## Why no Windows compiler errors?

Every dependency in `requirements.txt` ships prebuilt wheels (including
`faster-whisper`'s CTranslate2 backend) for Windows, macOS, and Linux —
nothing requires `gcc`/MSVC to build from source. The Dockerfile additionally
pins everything to `python:3.11-slim` (Debian/glibc), so the exact same
wheels resolve identically on your Windows dev machine, in Docker Desktop,
and in Cloud Run's build environment.
