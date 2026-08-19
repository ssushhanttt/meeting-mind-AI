"""
MeetingMind — AI Meeting Summarizer & Action Tracker
=====================================================
Single-file Streamlit application:
  - Ingests meeting notes from .txt, .pdf (native + OCR fallback), .docx,
    images (OCR), or audio files (faster-whisper transcription).
  - Summarizes + extracts action items via the Cerebras LLM API, with a
    graceful local MOCK fallback if no API key is configured.
  - Persists meetings + action items to SQLite via SQLAlchemy.
  - Provides a History view with search, export (CSV / Markdown), and
    inline action-item status editing.

Run locally:    streamlit run app/main.py
Run in Docker:  see Dockerfile / README.md
"""

from __future__ import annotations

import io
import os
import re
import json
import tempfile
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey, select
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session, selectinload

# Optional / heavy imports are wrapped defensively so the app still boots
# (in a degraded mode with a clear warning) if a system dependency like
# tesseract or ffmpeg happens to be missing in a given environment.
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx as python_docx
except ImportError:
    python_docx = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


# =============================================================================
# Configuration
# =============================================================================

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "cerebras")
CEREBRAS_API_BASE = os.getenv("CEREBRAS_API_BASE", "https://api.cerebras.ai/v1")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama3.1-8b")

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

DATABASE_PATH = os.getenv("DATABASE_PATH", "meetingmind.db")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))

MOCK_MODE = not bool(CEREBRAS_API_KEY)

APP_TITLE = "MeetingMind"
APP_ICON = "🧠"


# =============================================================================
# Database layer (SQLAlchemy)
# =============================================================================

Base = declarative_base()


class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    source_type = Column(String(50), nullable=False, default="text")
    raw_text = Column(Text, nullable=False, default="")
    summary = Column(Text, nullable=True)
    key_points = Column(Text, nullable=True)     # JSON-encoded list[str]
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    llm_mode = Column(String(20), default="mock")

    action_items = relationship(
        "ActionItem", back_populates="meeting",
        cascade="all, delete-orphan", order_by="ActionItem.id"
    )


class ActionItem(Base):
    __tablename__ = "action_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), nullable=False)
    description = Column(Text, nullable=False)
    owner = Column(String(255), nullable=True)
    due_date = Column(String(50), nullable=True)
    priority = Column(String(20), default="Medium")
    status = Column(String(20), default="Open")

    meeting = relationship("Meeting", back_populates="action_items")


@st.cache_resource(show_spinner=False)
def get_engine():
    engine = create_engine(
        f"sqlite:///{DATABASE_PATH}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return engine


def get_session() -> Session:
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    return SessionLocal()


def save_meeting(title: str, source_type: str, raw_text: str,
                  summary: str, key_points: list[str],
                  action_items: list[dict], llm_mode: str) -> int:
    session = get_session()
    try:
        meeting = Meeting(
            title=title or "Untitled Meeting",
            source_type=source_type,
            raw_text=raw_text,
            summary=summary,
            key_points=json.dumps(key_points, ensure_ascii=False),
            llm_mode=llm_mode,
        )
        for item in action_items:
            meeting.action_items.append(ActionItem(
                description=item.get("description", "").strip(),
                owner=item.get("owner") or "Unassigned",
                due_date=item.get("due_date") or "",
                priority=item.get("priority") or "Medium",
                status="Open",
            ))
        session.add(meeting)
        session.commit()
        return meeting.id
    finally:
        session.close()


def list_meetings() -> list[Meeting]:
    session = get_session()
    try:
        stmt = (
            select(Meeting)
            .options(selectinload(Meeting.action_items))
            .order_by(Meeting.created_at.desc())
        )
        return list(session.scalars(stmt))
    finally:
        session.close()


def get_meeting(meeting_id: int) -> Optional[Meeting]:
    session = get_session()
    try:
        stmt = (
            select(Meeting)
            .where(Meeting.id == meeting_id)
            .options(selectinload(Meeting.action_items))
        )
        return session.scalars(stmt).first()
    finally:
        session.close()


def update_action_item_status(item_id: int, new_status: str) -> None:
    session = get_session()
    try:
        item = session.get(ActionItem, item_id)
        if item:
            item.status = new_status
            session.commit()
    finally:
        session.close()


def delete_meeting(meeting_id: int) -> None:
    session = get_session()
    try:
        meeting = session.get(Meeting, meeting_id)
        if meeting:
            session.delete(meeting)
            session.commit()
    finally:
        session.close()


# =============================================================================
# File parsing / extraction utilities
# =============================================================================

def extract_text_from_txt(uploaded_file) -> str:
    raw = uploaded_file.read()
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _ocr_image(image: Image.Image) -> str:
    if pytesseract is None:
        return "[OCR unavailable: pytesseract not installed]"
    try:
        return pytesseract.image_to_string(image)
    except Exception as exc:  # tesseract binary missing, etc.
        return f"[OCR error: {exc}]"


def extract_text_from_pdf(uploaded_file) -> str:
    raw_bytes = uploaded_file.read()
    text_chunks: list[str] = []

    # 1) Try native text extraction first (fast, works for digital PDFs)
    if pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text_chunks.append(page_text)
        except Exception:
            text_chunks = []

    combined = "\n".join(t for t in text_chunks if t.strip())

    # 2) Fallback to OCR if little/no text was found (scanned document)
    if len(combined.strip()) < 20 and convert_from_bytes is not None:
        try:
            images = convert_from_bytes(raw_bytes, dpi=200)
            ocr_chunks = [_ocr_image(img) for img in images]
            combined = "\n".join(ocr_chunks)
        except Exception as exc:
            combined += f"\n[PDF OCR fallback failed: {exc}]"

    if not combined.strip() and PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            combined = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            pass

    return combined.strip() or "[No extractable text found in PDF]"


def extract_text_from_docx(uploaded_file) -> str:
    if python_docx is None:
        return "[.docx parsing unavailable: python-docx not installed]"
    document = python_docx.Document(uploaded_file)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            paragraphs.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(paragraphs)


def extract_text_from_image(uploaded_file) -> str:
    image = Image.open(uploaded_file)
    return _ocr_image(image)


@st.cache_resource(show_spinner=False)
def get_whisper_model():
    if WhisperModel is None:
        return None
    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )


def transcribe_audio(uploaded_file) -> str:
    model = get_whisper_model()
    if model is None:
        return "[Audio transcription unavailable: faster-whisper not installed]"

    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path, beam_size=5, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments)
        return text.strip() or "[No speech detected]"
    except Exception as exc:
        return f"[Transcription error: {exc}]"
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def extract_text(uploaded_file) -> tuple[str, str]:
    """Returns (extracted_text, source_type)."""
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file), "document"
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file), "document"
    if name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file), "document"
    if name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        return extract_text_from_image(uploaded_file), "image_ocr"
    if name.endswith((".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac", ".webm")):
        return transcribe_audio(uploaded_file), "audio"
    return "[Unsupported file type]", "unknown"


# =============================================================================
# LLM integration (Cerebras) with graceful mock fallback
# =============================================================================

SYSTEM_PROMPT = """You are MeetingMind, an expert meeting-notes analyst.
Given raw meeting transcript/notes, respond with ONLY valid JSON (no markdown
fences, no commentary) matching exactly this schema:

{
  "summary": "2-4 sentence executive summary",
  "key_points": ["point 1", "point 2", "..."],
  "action_items": [
    {"description": "...", "owner": "name or Unassigned", "due_date": "YYYY-MM-DD or empty string", "priority": "High|Medium|Low"}
  ]
}

If no action items exist, return an empty list for "action_items". Never
include any text outside the JSON object."""


@dataclass
class SummaryResult:
    summary: str
    key_points: list[str] = field(default_factory=list)
    action_items: list[dict] = field(default_factory=list)
    mode: str = "mock"


def _call_cerebras(text: str) -> dict:
    url = f"{CEREBRAS_API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CEREBRAS_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text[:20000]},
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    # Strip accidental markdown fences if the model adds them anyway
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
    return json.loads(content)


def _mock_summarize(text: str) -> dict:
    """Deterministic, dependency-free heuristic summarizer used when no
    CEREBRAS_API_KEY is configured, so the app remains fully demoable."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    summary = " ".join(sentences[:3]) or "No content available to summarize."
    key_points = sentences[3:8] if len(sentences) > 3 else sentences[:5]

    action_items = []
    action_patterns = re.compile(
        r"\b(will|needs? to|action item|todo|to-do|must|should|follow up|by\s+\w+day)\b",
        re.IGNORECASE,
    )
    owner_pattern = re.compile(r"^\s*([A-Z][a-zA-Z]+)\s+(?:will|needs to|to)\b")

    for sentence in sentences:
        if action_patterns.search(sentence):
            owner_match = owner_pattern.match(sentence)
            action_items.append({
                "description": sentence,
                "owner": owner_match.group(1) if owner_match else "Unassigned",
                "due_date": "",
                "priority": "Medium",
            })

    return {
        "summary": f"[MOCK MODE] {summary}",
        "key_points": key_points,
        "action_items": action_items[:10],
    }


def summarize_meeting(text: str) -> SummaryResult:
    if not text or not text.strip() or text.startswith("["):
        return SummaryResult(
            summary="No usable text was extracted from the input.",
            mode="mock",
        )

    if MOCK_MODE:
        result = _mock_summarize(text)
        return SummaryResult(
            summary=result["summary"],
            key_points=result["key_points"],
            action_items=result["action_items"],
            mode="mock",
        )

    try:
        result = _call_cerebras(text)
        return SummaryResult(
            summary=result.get("summary", ""),
            key_points=result.get("key_points", []),
            action_items=result.get("action_items", []),
            mode="cerebras",
        )
    except Exception as exc:
        st.warning(f"Cerebras API call failed, falling back to mock mode: {exc}")
        result = _mock_summarize(text)
        return SummaryResult(
            summary=result["summary"],
            key_points=result["key_points"],
            action_items=result["action_items"],
            mode="mock-fallback",
        )


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


def render_sidebar() -> str:
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.caption("AI Meeting Summarizer & Action Tracker")
        st.divider()

        if MOCK_MODE:
            st.warning("⚠️ Running in **MOCK MODE**\nSet `CEREBRAS_API_KEY` for real LLM summaries.")
        else:
            st.success(f"✅ Connected to Cerebras (`{CEREBRAS_MODEL}`)")

        if WhisperModel is None:
            st.warning("⚠️ Audio transcription disabled (faster-whisper not installed)")
        else:
            st.caption(f"🎙️ Whisper model: `{WHISPER_MODEL_SIZE}` ({WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE})")

        st.divider()
        page = st.radio("Navigate", ["📝 New Meeting", "📚 History"], label_visibility="collapsed")
        st.divider()
        st.caption(f"DB: `{DATABASE_PATH}`")
        return page


def render_new_meeting_page():
    st.header("📝 New Meeting")
    title = st.text_input("Meeting title", placeholder="e.g. Weekly Product Sync — Aug 19")

    tab_doc, tab_audio, tab_text = st.tabs(["📄 Upload Document / Image", "🎙️ Upload Audio", "✍️ Paste Text"])

    extracted_text = ""
    source_type = "text"

    with tab_doc:
        uploaded = st.file_uploader(
            "Upload .txt, .pdf, .docx, or an image (OCR)",
            type=["txt", "pdf", "docx", "png", "jpg", "jpeg", "bmp", "tiff"],
        )
        if uploaded is not None:
            with st.spinner(f"Extracting text from {uploaded.name}..."):
                extracted_text, source_type = extract_text(uploaded)
            st.text_area("Extracted text", extracted_text, height=250, key="doc_preview")

    with tab_audio:
        uploaded_audio = st.file_uploader(
            "Upload audio (.mp3, .wav, .m4a, .mp4, .ogg, .flac, .webm)",
            type=["mp3", "wav", "m4a", "mp4", "ogg", "flac", "webm"],
        )
        if uploaded_audio is not None:
            with st.spinner("Transcribing audio with faster-whisper... this can take a minute"):
                extracted_text = transcribe_audio(uploaded_audio)
                source_type = "audio"
            st.text_area("Transcript", extracted_text, height=250, key="audio_preview")

    with tab_text:
        pasted = st.text_area("Paste meeting notes / transcript directly", height=250)
        if pasted.strip():
            extracted_text = pasted
            source_type = "text"

    st.divider()

    if st.button("🚀 Generate Summary & Action Items", type="primary", disabled=not extracted_text.strip()):
        with st.spinner("Analyzing meeting content..."):
            result = summarize_meeting(extracted_text)
        st.session_state["last_result"] = result
        st.session_state["last_text"] = extracted_text
        st.session_state["last_source_type"] = source_type
        st.session_state["last_title"] = title

    result: Optional[SummaryResult] = st.session_state.get("last_result")
    if result:
        st.subheader("Summary")
        mode_badge = {"cerebras": "🟢 Cerebras", "mock": "🟡 Mock", "mock-fallback": "🟠 Mock (fallback)"}
        st.caption(mode_badge.get(result.mode, result.mode))
        st.write(result.summary)

        if result.key_points:
            st.subheader("Key Points")
            for point in result.key_points:
                st.markdown(f"- {point}")

        st.subheader("Action Items")
        if result.action_items:
            df = pd.DataFrame(result.action_items)
            edited = st.data_editor(
                df, num_rows="dynamic", use_container_width=True,
                column_config={
                    "priority": st.column_config.SelectboxColumn(options=["High", "Medium", "Low"]),
                },
                key="action_items_editor",
            )
        else:
            st.info("No action items detected.")
            edited = pd.DataFrame(columns=["description", "owner", "due_date", "priority"])

        if st.button("💾 Save Meeting to History"):
            meeting_id = save_meeting(
                title=st.session_state.get("last_title") or "Untitled Meeting",
                source_type=st.session_state.get("last_source_type", "text"),
                raw_text=st.session_state.get("last_text", ""),
                summary=result.summary,
                key_points=result.key_points,
                action_items=edited.to_dict("records"),
                llm_mode=result.mode,
            )
            st.success(f"Saved meeting #{meeting_id} to history ✅")
            for key in ("last_result", "last_text", "last_source_type", "last_title"):
                st.session_state.pop(key, None)
            st.rerun()


def render_history_page():
    st.header("📚 Meeting History")
    meetings = list_meetings()

    if not meetings:
        st.info("No meetings saved yet. Create one from the **New Meeting** tab.")
        return

    options = {f"#{m.id} — {m.title} ({m.created_at:%Y-%m-%d %H:%M})": m.id for m in meetings}
    selected_label = st.selectbox("Select a meeting", list(options.keys()))
    meeting = get_meeting(options[selected_label])

    if meeting is None:
        st.warning("Meeting not found (it may have been deleted).")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(meeting.title)
        st.caption(f"Source: {meeting.source_type} · Mode: {meeting.llm_mode} · Created: {meeting.created_at:%Y-%m-%d %H:%M}")
    with col2:
        if st.button("🗑️ Delete meeting", type="secondary"):
            delete_meeting(meeting.id)
            st.success("Deleted.")
            st.rerun()

    st.markdown("**Summary**")
    st.write(meeting.summary or "—")

    try:
        key_points = json.loads(meeting.key_points or "[]")
    except json.JSONDecodeError:
        key_points = []
    if key_points:
        st.markdown("**Key Points**")
        for point in key_points:
            st.markdown(f"- {point}")

    st.markdown("**Action Items**")
    if meeting.action_items:
        for item in meeting.action_items:
            cols = st.columns([5, 2, 2, 1, 2])
            cols[0].write(item.description)
            cols[1].write(item.owner or "—")
            cols[2].write(item.due_date or "—")
            cols[3].write(item.priority)
            new_status = cols[4].selectbox(
                "Status", ["Open", "In Progress", "Done"],
                index=["Open", "In Progress", "Done"].index(item.status)
                if item.status in ("Open", "In Progress", "Done") else 0,
                key=f"status_{item.id}", label_visibility="collapsed",
            )
            if new_status != item.status:
                update_action_item_status(item.id, new_status)
                st.rerun()
    else:
        st.caption("No action items for this meeting.")

    st.divider()
    with st.expander("Raw source text"):
        st.text(meeting.raw_text)

    export_col1, export_col2 = st.columns(2)
    with export_col1:
        items_df = pd.DataFrame([{
            "description": i.description, "owner": i.owner,
            "due_date": i.due_date, "priority": i.priority, "status": i.status,
        } for i in meeting.action_items])
        st.download_button(
            "⬇️ Export action items (CSV)",
            items_df.to_csv(index=False).encode("utf-8"),
            file_name=f"meetingmind_actions_{meeting.id}.csv",
            mime="text/csv",
        )
    with export_col2:
        md_lines = [f"# {meeting.title}", "", "## Summary", meeting.summary or "", "", "## Action Items"]
        for i in meeting.action_items:
            md_lines.append(f"- [{i.status}] {i.description} (Owner: {i.owner}, Due: {i.due_date or 'N/A'}, Priority: {i.priority})")
        st.download_button(
            "⬇️ Export full report (Markdown)",
            "\n".join(md_lines).encode("utf-8"),
            file_name=f"meetingmind_report_{meeting.id}.md",
            mime="text/markdown",
        )


def main():
    page = render_sidebar()
    if page == "📝 New Meeting":
        render_new_meeting_page()
    else:
        render_history_page()


if __name__ == "__main__":
    main()
