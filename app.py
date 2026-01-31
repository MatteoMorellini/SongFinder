# !!!
# TO RUN THE SERVER: uvicorn app:app --host 0.0.0.0 --port 8000
# !!!

from __future__ import annotations

import sys
import os
import tempfile
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import soundfile as sf
import torchaudio
import torch
from wcwidth import wcswidth

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path

# -----------------------------
# Logging Configuration
# -----------------------------

class PrettyFormatter(logging.Formatter):
    """Custom formatter with colors and clear step indicators."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    ICONS = {
        'DEBUG': '🔍',
        'INFO': '✅',
        'WARNING': '⚠️ ',
        'ERROR': '❌',
        'CRITICAL': '🔥',
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        icon = self.ICONS.get(record.levelname, '')
        
        # Format timestamp
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        
        # Create formatted message
        formatted = (
            f"{self.BOLD}[{timestamp}]{self.RESET} "
            f"{color}{icon} {record.levelname:<8}{self.RESET} │ "
            f"{record.getMessage()}"
        )
        return formatted

def setup_logging():
    """Configure logging with pretty output."""
    logger = logging.getLogger("songfinder")
    logger.setLevel(logging.DEBUG)
    
    # Console handler with pretty formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(PrettyFormatter())
    
    logger.addHandler(console_handler)
    return logger

log = setup_logging()

def _center_display(s: str, target_cols: int) -> str:
    """Center using terminal display width (handles emoji/double-width chars)."""
    w = wcswidth(s)
    if w < 0:
        w = len(s)  # fallback

    if w >= target_cols:
        return s  # too wide, don't pad

    pad = target_cols - w
    left = pad // 2
    right = pad - left
    return (" " * left) + s + (" " * right)

def log_section(title: str):
    """Print a visually distinct section header."""
    width = 50
    border = "═" * width
    print(f"\n\033[1;34m╔{border}╗\033[0m")
    centered = _center_display(title, width - 2)
    print(f"\033[1;34m║\033[0m {centered} \033[1;34m║\033[0m")
    print(f"\033[1;34m╚{border}╝\033[0m\n")


def log_step(step_num: int, description: str):
    """Print a numbered step indicator."""
    print(f"  \033[1;36m[Step {step_num}]\033[0m ➜  {description}")

def log_success(message: str):
    """Print a success message."""
    print(f"  \033[1;32m✓\033[0m {message}")

def log_detail(key: str, value: str):
    """Print a key-value detail."""
    print(f"      \033[90m•\033[0m {key}: \033[1m{value}\033[0m")

# -----------------------------
# App Initialization
# -----------------------------

log_section("🎵 SongFinder API Server")

app = FastAPI(title="SongFinder API", version="1.0")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Using device: {device}")
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class RecognizerResult:
    title: Optional[str]
    confidence: float

# -----------------------------
# Shazam recognizer
# -----------------------------

from approaches.shazam import ShazamRecognizer
from config_app import DB_PATH_GRAFP, DB_PATH_SHAZAM, CHECKPOINT, TOP_SONGS_ENTROPY

log_section("📦 Loading Recognizers")

log_step(1, "Initializing Shazam recognizer...")
shazam_adapter = ShazamRecognizer()
db_path = Path(DB_PATH_SHAZAM) if DB_PATH_SHAZAM else Path("fingerprints/shazam")
log_detail("Database path", str(db_path))
shazam_adapter.load(db_path)
log_success("Shazam recognizer loaded successfully")

# -----------------------------
# GraFP recognizer
# -----------------------------

from approaches.grafp.inference import load_model, load_fingerprints, recognize as grafp_recognize
from approaches.grafp.util import load_config
from approaches.grafp.modules.transformations import AudioTransform
import torch

log_step(2, "Initializing GraFP recognizer...")
cfg = load_config('approaches/grafp/config/grafp.yaml')    
model = load_model(cfg, CHECKPOINT)
transform = AudioTransform(cfg).to(device)

# Load database fingerprints
db_path = Path(DB_PATH_GRAFP) if DB_PATH_GRAFP else Path("fingerprints/grafp")
log_detail("Database path", str(db_path))
log_detail("Checkpoint", str(CHECKPOINT))
try:
    db_fp, db_meta = load_fingerprints(db_path)
    log_success("GraFP recognizer loaded successfully")
except Exception as e:
    log.error(f"Error loading database from {db_path}: {e}")
    sys.exit(1)

log_section("🚀 Server Ready")
log.info("All recognizers loaded. API is ready to accept requests.")

# -----------------------------
# Aggregator function
# -----------------------------

def run_recognizer(method: str, mp3_path: str) -> RecognizerResult:
    method = method.lower().strip()
    if method not in {"shazam", "grafp"}:
        raise HTTPException(status_code=400, detail="method must be 'shazam' or 'grafp'")

    log.info(f"Processing audio with '{method.upper()}' recognizer")
    log_detail("Audio file", mp3_path)
    
    if method == "shazam":
        title, confidence, meta = shazam_adapter.recognize(mp3_path, top_songs_entropy = TOP_SONGS_ENTROPY)
    else:
        signal, sr = sf.read(mp3_path)
        waveform = torch.from_numpy(signal).float()
        
        # Convert to mono if stereo
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=1)
            
        if sr != cfg['fs']:
            waveform = torchaudio.transforms.Resample(sr, cfg['fs'])(waveform)
            
        segments = transform(waveform.unsqueeze(0).to(device))
        
        with torch.no_grad():
            _, _, query_fp, _ = model(segments, segments)
        
        title, confidence = grafp_recognize(query_fp.cpu().numpy(), db_fp, db_meta, top_songs_entropy = TOP_SONGS_ENTROPY)

    # Log result
    if title:
        log_success(f"Match found: '{title}' (confidence: {confidence:.2%})")
    else:
        log.warning(f"No match found (confidence: {confidence:.2%})")

    return RecognizerResult(
        title=title,
        confidence=float(confidence),
    )


# -----------------------------
# API endpoints
# -----------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    log.debug("Health check requested")
    return {"status": "ok"}


@app.post("/recognize")
async def recognize(
    method: str = Form(...),
    file: UploadFile = File(...),
) -> JSONResponse:
    log.info(f"🎧 New recognition request received")
    log_detail("Method", method)
    log_detail("Filename", file.filename or "unknown")
    
    # Basic validation
    filename = (file.filename or "").lower()
    if not (filename.endswith(".mp3") or file.content_type in {"audio/mpeg", "audio/mp3"}):
        log.warning("Invalid file format - expected MP3")
        raise HTTPException(status_code=400, detail="Please upload an MP3 file.")

    # Save to a temp file (many audio libs want a path)
    suffix = ".mp3"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            if not content:
                log.warning("Empty file upload rejected")
                raise HTTPException(status_code=400, detail="Empty upload.")
            tmp.write(content)
            log_detail("File size", f"{len(content) / 1024:.1f} KB")

        result = run_recognizer(method, tmp_path)

        log.info("✨ Request completed successfully")
        return JSONResponse(
            {
                "method": method.lower().strip(),
                "title": result.title,
                "confidence": result.confidence,   # normalized [0,1]
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Recognition failed: {e}")
        raise
    finally:
        # Cleanup
        try:
            if "tmp_path" in locals() and tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                log.debug("Temporary file cleaned up")
        except Exception:
            pass
