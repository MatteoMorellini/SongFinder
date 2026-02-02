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
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
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

# Add CORS middleware to allow requests from the web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Using device: {device}")
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class RecognizerResult:
    title: Optional[str]
    confidence: float
    cumulative_votes: Optional[Dict[int, int]] = None  # Only for Shazam

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

from approaches.grafp.inference import load_model, load_fingerprints, build_index, recognize as grafp_recognize
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
    faiss_index = build_index(db_fp, use_gpu=False)
    log_success("GraFP recognizer loaded successfully")
except Exception as e:
    log.error(f"Error loading database from {db_path}: {e}")
    sys.exit(1)

log_section("🚀 Server Ready")
log.info("All recognizers loaded. API is ready to accept requests.")

# -----------------------------
# Aggregator function
# -----------------------------

def run_recognizer(method: str, mp3_path: str, cumulative_votes: Optional[Dict[int, int]] = None) -> RecognizerResult:
    method = method.lower().strip()
    if method not in {"shazam", "grafp"}:
        raise HTTPException(status_code=400, detail="method must be 'shazam' or 'grafp'")

    log.info(f"Processing audio with '{method.upper()}' recognizer")
    log_detail("Audio file", mp3_path)
    
    if method == "shazam":
        # Helper function to get song name safely
        def get_song_name(song_id):
            try:
                if hasattr(db_meta, '__getitem__') and 0 <= song_id < len(db_meta):
                    return str(db_meta[song_id])
                else:
                    return f"Song {song_id}"
            except (IndexError, TypeError):
                return f"Song {song_id}"
        
        # The recognizer handles vote accumulation internally, we just pass cumulative_votes
        title, confidence, meta = shazam_adapter.recognize(
            mp3_path, 
            top_songs_entropy=TOP_SONGS_ENTROPY,
            cumulative_votes=cumulative_votes,
            debug=True  # Enable debug mode to see vote details
        )
        
        # Get the updated cumulative votes from the recognizer
        # The recognizer already adds current votes to cumulative votes
        updated_votes = meta.get("cumulative_votes", {})
        
        # Print updated cumulative votes from app side
        if updated_votes:
            log.info("📈 Cumulative votes summary (top 10):")
            sorted_updated = sorted(updated_votes.items(), key=lambda x: x[1], reverse=True)[:10]
            for song_id, votes in sorted_updated:
                song_name = get_song_name(song_id)
                log_detail(f"  Song {song_id}", f"{song_name} - {votes} votes")
        
        # Log result
        if title:
            log_success(f"Match found: '{title}' (confidence: {confidence:.2%})")
        
        return RecognizerResult(
            title=title,
            confidence=float(confidence),
            cumulative_votes=updated_votes
        )
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
        
        title, confidence = grafp_recognize(query_fp.cpu().numpy(), db_fp, db_meta, faiss_index, top_songs_entropy = TOP_SONGS_ENTROPY)
        
        # Log result
        if title:
            log_success(f"Match found: '{title}' (confidence: {confidence:.2%})")
        
        return RecognizerResult(
            title=title,
            confidence=float(confidence),
            cumulative_votes=None  # GraFP doesn't support cumulative votes
        )


# -----------------------------
# API endpoints
# -----------------------------

# Serve the HTML interface at the root
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the SongFinder web interface."""
    html_file = Path(__file__).parent / "songfinder.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head><title>SongFinder</title></head>
        <body>
            <h1>SongFinder API</h1>
            <p>Place songfinder.html in the same directory as this script to use the web interface.</p>
            <p>API Endpoint: POST /recognize</p>
        </body>
        </html>
        """)

@app.get("/health")
def health() -> Dict[str, str]:
    log.debug("Health check requested")
    return {"status": "ok"}


@app.get("/stats")
def get_stats() -> JSONResponse:
    """Get database statistics."""
    try:
        # Count total songs
        total_songs = len(db_meta) if db_meta is not None and len(db_meta) > 0 else 0
        
        # Count Shazam fingerprints
        shazam_fps = 0
        if hasattr(shazam_adapter, 'hash_table'):
            shazam_fps = len(shazam_adapter.hash_table)
        elif hasattr(shazam_adapter, 'database'):
            shazam_fps = len(shazam_adapter.database)
        
        # Count GraFP fingerprints (handle numpy array properly)
        grafp_fps = 0
        try:
            if db_fp is not None and hasattr(db_fp, '__len__'):
                grafp_fps = len(db_fp)
        except (TypeError, ValueError):
            grafp_fps = 0
        
        stats = {
            "total_songs": total_songs,
            "shazam_fingerprints": shazam_fps,
            "grafp_fingerprints": grafp_fps,
            "status": "ready"
        }
        print(stats)
        return JSONResponse(stats)
    except Exception as e:
        log.error(f"Error getting stats: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/recognize")
async def recognize(
    method: str = Form(...),
    file: UploadFile = File(...),
    cumulative_votes: Optional[str] = Form(None),
) -> JSONResponse:
    log.info(f"🎧 New recognition request received")
    log_detail("Method", method)
    log_detail("Filename", file.filename or "unknown")
    
    # Basic validation - accept both webm and mp3
    filename = (file.filename or "").lower()
    content_type = file.content_type or ""
    
    # Accept webm (from browser) and mp3
    valid_extensions = (".mp3", ".webm", ".wav", ".m4a", ".ogg")
    valid_types = ("audio/mpeg", "audio/mp3", "audio/webm", "audio/wav", "audio/x-m4a", "audio/ogg")
    
    if not (filename.endswith(valid_extensions) or content_type in valid_types):
        log.warning(f"Invalid file format - got {filename} with type {content_type}")
        raise HTTPException(
            status_code=400, 
            detail="Please upload an audio file (MP3, WebM, WAV, or M4A)."
        )

    # Determine file extension from filename or content type
    if filename.endswith(".webm") or "webm" in content_type:
        suffix = ".webm"
    elif filename.endswith(".mp4") or filename.endswith(".m4a") or "mp4" in content_type:
        suffix = ".mp4"
    elif filename.endswith(".ogg") or "ogg" in content_type:
        suffix = ".ogg"
    elif filename.endswith(".wav") or "wav" in content_type:
        suffix = ".wav"
    else:
        suffix = ".mp3"
    
    tmp_path = None
    converted_path = None
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            if not content:
                log.warning("Empty file upload rejected")
                raise HTTPException(status_code=400, detail="Empty upload.")
            tmp.write(content)
            log_detail("File size", f"{len(content) / 1024:.1f} KB")

        # Convert to WAV if needed (WebM/MP4/OGG need conversion)
        if suffix in [".webm", ".mp4", ".ogg"]:
            log.info(f"Converting {suffix} to WAV for processing...")
            converted_path = tempfile.mktemp(suffix=".wav")
            
            # Use ffmpeg to convert to WAV
            import subprocess
            result = subprocess.run(
                ["ffmpeg", "-i", tmp_path, "-ar", "44100", "-ac", "1", "-y", converted_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                log.error(f"FFmpeg conversion failed: {result.stderr}")
                raise HTTPException(
                    status_code=500, 
                    detail="Audio conversion failed. Make sure ffmpeg is installed."
                )
            
            log_detail("Converted to", converted_path)
            processing_path = converted_path
        else:
            processing_path = tmp_path

        # Parse cumulative_votes from JSON string if provided
        parsed_cumulative_votes = None
        if cumulative_votes and method.lower().strip() == "shazam":
            try:
                import json
                raw_votes = json.loads(cumulative_votes)
                # Convert string keys back to int (JSON doesn't support int keys)
                parsed_cumulative_votes = {int(k): v for k, v in raw_votes.items()}
                log_detail("Cumulative votes received", f"{len(parsed_cumulative_votes)} songs")
            except (json.JSONDecodeError, ValueError) as e:
                log.warning(f"Failed to parse cumulative_votes: {e}")
        
        result = run_recognizer(method, processing_path, parsed_cumulative_votes)

        log.info("✨ Request completed successfully")
        
        response_data = {
            "method": method.lower().strip(),
            "title": result.title,
            "confidence": result.confidence,   # normalized [0,1]
        }
        
        # Include cumulative_votes in response for Shazam
        if result.cumulative_votes is not None:
            # Convert int keys to strings for JSON serialization
            response_data["cumulative_votes"] = {str(k): v for k, v in result.cumulative_votes.items()}
        
        return JSONResponse(response_data)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Recognition failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                log.debug("Temporary file cleaned up")
            if converted_path and os.path.exists(converted_path):
                os.remove(converted_path)
                log.debug("Converted file cleaned up")
        except Exception:
            pass