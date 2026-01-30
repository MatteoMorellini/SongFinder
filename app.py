from __future__ import annotations

import sys
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import soundfile as sf
import torchaudio
import torch

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path

app = FastAPI(title="SongFinder API", version="1.0")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, str(Path(__file__).parent.parent))


# -----------------------------
# Plug your project here
# -----------------------------
@dataclass
class RecognizerResult:
    title: Optional[str]
    confidence: float


# -----------------------------
# Recognizer adapters
# -----------------------------
from approaches.shazam import ShazamRecognizer
from config_app import DB_PATH_GRAFP, DB_PATH_SHAZAM, CHECKPOINT
        
shazam_adapter = ShazamRecognizer()
db_path = Path(DB_PATH_SHAZAM) if DB_PATH_SHAZAM else Path("fingerprints/shazam")
print('loading shazam...')
shazam_adapter.load(db_path)
print('shazam loaded')

from approaches.grafp.inference import load_model, load_fingerprints, recognize as grafp_recognize
from approaches.grafp.util import load_config
from approaches.grafp.modules.transformations import AudioTransform
import torch

cfg = load_config('approaches/grafp/config/grafp.yaml')    
model = load_model(cfg, CHECKPOINT)
transform = AudioTransform(cfg).to(device)

# Load database fingerprints
db_path = Path(DB_PATH_GRAFP) if DB_PATH_GRAFP else Path("fingerprints/grafp")
print('laoding grafp...')
try:
    db_fp, db_meta = load_fingerprints(db_path)
    print('grafp loaded')
except Exception as e:
    print(f"Error loading database from {db_path}: {e}")
    sys.exit(1)












def run_recognizer(method: str, mp3_path: str) -> RecognizerResult:
    method = method.lower().strip()
    if method not in {"shazam", "grafp"}:
        raise HTTPException(status_code=400, detail="method must be 'shazam' or 'grafp'")

    print(f"method is {method}")
    if method == "shazam":
        title, confidence, meta = shazam_adapter.recognize(mp3_path)
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
        
        title, confidence = grafp_recognize(query_fp.cpu().numpy(), db_fp, db_meta)


    print(title,confidence)
    return RecognizerResult(
        title=title,
        confidence=float(confidence),
    )


# -----------------------------
# API endpoints
# -----------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/recognize")
async def recognize(
    method: str = Form(...),
    file: UploadFile = File(...),
) -> JSONResponse:
    # Basic validation
    filename = (file.filename or "").lower()
    if not (filename.endswith(".mp3") or file.content_type in {"audio/mpeg", "audio/mp3"}):
        raise HTTPException(status_code=400, detail="Please upload an MP3 file.")

    # Save to a temp file (many audio libs want a path)
    suffix = ".mp3"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty upload.")
            tmp.write(content)

        result = run_recognizer(method, tmp_path)

        return JSONResponse(
            {
                "method": method.lower().strip(),
                "title": result.title,
                "confidence": result.confidence,   # normalized [0,1]
            }
        )
    finally:
        # Cleanup
        try:
            if "tmp_path" in locals() and tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
