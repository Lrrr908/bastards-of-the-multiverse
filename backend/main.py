from pathlib import Path
import subprocess
import shutil

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="BotMCMS API")

# Allow your frontend and local dev to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# PATH SETUP
# backend/main.py -> backend (parent) -> repo root
# In the container this should be something like /app/backend/main.py
# --------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent        # /app/backend
ROOT_DIR = BACKEND_DIR.parent                        # /app
INDEX_PATH = ROOT_DIR / "index.html"                 # /app/index.html


def find_transicc() -> Path | None:
    """
    Prefer the system-installed transicc (from lcms2-utils).
    Fall back to any custom binary we find in the repo.
    """

    # 1) Look in PATH (e.g., /usr/bin/transicc)
    in_path = shutil.which("transicc")
    if in_path:
        return Path(in_path)

    # 2) Fallback: look in common repo locations
    candidates = [
        ROOT_DIR / "botmcms" / "icc" / "transicc",
        ROOT_DIR / "botmcms" / "icc" / "transicc.exe",
        BACKEND_DIR / "icc" / "transicc",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p

    # 3) Last resort: search everything under /app
    for p in ROOT_DIR.rglob("transicc*"):
        if p.is_file():
            return p

    return None


# --------------------------------------------------------------------
# BASIC ROUTES
# --------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    """
    Serve the main site HTML at /
    """
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    raise HTTPException(status_code=500, detail="index.html not found in container")


# --------------------------------------------------------------------
# LCMS / TRANSICC DEBUG + HEALTH
# --------------------------------------------------------------------
@app.get("/lcms/debug")
def lcms_debug():
    """
    Debug endpoint: shows where the app *thinks* the repo root is,
    what transicc it finds via PATH, and any transicc-like files.
    """
    path_transicc = shutil.which("transicc")
    matches = [str(p) for p in ROOT_DIR.rglob("transicc*")]

    return {
        "root_dir": str(ROOT_DIR),
        "backend_dir": str(BACKEND_DIR),
        "which_transicc": path_transicc,
        "matches": matches,
    }


@app.get("/lcms/health")
def lcms_health():
    """
    Sanity check that the LittleCMS transicc binary is present and runnable.
    """
    transicc_path = find_transicc()

    if transicc_path is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "transicc binary not found on server",
                "root_dir": str(ROOT_DIR),
            },
        )

    try:
        result = subprocess.run(
            [str(transicc_path), "-v"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Failed to run transicc: {e}",
                "path": str(transicc_path),
            },
        )

    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"transicc error: {msg}",
                "path": str(transicc_path),
            },
        )

    return {
        "ok": True,
        "version": result.stdout.strip(),
        "path": str(transicc_path),
    }


# --------------------------------------------------------------------
# COLOR MATCH PLACEHOLDER API
# --------------------------------------------------------------------
class ColorInput(BaseModel):
    """
    Placeholder input model.
    Replace with whatever your real colormatch payload looks like.
    """
    input_type: str  # e.g. "hex", "rgb", "cmyk", "bastone", "paint"
    value: str       # e.g. "#FFC845", "123C", "Behr Some Color"


@app.post("/colormatch")
def colormatch(payload: ColorInput):
    """
    Placeholder colormatch route.
    Right now this just returns hard coded sample data so you can test end to end.
    Once you are ready, delete the dummy body and drop in your LittleCMS logic.
    """

    dummy_result = {
        "input": {
            "type": payload.input_type,
            "value": payload.value,
        },
        "bastone": {
            "name": "Bastone 123C",
            "hex": "#FFC845",
            "lab": [85.0, 18.0, 70.0],
        },
        "matches": [
            {
                "family": "resin",
                "brand": "Siraya Tech",
                "product_line": "Fast",
                "name": "Fast Yellow",
                "hex": "#F8C94D",
                "delta_e": 2.7,
            },
            {
                "family": "filament",
                "brand": "eSun",
                "product_line": "PLA+",
                "name": "PLA+ Warm Yellow",
                "hex": "#FFCC4A",
                "delta_e": 1.3,
            },
        ],
    }

    return dummy_result
