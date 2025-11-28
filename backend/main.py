from pathlib import Path
import subprocess
import shutil

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="BotMCMS API")

# Allow your frontend and dev environments to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BACKEND_DIR = Path(__file__).resolve().parent      # /app/backend
ROOT_DIR = BACKEND_DIR.parent                      # /app
INDEX_PATH = ROOT_DIR / "index.html"               # /app/index.html


# -------------------------------------------------
# Basic health and index
# -------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    """
    Serve the main site HTML at /.
    """
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    raise HTTPException(status_code=500, detail="index.html not found in container")


# -------------------------------------------------
# LCMS / transicc helpers
# -------------------------------------------------

def get_transicc_path() -> Path:
    """
    Find the transicc binary inside the container.

    This expects Dockerfile to have installed lcms2-utils,
    which provides /usr/bin/transicc.
    """
    found = shutil.which("transicc")
    if not found:
        raise HTTPException(
            status_code=500,
            detail="transicc not found on PATH. lcms2-utils is probably not installed in the container.",
        )
    return Path(found)


@app.get("/lcms/health")
def lcms_health():
    """
    Confirm that LittleCMS (transicc) is installed and callable in the container.
    """
    transicc = get_transicc_path()

    try:
        result = subprocess.run(
            [str(transicc), "-v"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run transicc: {e}",
        )

    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise HTTPException(
            status_code=500,
            detail=f"transicc returned error: {msg}",
        )

    return {
        "ok": True,
        "path": str(transicc),
        "version": result.stdout.strip(),
    }


# -------------------------------------------------
# Color match models and endpoints (dummy for now)
# -------------------------------------------------

class ColorInput(BaseModel):
    """
    Placeholder input model for colormatch.

    Later you can expand this to match your real payload
    shape from the Color Match widget.
    """
    input_type: str  # "hex", "rgb", "cmyk", "bastone", "paint", etc
    value: str       # "#FFC845", "123C", "Behr 3405", etc


@app.post("/colormatch")
def colormatch(payload: ColorInput):
    """
    Temporary dummy colormatch route.

    Right now this does not call LittleCMS. It just returns
    a fixed sample response so you can test end to end.

    Once /lcms/health is working, you can replace this body
    with real LCMS based color math.
    """
    return {
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


@app.get("/colormatch")
def colormatch_get(
    input_type: str = Query(..., description="hex, rgb, cmyk, bastone, paint"),
    value: str = Query(..., description="e.g. #FFC845, 123C, Behr 3405"),
):
    """
    Convenience GET wrapper so you can hit colormatch directly in a browser:
    /colormatch?input_type=hex&value=%23FFC845
    """
    return colormatch(ColorInput(input_type=input_type, value=value))
