from pathlib import Path
import subprocess

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

# Paths inside the container
APP_DIR = Path(__file__).resolve().parent          # /app/botmcms
ROOT_DIR = APP_DIR.parent                          # /app
INDEX_PATH = ROOT_DIR / "index.html"               # /app/index.html
TRANSICC_PATH = APP_DIR / "icc" / "transicc"       # /app/botmcms/icc/transicc


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/lcms/health")
def lcms_health():
    """
    Simple sanity check that the LittleCMS transicc binary is present
    and runnable on the server.
    """
    if not TRANSICC_PATH.exists():
        raise HTTPException(status_code=500, detail="transicc binary not found on server")

    try:
        result = subprocess.run(
            [str(TRANSICC_PATH), "-v"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run transicc: {e}")

    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise HTTPException(status_code=500, detail=f"transicc error: {msg}")

    return {
        "ok": True,
        "version": result.stdout.strip(),
        "path": str(TRANSICC_PATH),
    }


@app.get("/")
def root():
    """
    Serve the main site HTML at /
    Render copies index.html into /app, which is ROOT_DIR here.
    """
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    raise HTTPException(status_code=500, detail="index.html not found in container")


# ------------- Placeholder models and routes for your color API -------------


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
