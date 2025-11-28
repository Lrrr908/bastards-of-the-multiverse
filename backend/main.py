from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="BotMCMS API")

# CORS so your front end can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
INDEX_PATH = ROOT_DIR / "index.html"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    raise HTTPException(status_code=500, detail="index.html not found in container")


# ---- DUMMY COLOR MATCH API (NO LCMS YET) ----

class ColorInput(BaseModel):
    input_type: str  # "hex", "rgb", "cmyk", "bastone", etc
    value: str       # "#FFC845", "123C", etc


@app.post("/colormatch")
def colormatch(payload: ColorInput):
    """
    Temporary stub: just echo back input and hard-coded matches
    so your front-end widget can be wired up and tested.
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
