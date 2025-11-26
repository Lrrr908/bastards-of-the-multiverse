from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="BotMCMS API")

# Allow your future frontend (and local dev) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to index.html inside the Docker container
BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "index.html"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    """
    Serve the main site HTML at /
    Render copies index.html into /app, which is BASE_DIR here.
    """
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    raise HTTPException(status_code=500, detail="index.html not found in container")


# ------------- Placeholder models and routes for your color API -------------
# You can change this to match however your LittleCMS code wants input.


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
    Right now this just returns hard-coded sample data so you can test end-to-end.
    Once you are ready, delete the dummy body and drop in your LittleCMS logic.
    """

    # TODO: Replace everything below with real LittleCMS + BotMCMS code.
    # This is just a stub so the endpoint works.
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
