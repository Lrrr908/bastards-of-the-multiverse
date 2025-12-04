from pathlib import Path
import re
from typing import Optional, List, Dict

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
ICC_PROFILES_DIR = BACKEND_DIR / "icc_profiles"    # /app/backend/icc_profiles

# Ensure ICC profiles directory exists
ICC_PROFILES_DIR.mkdir(exist_ok=True)


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
# LCMS health check
# -------------------------------------------------

@app.get("/lcms/health")
def lcms_health():
    """
    Confirm that color conversion is working.
    """
    try:
        # Test a simple conversion
        test_lab = rgb_to_lab(255, 200, 69)
        return {
            "ok": True,
            "method": "python-native",
            "test_conversion": f"RGB(255,200,69) -> Lab{test_lab}",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Color conversion failed: {e}",
        )


# -------------------------------------------------
# Color conversion utilities
# -------------------------------------------------

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple (0-255)"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color"""
    return f"#{r:02x}{g:02x}{b:02x}".upper()


def rgb_to_lab(r: int, g: int, b: int) -> tuple:
    """
    Convert RGB to L*a*b* using pure Python color conversion.
    This is accurate and doesn't require external binaries.
    Returns (L, a, b) tuple.
    """
    # Normalize RGB to 0-1
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    # Convert RGB to linear RGB
    def to_linear(c):
        if c <= 0.04045:
            return c / 12.92
        else:
            return ((c + 0.055) / 1.055) ** 2.4
    
    r_lin = to_linear(r_norm)
    g_lin = to_linear(g_norm)
    b_lin = to_linear(b_norm)
    
    # Convert linear RGB to XYZ (using sRGB/D65 matrix)
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
    
    # Normalize for D65 white point
    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883
    
    # Convert XYZ to Lab
    def f(t):
        delta = 6/29
        if t > delta**3:
            return t**(1/3)
        else:
            return t/(3*delta**2) + 4/29
    
    fx = f(x)
    fy = f(y)
    fz = f(z)
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    
    return (round(L, 2), round(a, 2), round(b_val, 2))


def calculate_delta_e(lab1: tuple, lab2: tuple) -> float:
    """
    Calculate Delta E (CIE76) between two Lab colors.
    This is a simple Euclidean distance in Lab space.
    """
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    
    delta_e = ((l2 - l1)**2 + (a2 - a1)**2 + (b2 - b1)**2) ** 0.5
    return round(delta_e, 2)


# -------------------------------------------------
# Mock color database
# -------------------------------------------------

# This would normally come from a database
# For now, here's a sample dataset
COLOR_DATABASE = [
    {
        "family": "resin",
        "brand": "Siraya Tech",
        "product_line": "Fast",
        "name": "Fast Yellow",
        "hex": "#F8C94D",
        "lab": (84.5, 15.2, 68.3)
    },
    {
        "family": "filament",
        "brand": "eSun",
        "product_line": "PLA+",
        "name": "PLA+ Warm Yellow",
        "hex": "#FFCC4A",
        "lab": (85.2, 16.8, 71.5)
    },
    {
        "family": "resin",
        "brand": "Elegoo",
        "product_line": "Standard",
        "name": "Standard Orange",
        "hex": "#FF8C42",
        "lab": (71.3, 35.6, 58.2)
    },
    {
        "family": "filament",
        "brand": "Polymaker",
        "product_line": "PolyLite",
        "name": "PolyLite Red",
        "hex": "#E63946",
        "lab": (52.8, 62.4, 38.9)
    },
    {
        "family": "resin",
        "brand": "Anycubic",
        "product_line": "Standard",
        "name": "Standard Blue",
        "hex": "#457B9D",
        "lab": (51.2, -8.3, -28.4)
    },
]


def find_closest_colors(target_lab: tuple, max_results: int = 5) -> List[Dict]:
    """
    Find the closest colors in the database to the target Lab color.
    Returns list sorted by Delta E (closest first).
    """
    matches = []
    
    for color in COLOR_DATABASE:
        delta_e = calculate_delta_e(target_lab, color["lab"])
        match = {
            **color,
            "delta_e": delta_e
        }
        matches.append(match)
    
    # Sort by Delta E (ascending)
    matches.sort(key=lambda x: x["delta_e"])
    
    return matches[:max_results]


# -------------------------------------------------
# Color match models and endpoints
# -------------------------------------------------

class ColorInput(BaseModel):
    """
    Input model for colormatch.
    """
    input_type: str  # "hex", "rgb", "cmyk", "bastone", "paint", etc
    value: str       # "#FFC845", "255,204,74", "123C", "Behr 3405", etc


@app.post("/colormatch")
def colormatch(payload: ColorInput):
    """
    Color matching endpoint using pure Python for accurate color conversions.
    
    Accepts various input types and returns the closest matching colors
    from the database using Delta E calculations in Lab color space.
    """
    input_type = payload.input_type.lower()
    value = payload.value.strip()
    
    # Convert input to RGB first
    try:
        if input_type == "hex":
            r, g, b = hex_to_rgb(value)
            hex_color = value if value.startswith('#') else f"#{value}"
            
        elif input_type == "rgb":
            # Expect format like "255,204,74" or "255 204 74"
            rgb_parts = re.split(r'[,\s]+', value)
            if len(rgb_parts) != 3:
                raise ValueError("RGB must have 3 values")
            r, g, b = [int(x) for x in rgb_parts]
            hex_color = rgb_to_hex(r, g, b)
            
        elif input_type == "bastone":
            # For Bastone codes, you'd look up in a database
            # For now, return a mock response
            return {
                "input": {
                    "type": payload.input_type,
                    "value": payload.value,
                },
                "error": "Bastone lookup not yet implemented. Please use hex or RGB input.",
            }
            
        elif input_type == "paint":
            # For paint brands, you'd look up in a database
            return {
                "input": {
                    "type": payload.input_type,
                    "value": payload.value,
                },
                "error": "Paint brand lookup not yet implemented. Please use hex or RGB input.",
            }
            
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")
            
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse color input: {str(e)}"
        )
    
    # Validate RGB ranges
    if not all(0 <= val <= 255 for val in [r, g, b]):
        raise HTTPException(
            status_code=400,
            detail="RGB values must be between 0 and 255"
        )
    
    # Convert to Lab using pure Python
    try:
        lab = rgb_to_lab(r, g, b)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to convert to Lab color space: {str(e)}"
        )
    
    # Find closest matches
    matches = find_closest_colors(lab, max_results=5)
    
    return {
        "input": {
            "type": payload.input_type,
            "value": payload.value,
            "hex": hex_color,
            "rgb": [r, g, b],
            "lab": list(lab),
        },
        "matches": matches,
    }


@app.get("/colormatch")
def colormatch_get(
    input_type: str = Query(..., description="hex, rgb, cmyk, bastone, paint"),
    value: str = Query(..., description="e.g. #FFC845, 255,204,74, 123C, Behr 3405"),
):
    """
    Convenience GET wrapper so you can hit colormatch directly in a browser:
    /colormatch?input_type=hex&value=%23FFC845
    """
    return colormatch(ColorInput(input_type=input_type, value=value))


# -------------------------------------------------
# Color database management endpoints
# -------------------------------------------------

@app.get("/colors")
def list_colors():
    """
    List all colors in the database.
    """
    return {
        "count": len(COLOR_DATABASE),
        "colors": COLOR_DATABASE
    }


@app.get("/colors/brands")
def list_brands():
    """
    List all unique brands in the database.
    """
    brands = sorted(set(color["brand"] for color in COLOR_DATABASE))
    return {"brands": brands}


@app.get("/colors/families")
def list_families():
    """
    List all unique families (resin, filament, etc) in the database.
    """
    families = sorted(set(color["family"] for color in COLOR_DATABASE))
    return {"families": families}
