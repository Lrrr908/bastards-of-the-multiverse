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
    Confirm that color conversion is working and show which method.
    """
    try:
        # Check if LittleCMS is available
        try:
            from PIL import ImageCms
            method = "LittleCMS (Professional)"
            has_littlecms = True
        except ImportError:
            method = "Pure Python (Fallback)"
            has_littlecms = False
        
        # Test a conversion
        test_lab = rgb_to_lab(255, 200, 69)
        
        # Test reverse conversion
        test_rgb = lab_to_rgb(test_lab[0], test_lab[1], test_lab[2])
        
        return {
            "ok": True,
            "method": method,
            "has_littlecms": has_littlecms,
            "test_conversion": f"RGB(255,200,69) -> Lab{test_lab}",
            "reverse_test": f"Lab{test_lab} -> RGB{test_rgb}",
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
    Convert RGB to L*a*b* using LittleCMS library for professional accuracy.
    Returns (L, a, b) tuple.
    """
    try:
        from PIL import Image, ImageCms
        import io
        
        # Create an sRGB profile and Lab profile
        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile = ImageCms.createProfile("LAB")
        
        # Create transform from sRGB to Lab
        transform = ImageCms.buildTransform(
            srgb_profile,
            lab_profile,
            "RGB",
            "LAB"
        )
        
        # Create a 1x1 pixel image with the RGB color
        img = Image.new("RGB", (1, 1), (r, g, b))
        
        # Apply the transform
        lab_img = ImageCms.applyTransform(img, transform)
        
        # Get the Lab values
        L, a, b_val = lab_img.getpixel((0, 0))
        
        # Convert from 0-255 range to proper Lab ranges
        L = (L / 255.0) * 100
        a = (a - 128)
        b_val = (b_val - 128)
        
        return (round(L, 2), round(a, 2), round(b_val, 2))
        
    except ImportError:
        # Fallback to pure Python if PIL/littlecms not available
        return rgb_to_lab_python(r, g, b)
    except Exception as e:
        # Fallback on any error
        print(f"LittleCMS conversion failed: {e}, using fallback")
        return rgb_to_lab_python(r, g, b)


def rgb_to_lab_python(r: int, g: int, b: int) -> tuple:
    """
    Fallback pure Python RGB to L*a*b* conversion.
    Used if LittleCMS is not available.
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


def lab_to_rgb(L: float, a: float, b: float) -> tuple:
    """
    Convert L*a*b* to RGB (0-255) using LittleCMS for professional accuracy.
    Returns (r, g, b) tuple.
    """
    try:
        from PIL import Image, ImageCms
        
        # Create Lab and sRGB profiles
        lab_profile = ImageCms.createProfile("LAB")
        srgb_profile = ImageCms.createProfile("sRGB")
        
        # Create transform from Lab to sRGB
        transform = ImageCms.buildTransform(
            lab_profile,
            srgb_profile,
            "LAB",
            "RGB"
        )
        
        # Convert Lab values to 0-255 range
        L_byte = int((L / 100.0) * 255)
        a_byte = int(a + 128)
        b_byte = int(b + 128)
        
        # Clamp to valid range
        L_byte = max(0, min(255, L_byte))
        a_byte = max(0, min(255, a_byte))
        b_byte = max(0, min(255, b_byte))
        
        # Create a 1x1 pixel Lab image
        lab_img = Image.new("LAB", (1, 1), (L_byte, a_byte, b_byte))
        
        # Apply the transform
        rgb_img = ImageCms.applyTransform(lab_img, transform)
        
        # Get the RGB values
        r, g, b_val = rgb_img.getpixel((0, 0))
        
        return (r, g, b_val)
        
    except ImportError:
        # Fallback to pure Python if PIL/littlecms not available
        return lab_to_rgb_python(L, a, b)
    except Exception as e:
        # Fallback on any error
        print(f"LittleCMS conversion failed: {e}, using fallback")
        return lab_to_rgb_python(L, a, b)


def lab_to_rgb_python(L: float, a: float, b: float) -> tuple:
    """
    Convert L*a*b* to RGB (0-255).
    Returns (r, g, b) tuple.
    """
    # Convert Lab to XYZ
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    def f_inv(t):
        delta = 6/29
        if t > delta:
            return t**3
        else:
            return 3 * delta**2 * (t - 4/29)
    
    x = f_inv(fx) * 0.95047
    y = f_inv(fy) * 1.00000
    z = f_inv(fz) * 1.08883
    
    # Convert XYZ to linear RGB
    r_lin = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g_lin = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b_lin = x * 0.0556434 + y * -0.2040259 + z * 1.0572252
    
    # Convert linear RGB to sRGB
    def from_linear(c):
        if c <= 0.0031308:
            return 12.92 * c
        else:
            return 1.055 * (c ** (1/2.4)) - 0.055
    
    r_norm = from_linear(r_lin)
    g_norm = from_linear(g_lin)
    b_norm = from_linear(b_lin)
    
    # Clamp to 0-255
    r = max(0, min(255, int(round(r_norm * 255))))
    g = max(0, min(255, int(round(g_norm * 255))))
    b = max(0, min(255, int(round(b_norm * 255))))
    
    return (r, g, b)


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


@app.get("/libraries/{library_name}")
def get_library(library_name: str):
    """
    Load a color library JSON file and add hex colors from Lab values.
    Supports: pantone, behr, sherwin, bm (benjamin moore)
    """
    import json
    
    # Map library names to file paths
    library_files = {
        "pantone": ROOT_DIR / "botmcms" / "libraries" / "pantone.json",
        "behr": ROOT_DIR / "botmcms" / "libraries" / "behr.json",
        "sherwin": ROOT_DIR / "botmcms" / "libraries" / "sherwin.json",
        "bm": ROOT_DIR / "botmcms" / "libraries" / "gracol2013_gamut_boundary.json",
    }
    
    if library_name not in library_files:
        raise HTTPException(
            status_code=404,
            detail=f"Library '{library_name}' not found. Available: {list(library_files.keys())}"
        )
    
    file_path = library_files[library_name]
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Library file not found: {file_path}"
        )
    
    try:
        with open(file_path, 'r') as f:
            colors = json.load(f)
        
        # Process each color: add hex if missing, convert Lab format
        processed_colors = []
        for color in colors:
            # Handle different Lab formats
            if "lab" in color:
                lab = color["lab"]
            elif "L" in color and "a" in color and "b" in color:
                lab = [color["L"], color["a"], color["b"]]
            else:
                continue  # Skip colors without Lab data
            
            # Generate hex from Lab if not present
            if "hex" not in color:
                r, g, b = lab_to_rgb(lab[0], lab[1], lab[2])
                hex_color = rgb_to_hex(r, g, b)
            else:
                hex_color = color["hex"]
            
            processed_colors.append({
                "name": color.get("name", "Unknown"),
                "hex": hex_color,
                "lab": lab
            })
        
        return {
            "library": library_name,
            "count": len(processed_colors),
            "colors": processed_colors
        }
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse JSON file: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading library: {str(e)}"
        )
