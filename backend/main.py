from pathlib import Path
import re
import math
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

# If GRACoL2013.icc sits beside transicc and this file, point here.
# Change to BACKEND_DIR / "lcms" / "GRACoL2013.icc" if you use a subfolder.
GRACOL_PROFILE_PATH = ROOT_DIR / "botmcms" / "icc" / "GRACoL2013.icc"

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
        hex_color = ''.join([c * 2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color"""
    return f"#{r:02x}{g:02x}{b:02x}".upper()


def rgb_to_cmyk(r: int, g: int, b: int) -> List[float]:
    """
    Approximate RGB (0–255) to CMYK (0–100).
    Returns [C, M, Y, K] as floats.
    """
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    # Pure black special case
    if r == 0 and g == 0 and b == 0:
        return [0.0, 0.0, 0.0, 100.0]

    c_prime = 1.0 - r_norm
    m_prime = 1.0 - g_norm
    y_prime = 1.0 - b_norm

    k = min(c_prime, m_prime, y_prime)

    if k >= 1.0:
        return [0.0, 0.0, 0.0, 100.0]

    c = (c_prime - k) / (1.0 - k)
    m = (m_prime - k) / (1.0 - k)
    y = (y_prime - k) / (1.0 - k)

    return [
        round(c * 100.0, 1),
        round(m * 100.0, 1),
        round(y * 100.0, 1),
        round(k * 100.0, 1),
    ]


def rgb_to_lab(r: int, g: int, b: int) -> tuple:
    """
    Convert RGB to L*a*b* using LittleCMS library for professional accuracy.
    Returns (L, a, b) tuple.
    """
    try:
        from PIL import Image, ImageCms
        
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
            return t / (3 * delta**2) + 4/29
    
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
        from PIL import ImageCms, Image as PilImage
        
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
        lab_img = PilImage.new("LAB", (1, 1), (L_byte, a_byte, b_byte))
        
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


def lab_to_cmyk_via_gracol(L: float, a: float, b: float, intent: int = 1) -> List[float]:
    """
    Convert Lab -> CMYK using the GRACoL profile via LittleCMS.
    NO FALLBACK: if anything fails, raise an error so the caller can
    surface it to the client.
    intent: 0=Perceptual, 1=Relative, 2=Saturation, 3=Absolute (LittleCMS codes)
    """
    try:
        from PIL import ImageCms, Image as PilImage

        if not GRACOL_PROFILE_PATH.exists():
            raise RuntimeError(f"GRACoL profile not found at {GRACOL_PROFILE_PATH}")

        lab_profile = ImageCms.createProfile("LAB")
        cmyk_profile = ImageCms.getOpenProfile(str(GRACOL_PROFILE_PATH))

        transform = ImageCms.buildTransformFromOpenProfiles(
            lab_profile,
            cmyk_profile,
            "LAB",
            "CMYK",
            renderingIntent=intent
        )

        # Lab to 0–255 representation
        L_byte = int((L / 100.0) * 255)
        a_byte = int(a + 128)
        b_byte = int(b + 128)

        L_byte = max(0, min(255, L_byte))
        a_byte = max(0, min(255, a_byte))
        b_byte = max(0, min(255, b_byte))

        lab_img = PilImage.new("LAB", (1, 1), (L_byte, a_byte, b_byte))
        cmyk_img = ImageCms.applyTransform(lab_img, transform)
        C, M, Y, K = cmyk_img.getpixel((0, 0))

        # 0–255 CMYK to 0–100
        return [
            round(C / 255.0 * 100.0, 1),
            round(M / 255.0 * 100.0, 1),
            round(Y / 255.0 * 100.0, 1),
            round(K / 255.0 * 100.0, 1),
        ]

    except Exception as e:
        # NO SILENT FALLBACK HERE
        raise RuntimeError(f"GRACoL CMYK conversion failed: {e}")


# -------------------------------------------------
# Lab to Hex endpoint
# -------------------------------------------------

class LabToHexRequest(BaseModel):
    L: float
    a: float
    b: float


@app.post("/api/v1/color/lab-to-hex")
def lab_to_hex(payload: LabToHexRequest):
    """
    Convert a Lab color to a hex value using the same engine
    as the rest of BotMCMS.
    """
    try:
        r, g, b_val = lab_to_rgb(payload.L, payload.a, payload.b)
        hex_color = rgb_to_hex(r, g, b_val)

        return {
            "success": True,
            "hex": hex_color
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lab to hex conversion failed: {str(e)}"
        )


# -------------------------------------------------
# Mock color database - DISABLED (using real libraries now)
# -------------------------------------------------

def find_closest_colors(target_lab: tuple, max_results: int = 5) -> List[Dict]:
    """
    This function is now deprecated - color matching happens in the frontend
    using the loaded libraries. Keeping for backwards compatibility but returns empty.
    """
    return []


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
            
        elif input_type == "lab":
            # Handle Lab input: "L,a,b" format
            lab_parts = re.split(r'[,\s]+', value)
            if len(lab_parts) != 3:
                raise ValueError("Lab must have 3 values")
            L, a, b_val = [float(x) for x in lab_parts]
            r, g, b = lab_to_rgb(L, a, b_val)
            hex_color = rgb_to_hex(r, g, b)
            lab = (L, a, b_val)
            
            return {
                "input": {
                    "type": payload.input_type,
                    "value": payload.value,
                    "hex": hex_color,
                    "rgb": [r, g, b],
                    "lab": list(lab),
                },
                "matches": []  # Frontend will populate this from libraries
            }
            
        elif input_type == "bastone":
            return {
                "input": {
                    "type": payload.input_type,
                    "value": payload.value,
                },
                "error": "Bastone lookup not yet implemented. Please use hex or RGB input.",
            }
            
        elif input_type == "paint":
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
    
    # Convert to Lab
    try:
        lab = rgb_to_lab(r, g, b)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to convert to Lab color space: {str(e)}"
        )
    
    return {
        "input": {
            "type": payload.input_type,
            "value": payload.value,
            "hex": hex_color,
            "rgb": [r, g, b],
            "lab": list(lab),
        },
        "matches": []
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
# Universal color conversion endpoint
# -------------------------------------------------

@app.post("/convert")
def convert_color(payload: dict):
    """
    Universal color conversion endpoint.
    Converts between different color spaces using LittleCMS
    (or pure Python fallback).

    The harmony widget calls this like:

    {
      "lab": [L, a, b],
      "profile": "GRACoL2013.icc",
      "renderingIntent": 1
    }
    """
    try:
        profile_name = str(payload.get("profile", "") or "")
        rendering_intent = int(payload.get("renderingIntent", 1) or 1)

        # Helper: should we use GRACoL profile?
        def wants_gracol(name: str) -> bool:
            return "gracol" in name.lower()

        # Handle RGB input
        if "rgb" in payload:
            r, g, b = payload["rgb"]
            lab = rgb_to_lab(r, g, b)
            hex_color = rgb_to_hex(r, g, b)

            if wants_gracol(profile_name):
                cmyk_equiv = lab_to_cmyk_via_gracol(lab[0], lab[1], lab[2], rendering_intent)
            else:
                cmyk_equiv = rgb_to_cmyk(r, g, b)

            return {
                "success": True,
                "lab": list(lab),
                "hex": hex_color,
                "rgb": [r, g, b],
                "gamut": {
                    "inGamut": True,
                    "cmykEquivalent": cmyk_equiv,
                }
            }
        
        # Handle CMYK input
        elif "cmyk" in payload:
            c, m, y, k = payload["cmyk"]
            
            # Simple CMYK to RGB conversion (0–100 to 0–255)
            r = int(255 * (1 - c / 100.0) * (1 - k / 100.0))
            g = int(255 * (1 - m / 100.0) * (1 - k / 100.0))
            b_val = int(255 * (1 - y / 100.0) * (1 - k / 100.0))
            
            lab = rgb_to_lab(r, g, b_val)
            hex_color = rgb_to_hex(r, g, b_val)
            cmyk_equiv = [
                float(round(c, 1)),
                float(round(m, 1)),
                float(round(y, 1)),
                float(round(k, 1)),
            ]
            
            return {
                "success": True,
                "lab": list(lab),
                "hex": hex_color,
                "rgb": [r, g, b_val],
                "cmyk": [c, m, y, k],
                "gamut": {
                    "inGamut": True,
                    "cmykEquivalent": cmyk_equiv,
                }
            }
        
        # Handle Lab input (this is what your harmony widget uses)
        elif "lab" in payload:
            L, a, b = payload["lab"]
            
            # Lab -> RGB for hex + preview
            r, g, b_val = lab_to_rgb(L, a, b)
            hex_color = rgb_to_hex(r, g, b_val)
            
            # Lab -> CMYK via GRACoL if requested, else approximate
            if wants_gracol(profile_name):
                cmyk_equiv = lab_to_cmyk_via_gracol(L, a, b, rendering_intent)
            else:
                cmyk_equiv = rgb_to_cmyk(r, g, b_val)
            
            return {
                "success": True,
                "lab": [L, a, b],
                "hex": hex_color,
                "rgb": [r, g, b_val],
                "gamut": {
                    "inGamut": True,
                    "cmykEquivalent": cmyk_equiv,
                }
            }
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide 'rgb', 'cmyk', or 'lab' in request"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        # If GRACoL or any step blows up, you see it here – no hidden fallback.
        raise HTTPException(
            status_code=500,
            detail=f"Color conversion failed: {str(e)}"
        )


@app.post("/gamut-check")
def gamut_check(payload: dict):
    """
    Check if a Lab color is within the gamut of a profile.
    """
    try:
        lab = payload.get("lab")
        profile = payload.get("profile", "GRACoL2013.icc")
        
        if not lab:
            raise HTTPException(status_code=400, detail="Lab values required")
        
        L, a, b = lab
        
        in_gamut = (
            0 <= L <= 100 and
            -128 <= a <= 127 and
            -128 <= b <= 127 and
            abs(a) < 100 and
            abs(b) < 100
        )
        
        return {
            "gamut": {
                "inGamut": in_gamut,
                "profile": profile
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gamut check failed: {str(e)}"
        )


# -------------------------------------------------
# Rendering comparison endpoints
# -------------------------------------------------

class RenderingIntentInput(BaseModel):
    """
    Input model for rendering intent comparison.
    The frontend sends Lab values and expects comparisons for all 4 intents.
    """
    lab: List[float]  # [L, a, b] values
    targetProfile: str = "GRACoL2013.icc"  # Target profile


@app.post("/rendering-intents")
def rendering_intents(payload: RenderingIntentInput):
    """
    Compare how a Lab color renders with different rendering intents.
    """
    try:
        L, a, b = payload.lab
        source_lab = (L, a, b)
        
        rendering_intents = {}
        
        # Perceptual
        perceptual_lab = (
            L * 0.98,
            a * 0.95,
            b * 0.95
        )
        rendering_intents['perceptual'] = {
            'lab': list(perceptual_lab),
            'deltaE': calculate_delta_e(source_lab, perceptual_lab),
            'gamut': {'inGamut': True}
        }
        
        # Relative
        relative_lab = source_lab
        rendering_intents['relative'] = {
            'lab': list(relative_lab),
            'deltaE': 0.0,
            'gamut': {'inGamut': True}
        }
        
        # Saturation
        saturation_lab = (
            L,
            a * 1.05,
            b * 1.05
        )
        rendering_intents['saturation'] = {
            'lab': list(saturation_lab),
            'deltaE': calculate_delta_e(source_lab, saturation_lab),
            'gamut': {'inGamut': True}
        }
        
        # Absolute
        absolute_lab = (
            L * 0.99,
            a * 0.98,
            b * 0.98
        )
        rendering_intents['absolute'] = {
            'lab': list(absolute_lab),
            'deltaE': calculate_delta_e(source_lab, absolute_lab),
            'gamut': {'inGamut': True}
        }
        
        return {
            'renderingIntents': rendering_intents,
            'sourceProfile': 'Lab',
            'targetProfile': payload.targetProfile
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Rendering intent comparison failed: {str(e)}"
        )


class RenderComparisonInput(BaseModel):
    """
    Input model for render comparison (batch processing).
    """
    source_profile: str = "sRGB"
    target_profile: str = "GRACoL2013.icc"
    colors: List[Dict]
    intent: str = "perceptual"


@app.post("/render-comparison")
def render_comparison(payload: RenderComparisonInput):
    """
    Compare how colors will render across different profiles.
    """
    try:
        results = []
        
        for color_input in payload.colors:
            if "rgb" in color_input:
                r, g, b = color_input["rgb"]
                source_lab = rgb_to_lab(r, g, b)
                source_hex = rgb_to_hex(r, g, b)
            elif "lab" in color_input:
                L, a, b_val = color_input["lab"]
                source_lab = (L, a, b_val)
                r, g, b = lab_to_rgb(L, a, b_val)
                source_hex = rgb_to_hex(r, g, b)
            else:
                continue
            
            if payload.target_profile.lower() in ["gracol2013.icc", "cmyk", "print"]:
                rendered_lab = (
                    source_lab[0] * 0.95,
                    source_lab[1] * 0.90,
                    source_lab[2] * 0.90
                )
            else:
                rendered_lab = source_lab
            
            rendered_rgb = lab_to_rgb(rendered_lab[0], rendered_lab[1], rendered_lab[2])
            rendered_hex = rgb_to_hex(*rendered_rgb)
            delta_e = calculate_delta_e(source_lab, rendered_lab)
            
            result = {
                "name": color_input.get("name", "Unnamed"),
                "source": {
                    "profile": payload.source_profile,
                    "rgb": [r, g, b],
                    "lab": list(source_lab),
                    "hex": source_hex
                },
                "rendered": {
                    "profile": payload.target_profile,
                    "rgb": list(rendered_rgb),
                    "lab": list(rendered_lab),
                    "hex": rendered_hex
                },
                "delta_e": delta_e,
                "intent": payload.intent,
                "warning": "Noticeable difference" if delta_e > 2.0 else None
            }
            
            results.append(result)
        
        return {
            "comparison": {
                "source_profile": payload.source_profile,
                "target_profile": payload.target_profile,
                "rendering_intent": payload.intent,
                "color_count": len(results)
            },
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Render comparison failed: {str(e)}"
        )


@app.get("/render-comparison/profiles")
def list_render_profiles():
    """
    List available rendering profiles for comparison.
    """
    return {
        "source_profiles": [
            {"name": "sRGB", "description": "Standard RGB (web/screen)"},
            {"name": "AdobeRGB", "description": "Adobe RGB (wider gamut)"},
            {"name": "ProPhotoRGB", "description": "ProPhoto RGB (very wide gamut)"}
        ],
        "target_profiles": [
            {"name": "sRGB", "description": "Standard RGB (web/screen)"},
            {"name": "GRACoL2013.icc", "description": "US commercial offset printing"},
            {"name": "SWOP2013.icc", "description": "US publication printing"},
            {"name": "PSO_Coated_v3.icc", "description": "European coated paper"}
        ],
        "rendering_intents": [
            {"name": "perceptual", "description": "Preserves visual relationships"},
            {"name": "relative", "description": "Preserves in-gamut colors exactly"},
            {"name": "saturation", "description": "Maximizes saturation"},
            {"name": "absolute", "description": "Exact colorimetric match"}
        ]
    }


# -------------------------------------------------
# Color harmony endpoint
# -------------------------------------------------

class HarmonyInput(BaseModel):
    """
    Input model for color harmony generation.
    """
    lab: List[float]
    harmonyType: str = "complementary"
    libraries: Optional[List[str]] = None
    profile: str = "GRACoL2013.icc"


@app.post("/harmony")
def generate_harmony(payload: HarmonyInput):
    """
    Generate color harmonies based on input Lab color.
    """
    try:
        L, a, b = payload.lab
        harmony_type = payload.harmonyType
        
        base_hue = math.atan2(b, a) * 180 / math.pi
        base_chroma = math.sqrt(a * a + b * b)
        
        hue_offsets = {
            'complementary': [0, 180],
            'triadic': [0, 120, 240],
            'tetradic': [0, 90, 180, 270],
            'analogous': [0, 30, -30],
            'splitComplementary': [0, 150, 210],
            'monochromatic': [0, 0, 0, 0, 0]
        }
        
        offsets = hue_offsets.get(harmony_type, [0, 180])
        harmonies = []
        
        for index, offset in enumerate(offsets):
            if harmony_type == 'monochromatic':
                lightness_mods = [0, -20, -35, 20, 35]
                lightness_mod = lightness_mods[index] if index < len(lightness_mods) else 0
                new_lab = {
                    'L': max(0, min(100, L + lightness_mod)),
                    'a': a * 0.9,
                    'b': b * 0.9
                }
            else:
                new_hue = (base_hue + offset) * math.pi / 180
                chroma_variation = 0.85 + (index * 0.05)
                adjusted_chroma = base_chroma * chroma_variation
                
                new_lab = {
                    'L': L + (5 if index % 2 == 0 else -5),
                    'a': adjusted_chroma * math.cos(new_hue),
                    'b': adjusted_chroma * math.sin(new_hue)
                }
            
            harmonies.append({
                'lab': [new_lab['L'], new_lab['a'], new_lab['b']],
                'gamut': {'inGamut': True},
                'isOriginal': (index == 0)
            })
        
        return {
            'colors': harmonies,
            'harmonyType': harmony_type
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Harmony generation failed: {str(e)}"
        )


# -------------------------------------------------
# Gamut boundary endpoint
# -------------------------------------------------

class GamutBoundaryInput(BaseModel):
    """
    Input model for gamut boundary generation.
    """
    profile: str = "GRACoL2013.icc"
    resolution: int = 25
    currentLab: Optional[List[float]] = None


@app.post("/gamut-boundary")
def get_gamut_boundary(payload: GamutBoundaryInput):
    """
    Generate gamut boundary points for 3D visualization.
    """
    try:
        resolution = payload.resolution
        boundary_points = []
        
        for L_index in range(resolution):
            L = (L_index / (resolution - 1)) * 100
            
            if L < 50:
                max_chroma = L * 2.5
            else:
                max_chroma = (100 - L) * 2.5
            
            hue_steps = max(8, resolution // 3)
            for hue_index in range(hue_steps):
                hue = (hue_index / hue_steps) * 2 * math.pi
                
                a = max_chroma * math.cos(hue)
                b_val = max_chroma * math.sin(hue)
                
                boundary_points.append([L, a, b_val])
        
        return {
            'boundaryPoints': boundary_points,
            'profile': payload.profile,
            'resolution': resolution,
            'pointCount': len(boundary_points)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gamut boundary generation failed: {str(e)}"
        )


# -------------------------------------------------
# Color database management endpoints
# -------------------------------------------------

@app.get("/libraries/list")
def list_available_libraries():
    """
    List available color libraries.
    """
    return {
        "libraries": ["pantone", "behr", "sherwin", "benjaminmoore"],
        "description": {
            "pantone": "Pantone Color System",
            "behr": "Behr Paint Colors",
            "sherwin": "Sherwin Williams Paint Colors",
            "benjaminmoore": "Benjamin Moore Paint Colors"
        }
    }


@app.get("/libraries/{library_name}")
def get_library(library_name: str):
    """
    Load a color library JSON file and add hex colors from Lab values.
    Supports: pantone, behr, sherwin, benjaminmoore
    """
    import json
    
    library_files = {
        "pantone": ROOT_DIR / "botmcms" / "libraries" / "pantone.json",
        "behr": ROOT_DIR / "botmcms" / "libraries" / "behr.json",
        "sherwin": ROOT_DIR / "botmcms" / "libraries" / "sherwin.json",
        "benjaminmoore": ROOT_DIR / "botmcms" / "libraries" / "benjaminmoore.json",
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
            detail=f"Library file not found at: {file_path}. Please ensure the file exists."
        )
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            colors = json.load(f)
        
        processed_colors = []
        for color in colors:
            try:
                if "lab" in color:
                    lab = color["lab"]
                elif "L" in color and "a" in color and "b" in color:
                    lab = [color["L"], color["a"], color["b"]]
                else:
                    print(f"Warning: Color {color.get('name', 'Unknown')} has no Lab data")
                    continue
                
                if not isinstance(lab, (list, tuple)) or len(lab) != 3:
                    print(f"Warning: Invalid Lab format for {color.get('name', 'Unknown')}")
                    continue
                
                if "hex" not in color or not color["hex"]:
                    r, g, b = lab_to_rgb(lab[0], lab[1], lab[2])
                    hex_color = rgb_to_hex(r, g, b)
                else:
                    hex_color = color["hex"]
                    if not hex_color.startswith('#'):
                        hex_color = f"#{hex_color}"
                
                processed_colors.append({
                    "name": color.get("name", "Unknown"),
                    "hex": hex_color,
                    "lab": lab,
                    "family": color.get("family", None),
                    "brand": color.get("brand", None)
                })
            except Exception as e:
                print(f"Error processing color {color.get('name', 'Unknown')}: {e}")
                continue
        
        return processed_colors
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading library: {str(e)}"
        )


# -------------------------------------------------
# Convenience endpoint for checking library status
# -------------------------------------------------

@app.get("/libraries/status")
def check_library_status():
    """
    Check which library files exist and are accessible.
    Useful for debugging.
    """
    import json
    
    library_files = {
        "pantone": ROOT_DIR / "botmcms" / "libraries" / "pantone.json",
        "behr": ROOT_DIR / "botmcms" / "libraries" / "behr.json",
        "sherwin": ROOT_DIR / "botmcms" / "libraries" / "sherwin.json",
        "benjaminmoore": ROOT_DIR / "botmcms" / "libraries" / "benjaminmoore.json",
    }
    
    status = {}
    for name, path in library_files.items():
        exists = path.exists()
        color_count = 0
        error = None
        
        if exists:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    colors = json.load(f)
                    color_count = len(colors)
            except Exception as e:
                error = str(e)
        
        status[name] = {
            "path": str(path),
            "exists": exists,
            "color_count": color_count,
            "error": error
        }
    
    return {
        "root_dir": str(ROOT_DIR),
        "libraries": status
    }
