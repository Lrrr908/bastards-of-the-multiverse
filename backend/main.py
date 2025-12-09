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
# Mock color database - DISABLED (using real libraries now)
# -------------------------------------------------

# Mock database removed - we're using real color libraries from JSON files
# The /colormatch endpoint now only returns the input color data
# The frontend will search through the loaded libraries

# COLOR_DATABASE = [
#     {
#         "family": "resin",
#         "brand": "Siraya Tech",
#         "product_line": "Fast",
#         "name": "Fast Yellow",
#         "hex": "#F8C94D",
#         "lab": (84.5, 15.2, 68.3)
#     },
#     ...
# ]


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
    
    # Return only the input data - frontend will search libraries
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
    Converts between different color spaces using LittleCMS.
    """
    try:
        # Handle RGB input
        if "rgb" in payload:
            r, g, b = payload["rgb"]
            lab = rgb_to_lab(r, g, b)
            hex_color = rgb_to_hex(r, g, b)
            
            return {
                "lab": list(lab),
                "hex": hex_color,
                "rgb": [r, g, b],
                "gamut": {
                    "inGamut": True  # Simplified for now
                }
            }
        
        # Handle CMYK input
        elif "cmyk" in payload:
            c, m, y, k = payload["cmyk"]
            
            # Simple CMYK to RGB conversion
            r = int(255 * (1 - c/100) * (1 - k/100))
            g = int(255 * (1 - m/100) * (1 - k/100))
            b_val = int(255 * (1 - y/100) * (1 - k/100))
            
            lab = rgb_to_lab(r, g, b_val)
            hex_color = rgb_to_hex(r, g, b_val)
            
            return {
                "lab": list(lab),
                "hex": hex_color,
                "rgb": [r, g, b_val],
                "cmyk": [c, m, y, k],
                "gamut": {
                    "inGamut": True,
                    "cmykEquivalent": [c, m, y, k]
                }
            }
        
        # Handle Lab input
        elif "lab" in payload:
            L, a, b = payload["lab"]
            r, g, b_val = lab_to_rgb(L, a, b)
            hex_color = rgb_to_hex(r, g, b_val)
            
            return {
                "lab": [L, a, b],
                "hex": hex_color,
                "rgb": [r, g, b_val],
                "gamut": {
                    "inGamut": True
                }
            }
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide 'rgb', 'cmyk', or 'lab' in request"
            )
            
    except Exception as e:
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
        
        # For now, return a simple in-gamut check
        # Real implementation would use ICC profiles
        L, a, b = lab
        
        # Simple heuristic: colors are "in gamut" if they're not too extreme
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
    
    Returns all 4 rendering intents (perceptual, relative, saturation, absolute)
    with their Lab values and Delta E from the original.
    
    Expected by the frontend colormatch widget.
    """
    try:
        L, a, b = payload.lab
        source_lab = (L, a, b)
        
        # For now, simulate rendering intent transformations
        # In a real implementation, this would use ICC profiles with different intents
        rendering_intents = {}
        
        # Perceptual - slight compression for out-of-gamut colors
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
        
        # Relative Colorimetric - preserve in-gamut colors exactly
        relative_lab = source_lab
        rendering_intents['relative'] = {
            'lab': list(relative_lab),
            'deltaE': 0.0,
            'gamut': {'inGamut': True}
        }
        
        # Saturation - boost chroma slightly
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
        
        # Absolute Colorimetric - adjust for paper white
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
    source_profile: str = "sRGB"  # Source color profile
    target_profile: str = "GRACoL2013.icc"  # Target rendering profile
    colors: List[Dict]  # List of colors to compare, each with rgb or lab
    intent: str = "perceptual"  # Rendering intent: perceptual, relative, saturation, absolute


@app.post("/render-comparison")
def render_comparison(payload: RenderComparisonInput):
    """
    Compare how colors will render across different profiles.
    
    Takes a list of colors and shows how they would appear when converted
    between different color profiles (e.g., sRGB screen vs. CMYK print).
    
    Returns the original color and the rendered version with Delta E calculations.
    """
    try:
        results = []
        
        for color_input in payload.colors:
            # Get source color
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
            
            # For now, simulate rendering by applying a simple transformation
            # In a full implementation, this would use ICC profiles
            if payload.target_profile.lower() in ["gracol2013.icc", "cmyk", "print"]:
                # Simulate CMYK gamut compression
                # Colors tend to get slightly duller in print
                rendered_lab = (
                    source_lab[0] * 0.95,  # Slightly reduce lightness
                    source_lab[1] * 0.90,  # Compress a* slightly
                    source_lab[2] * 0.90   # Compress b* slightly
                )
            else:
                # No transformation for same profile
                rendered_lab = source_lab
            
            # Convert rendered Lab back to RGB
            rendered_rgb = lab_to_rgb(rendered_lab[0], rendered_lab[1], rendered_lab[2])
            rendered_hex = rgb_to_hex(*rendered_rgb)
            
            # Calculate Delta E
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
    lab: List[float]  # [L, a, b] values
    harmonyType: str = "complementary"  # complementary, triadic, tetradic, analogous, splitComplementary, monochromatic
    libraries: Optional[List[str]] = None  # Optional list of libraries to search
    profile: str = "GRACoL2013.icc"


@app.post("/harmony")
def generate_harmony(payload: HarmonyInput):
    """
    Generate color harmonies based on input Lab color.
    
    Returns a set of harmonious colors with their Lab values and gamut information.
    The frontend will match these to library colors.
    """
    try:
        L, a, b = payload.lab
        harmony_type = payload.harmonyType
        
        # Calculate base hue and chroma
        base_hue = math.atan2(b, a) * 180 / math.pi
        base_chroma = math.sqrt(a * a + b * b)
        
        # Define hue offsets for different harmony types
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
                # Vary lightness for monochromatic
                lightness_mods = [0, -20, -35, 20, 35]
                lightness_mod = lightness_mods[index] if index < len(lightness_mods) else 0
                new_lab = {
                    'L': max(0, min(100, L + lightness_mod)),
                    'a': a * 0.9,
                    'b': b * 0.9
                }
            else:
                # Calculate new hue
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
    
    Returns a list of Lab coordinates that represent the gamut boundary
    for the specified ICC profile.
    """
    try:
        import math
        
        resolution = payload.resolution
        boundary_points = []
        
        # Generate a simplified gamut boundary for sRGB-like space
        # This creates a rough approximation - real implementation would use ICC profiles
        
        for L_index in range(resolution):
            L = (L_index / (resolution - 1)) * 100
            
            # Calculate approximate max chroma at this lightness
            # sRGB gamut is roughly egg-shaped
            if L < 50:
                max_chroma = L * 2.5
            else:
                max_chroma = (100 - L) * 2.5
            
            # Generate points around the chroma circle
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
# Color database management endpoints (FIXED)
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
    
    FIXED: Corrected file paths for all libraries.
    """
    import json
    
    # Map library names to file paths - CORRECTED PATHS
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
        
        # Process each color: add hex if missing, convert Lab format
        processed_colors = []
        for color in colors:
            try:
                # Handle different Lab formats
                if "lab" in color:
                    lab = color["lab"]
                elif "L" in color and "a" in color and "b" in color:
                    lab = [color["L"], color["a"], color["b"]]
                else:
                    # Skip colors without Lab data
                    print(f"Warning: Color {color.get('name', 'Unknown')} has no Lab data")
                    continue
                
                # Ensure lab is a list of 3 numbers
                if not isinstance(lab, (list, tuple)) or len(lab) != 3:
                    print(f"Warning: Invalid Lab format for {color.get('name', 'Unknown')}")
                    continue
                
                # Generate hex from Lab if not present
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
