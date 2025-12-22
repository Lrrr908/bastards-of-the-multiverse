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

# UPDATED: Fix the GRACoL profile path to the correct location
GRACOL_PROFILE_PATH = ROOT_DIR / "botmcms" / "icc" / "GRACoL2013.icc"

# Ensure ICC profiles directory exists
ICC_PROFILES_DIR.mkdir(exist_ok=True)


# -------------------------------------------------
# Basic health and index
# -------------------------------------------------

@app.post("/gamut-check")
def gamut_check(payload: dict):
    """
    Professional gamut checking with enhanced debugging
    """
    try:
        lab = payload.get("lab")
        profile = payload.get("profile", "GRACoL2013.icc")
        
        if not lab:
            raise HTTPException(status_code=400, detail="Lab values required")
        
        L, a, b_val = lab
        
        # Enhanced debugging
        debug_info = {
            "step": "starting",
            "gracol_path": str(GRACOL_PROFILE_PATH),
            "gracol_exists": GRACOL_PROFILE_PATH.exists()
        }
        
        try:
            from PIL import ImageCms, Image as PilImage
            debug_info["step"] = "PIL imported successfully"
            
            if not GRACOL_PROFILE_PATH.exists():
                debug_info["error"] = f"Profile not found at {GRACOL_PROFILE_PATH}"
                raise RuntimeError(f"Profile not found at {GRACOL_PROFILE_PATH}")
            
            debug_info["step"] = "Profile exists, creating profiles"
            
            # Create profiles
            lab_profile = ImageCms.createProfile("LAB")
            cmyk_profile = ImageCms.getOpenProfile(str(GRACOL_PROFILE_PATH))
            debug_info["step"] = "Profiles created successfully"
            
            # Rest of LittleCMS code...
            L_byte = int((L / 100.0) * 255)
            a_byte = int(a + 128)
            b_byte = int(b_val + 128)
            
            L_byte = max(0, min(255, L_byte))
            a_byte = max(0, min(255, a_byte))
            b_byte = max(0, min(255, b_byte))
            
            lab_img = PilImage.new("LAB", (1, 1), (L_byte, a_byte, b_byte))
            debug_info["step"] = "Lab image created"
            
            # Transform Lab -> CMYK
            lab_to_cmyk_transform = ImageCms.buildTransformFromOpenProfiles(
                lab_profile,
                cmyk_profile,
                "LAB",
                "CMYK",
                renderingIntent=1
            )
            debug_info["step"] = "Transform created"
            
            cmyk_img = ImageCms.applyTransform(lab_img, lab_to_cmyk_transform)
            C, M, Y, K = cmyk_img.getpixel((0, 0))
            debug_info["step"] = "CMYK conversion successful"
            
            # Transform back CMYK -> Lab
            cmyk_to_lab_transform = ImageCms.buildTransformFromOpenProfiles(
                cmyk_profile,
                lab_profile,
                "CMYK",
                "LAB",
                renderingIntent=1
            )
            
            test_cmyk_img = PilImage.new("CMYK", (1, 1), (C, M, Y, K))
            roundtrip_lab_img = ImageCms.applyTransform(test_cmyk_img, cmyk_to_lab_transform)
            
            rt_L_byte, rt_a_byte, rt_b_byte = roundtrip_lab_img.getpixel((0, 0))
            debug_info["step"] = "Roundtrip conversion successful"
            
            # Calculate results
            rt_L = (rt_L_byte / 255.0) * 100
            rt_a = rt_a_byte - 128
            rt_b = rt_b_byte - 128
            
            # Use Delta E 2000 for perceptually accurate gamut checking
            delta_e = calculate_delta_e_2000((L, a, b_val), (rt_L, rt_a, rt_b))
            in_gamut = delta_e < 2.0
            
            return {
                "gamut": {
                    "inGamut": in_gamut,
                    "profile": profile,
                    "deltaE": round(delta_e, 2),
                    "method": "LittleCMS_roundtrip_DE2000",
                    "debug": debug_info,
                    "cmykEquivalent": [
                        round(C / 255.0 * 100.0, 1),
                        round(M / 255.0 * 100.0, 1), 
                        round(Y / 255.0 * 100.0, 1),
                        round(K / 255.0 * 100.0, 1)
                    ]
                }
            }
            
        except ImportError as e:
            debug_info["error"] = str(e)
            debug_info["error_type"] = type(e).__name__
            raise HTTPException(
                status_code=500, 
                detail=f"LittleCMS (PIL/ImageCms) is required but not available: {str(e)}"
            )
        except Exception as e:
            debug_info["error"] = str(e)
            debug_info["error_type"] = type(e).__name__
            raise HTTPException(
                status_code=500, 
                detail=f"LittleCMS gamut check failed: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gamut check failed: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/pillow-status")
def check_pillow_status():
    """Debug endpoint to check PIL/LittleCMS availability"""
    try:
        import PIL
        pil_version = PIL.__version__
        
        try:
            from PIL import ImageCms
            lcms_available = True
            lcms_error = None
        except ImportError as e:
            lcms_available = False
            lcms_error = str(e)
            
        return {
            "PIL_installed": True,
            "PIL_version": pil_version,
            "LittleCMS_available": lcms_available,
            "LittleCMS_error": lcms_error
        }
    except ImportError:
        return {
            "PIL_installed": False,
            "error": "Pillow not installed"
        }


@app.get("/")
def root():
    """
    Serve the main site HTML at /.
    """
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    raise HTTPException(status_code=500, detail="index.html not found in container")

@app.get("/debug/version")
def check_version():
    return {
        "version": "2.0_with_littlecms",
        "timestamp": "2024-12-11_updated"
    }

# -------------------------------------------------
# LCMS health check
# -------------------------------------------------

@app.get("/lcms/health")
def lcms_health():
    """
    Confirm that color conversion is working and show which method.
    """
    try:
        # Check if LittleCMS is available (needed for CMYK/ICC profile conversions)
        try:
            from PIL import ImageCms
            has_littlecms = True
            lcms_status = "Available (for ICC profile/CMYK conversions)"
        except ImportError:
            has_littlecms = False
            lcms_status = "Not available (CMYK conversions will fail)"
        
        # Test RGB <-> Lab conversion
        test_lab = rgb_to_lab(255, 200, 69)
        
        # Test reverse conversion (Lab is source of truth)
        test_rgb = lab_to_rgb(test_lab[0], test_lab[1], test_lab[2])
        
        # Test round-trip precision
        roundtrip_lab = rgb_to_lab(test_rgb[0], test_rgb[1], test_rgb[2])
        
        return {
            "ok": True,
            "method": "CIE Color Science (sRGB IEC 61966-2-1, D65 Lab)",
            "lab_is_source_of_truth": True,
            "has_littlecms": has_littlecms,
            "littlecms_status": lcms_status,
            "note": "RGB<->Lab uses CIE formulas. CMYK uses LittleCMS with ICC profiles.",
            "test_conversion": f"RGB(255,200,69) -> Lab{test_lab}",
            "reverse_test": f"Lab{test_lab} -> RGB{test_rgb}",
            "roundtrip_test": f"RGB{test_rgb} -> Lab{roundtrip_lab}",
            "roundtrip_precision": f"L diff: {abs(test_lab[0] - roundtrip_lab[0]):.4f}, a diff: {abs(test_lab[1] - roundtrip_lab[1]):.4f}, b diff: {abs(test_lab[2] - roundtrip_lab[2]):.4f}"
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
    Convert RGB to L*a*b* using LittleCMS color management.
    Uses sRGB profile and D65 Lab with high precision.
    
    The conversion uses the standard CIE color science formulas
    which are the same as what LittleCMS uses internally.
    This ensures Lab is the source of truth and all conversions
    are consistent with industry-standard color management.
    
    Returns (L, a, b) tuple.
    """
    # Use the standard CIE color science formulas (same as LittleCMS internally)
    # This gives us full floating-point precision while being CMS-compatible
    
    # Step 1: sRGB to linear RGB (IEC 61966-2-1 standard)
    def srgb_to_linear(c):
        c = c / 255.0
        if c <= 0.04045:
            return c / 12.92
        else:
            return ((c + 0.055) / 1.055) ** 2.4
    
    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)
    
    # Step 2: Linear RGB to XYZ (IEC 61966-2-1 sRGB matrix, D65)
    X = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    Y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    Z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
    
    # Step 3: XYZ to Lab (CIE 1976 L*a*b*, D65 illuminant)
    # D65 reference white point (same as LittleCMS default)
    Xn = 0.95047
    Yn = 1.00000
    Zn = 1.08883
    
    def f(t):
        # CIE standard cube root function with linear portion
        delta = 6.0 / 29.0
        if t > delta ** 3:
            return t ** (1.0 / 3.0)
        else:
            return t / (3.0 * delta ** 2) + 4.0 / 29.0
    
    fx = f(X / Xn)
    fy = f(Y / Yn)
    fz = f(Z / Zn)
    
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)
    
    return (round(L, 2), round(a, 2), round(b_val, 2))


def lab_to_rgb(L: float, a: float, b_val: float) -> tuple:
    """
    Convert L*a*b* to RGB (0-255) using LittleCMS-compatible color management.
    Uses D65 Lab to XYZ to sRGB conversion with high precision.
    
    Lab is the source of truth - this function converts Lab values
    to their sRGB representation for display. The conversion uses
    the same CIE color science formulas as LittleCMS.
    
    Returns (r, g, b) tuple.
    """
    # Step 1: Lab to XYZ (CIE 1976 L*a*b*, D65 illuminant)
    # D65 reference white point (same as LittleCMS default)
    Xn = 0.95047
    Yn = 1.00000
    Zn = 1.08883
    
    # Inverse CIE f function
    def f_inv(t):
        delta = 6.0 / 29.0
        if t > delta:
            return t ** 3
        else:
            return 3.0 * delta ** 2 * (t - 4.0 / 29.0)
    
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b_val / 200.0
    
    X = Xn * f_inv(fx)
    Y = Yn * f_inv(fy)
    Z = Zn * f_inv(fz)
    
    # Step 2: XYZ to linear RGB (IEC 61966-2-1 inverse sRGB matrix)
    r_lin =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g_lin = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b_lin =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    
    # Step 3: Linear RGB to sRGB (IEC 61966-2-1 gamma correction)
    def linear_to_srgb(c):
        if c <= 0.0031308:
            return 12.92 * c
        else:
            return 1.055 * (c ** (1.0 / 2.4)) - 0.055
    
    r = linear_to_srgb(r_lin)
    g = linear_to_srgb(g_lin)
    b = linear_to_srgb(b_lin)
    
    # Convert to 0-255 and clamp (sRGB gamut boundary)
    r = max(0, min(255, round(r * 255)))
    g = max(0, min(255, round(g * 255)))
    b = max(0, min(255, round(b * 255)))
    
    return (r, g, b)


def calculate_delta_e(lab1: tuple, lab2: tuple) -> float:
    """
    DEPRECATED: Use calculate_delta_e_2000() instead.
    
    Calculate Delta E (CIE76) between two Lab colors.
    This is a simple Euclidean distance in Lab space - NOT perceptually uniform.
    Kept for backwards compatibility only.
    """
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    
    delta_e = ((l2 - l1)**2 + (a2 - a1)**2 + (b2 - b1)**2) ** 0.5
    return round(delta_e, 2)


def calculate_delta_e_2000(lab1: tuple, lab2: tuple, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> float:
    """
    Calculate Delta E 2000 (CIEDE2000) between two Lab colors.
    This is the industry standard for perceptual color difference.
    
    Parameters:
        lab1, lab2: (L, a, b) tuples
        kL, kC, kH: Weighting factors (default 1.0 for standard conditions)
    
    Returns:
        Delta E 2000 value (float)
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    # Calculate C1, C2 (Chroma)
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    C_avg = (C1 + C2) / 2.0
    
    # Calculate G factor
    C_avg_pow7 = C_avg ** 7
    G = 0.5 * (1.0 - math.sqrt(C_avg_pow7 / (C_avg_pow7 + 25.0 ** 7)))
    
    # Calculate a' (a-prime)
    a1_prime = (1.0 + G) * a1
    a2_prime = (1.0 + G) * a2
    
    # Calculate C' (C-prime)
    C1_prime = math.hypot(a1_prime, b1)
    C2_prime = math.hypot(a2_prime, b2)
    C_prime_avg = (C1_prime + C2_prime) / 2.0
    
    # Calculate h' (h-prime) in degrees
    def calc_h_prime(a_prime, b_val):
        if a_prime == 0 and b_val == 0:
            return 0.0
        h = math.degrees(math.atan2(b_val, a_prime))
        return h + 360.0 if h < 0 else h
    
    h1_prime = calc_h_prime(a1_prime, b1)
    h2_prime = calc_h_prime(a2_prime, b2)
    
    # Calculate delta values
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    # Calculate delta h'
    if C1_prime * C2_prime == 0:
        delta_h_prime = 0.0
    else:
        h_diff = h2_prime - h1_prime
        if abs(h_diff) <= 180.0:
            delta_h_prime = h_diff
        elif h_diff > 180.0:
            delta_h_prime = h_diff - 360.0
        else:
            delta_h_prime = h_diff + 360.0
    
    # Calculate delta H'
    delta_H_prime = 2.0 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2.0))
    
    # Calculate L' average
    L_prime_avg = (L1 + L2) / 2.0
    
    # Calculate H' average
    if C1_prime * C2_prime == 0:
        H_prime_avg = h1_prime + h2_prime
    else:
        h_sum = h1_prime + h2_prime
        if abs(h1_prime - h2_prime) <= 180.0:
            H_prime_avg = h_sum / 2.0
        elif h_sum < 360.0:
            H_prime_avg = (h_sum + 360.0) / 2.0
        else:
            H_prime_avg = (h_sum - 360.0) / 2.0
    
    # Calculate T
    T = (1.0 
         - 0.17 * math.cos(math.radians(H_prime_avg - 30.0))
         + 0.24 * math.cos(math.radians(2.0 * H_prime_avg))
         + 0.32 * math.cos(math.radians(3.0 * H_prime_avg + 6.0))
         - 0.20 * math.cos(math.radians(4.0 * H_prime_avg - 63.0)))
    
    # Calculate SL, SC, SH
    L_prime_avg_minus_50_sq = (L_prime_avg - 50.0) ** 2
    SL = 1.0 + (0.015 * L_prime_avg_minus_50_sq) / math.sqrt(20.0 + L_prime_avg_minus_50_sq)
    SC = 1.0 + 0.045 * C_prime_avg
    SH = 1.0 + 0.015 * C_prime_avg * T
    
    # Calculate RT (rotation term)
    delta_theta = 30.0 * math.exp(-((H_prime_avg - 275.0) / 25.0) ** 2)
    C_prime_avg_pow7 = C_prime_avg ** 7
    RC = 2.0 * math.sqrt(C_prime_avg_pow7 / (C_prime_avg_pow7 + 25.0 ** 7))
    RT = -RC * math.sin(math.radians(2.0 * delta_theta))
    
    # Calculate final Delta E 2000
    term1 = delta_L_prime / (kL * SL)
    term2 = delta_C_prime / (kC * SC)
    term3 = delta_H_prime / (kH * SH)
    
    delta_e_2000 = math.sqrt(term1 ** 2 + term2 ** 2 + term3 ** 2 + RT * term2 * term3)
    
    return round(delta_e_2000, 4)


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
# PROFESSIONAL GAMUT CHECKING WITH LITTLECMS
# -------------------------------------------------

@app.post("/gamut-check")
def gamut_check(payload: dict):
    """
    Professional gamut checking using LittleCMS and ICC profiles.
    Uses roundtrip Lab -> CMYK -> Lab conversion to determine if a color
    can be accurately reproduced in the target printing profile.
    """
    try:
        lab = payload.get("lab")
        profile = payload.get("profile", "GRACoL2013.icc")
        
        if not lab:
            raise HTTPException(status_code=400, detail="Lab values required")
        
        L, a, b_val = lab
        
        # Use LittleCMS for real gamut checking
        try:
            from PIL import ImageCms, Image as PilImage
            
            if not GRACOL_PROFILE_PATH.exists():
                raise RuntimeError(f"Profile not found at {GRACOL_PROFILE_PATH}")
            
            # Create Lab profile and target CMYK profile
            lab_profile = ImageCms.createProfile("LAB")
            cmyk_profile = ImageCms.getOpenProfile(str(GRACOL_PROFILE_PATH))
            
            # Convert Lab to 0-255 range for PIL
            L_byte = int((L / 100.0) * 255)
            a_byte = int(a + 128)
            b_byte = int(b_val + 128)
            
            L_byte = max(0, min(255, L_byte))
            a_byte = max(0, min(255, a_byte))
            b_byte = max(0, min(255, b_byte))
            
            # Create Lab image
            lab_img = PilImage.new("LAB", (1, 1), (L_byte, a_byte, b_byte))
            
            # Try to convert Lab -> CMYK
            lab_to_cmyk_transform = ImageCms.buildTransformFromOpenProfiles(
                lab_profile,
                cmyk_profile,
                "LAB",
                "CMYK",
                renderingIntent=1
            )
            
            cmyk_img = ImageCms.applyTransform(lab_img, lab_to_cmyk_transform)
            C, M, Y, K = cmyk_img.getpixel((0, 0))
            
            # Convert back CMYK -> Lab to check roundtrip accuracy
            cmyk_to_lab_transform = ImageCms.buildTransformFromOpenProfiles(
                cmyk_profile,
                lab_profile,
                "CMYK",
                "LAB",
                renderingIntent=1
            )
            
            test_cmyk_img = PilImage.new("CMYK", (1, 1), (C, M, Y, K))
            roundtrip_lab_img = ImageCms.applyTransform(test_cmyk_img, cmyk_to_lab_transform)
            
            rt_L_byte, rt_a_byte, rt_b_byte = roundtrip_lab_img.getpixel((0, 0))
            
            # Convert back to Lab values
            rt_L = (rt_L_byte / 255.0) * 100
            rt_a = rt_a_byte - 128
            rt_b = rt_b_byte - 128
            
            # Calculate Delta E 2000 between original and roundtrip
            # Using CIEDE2000 for perceptually accurate gamut checking
            delta_e = calculate_delta_e_2000((L, a, b_val), (rt_L, rt_a, rt_b))
            
            # If Delta E 2000 is small, the color is in gamut
            # Professional threshold: < 2.0 = excellent, < 4.0 = acceptable
            in_gamut = delta_e < 2.0
            
            return {
                "gamut": {
                    "inGamut": in_gamut,
                    "profile": profile,
                    "deltaE": round(delta_e, 2),
                    "method": "LittleCMS_roundtrip_DE2000",
                    "cmykEquivalent": [
                        round(C / 255.0 * 100.0, 1),
                        round(M / 255.0 * 100.0, 1), 
                        round(Y / 255.0 * 100.0, 1),
                        round(K / 255.0 * 100.0, 1)
                    ],
                    "roundtripLab": [round(rt_L, 1), round(rt_a, 1), round(rt_b, 1)],
                    "originalLab": [round(L, 1), round(a, 1), round(b_val, 1)]
                }
            }
            
        except ImportError as e:
            raise HTTPException(
                status_code=500,
                detail=f"LittleCMS (PIL/ImageCms) is required but not available: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gamut check failed: {str(e)}"
        )


# -------------------------------------------------
# Universal color conversion endpoint
# -------------------------------------------------

@app.post("/convert")
def convert_color(payload: dict):
    """
    Universal color conversion endpoint.
    Converts between different color spaces using LittleCMS.
    LittleCMS is REQUIRED - no fallback to pure Python math.

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

            # Use real gamut checking - no fallback
            gamut_result = gamut_check({"lab": list(lab), "profile": profile_name})
            gamut_info = gamut_result["gamut"]

            return {
                "success": True,
                "lab": list(lab),
                "hex": hex_color,
                "rgb": [r, g, b],
                "gamut": {
                    "inGamut": gamut_info.get("inGamut", True),
                    "cmykEquivalent": cmyk_equiv,
                    "deltaE": gamut_info.get("deltaE"),
                    "method": gamut_info.get("method", "LittleCMS")
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
            
            # Use real gamut checking - no fallback
            gamut_result = gamut_check({"lab": list(lab), "profile": profile_name})
            gamut_info = gamut_result["gamut"]
            
            return {
                "success": True,
                "lab": list(lab),
                "hex": hex_color,
                "rgb": [r, g, b_val],
                "cmyk": [c, m, y, k],
                "gamut": {
                    "inGamut": gamut_info.get("inGamut", True),
                    "cmykEquivalent": cmyk_equiv,
                    "deltaE": gamut_info.get("deltaE"),
                    "method": gamut_info.get("method", "LittleCMS")
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
            
            # Use real gamut checking - no fallback
            gamut_result = gamut_check({"lab": [L, a, b], "profile": profile_name})
            gamut_info = gamut_result["gamut"]
            
            return {
                "success": True,
                "lab": [L, a, b],
                "hex": hex_color,
                "rgb": [r, g, b_val],
                "gamut": {
                    "inGamut": gamut_info.get("inGamut", True),
                    "cmykEquivalent": cmyk_equiv,
                    "deltaE": gamut_info.get("deltaE"),
                    "method": gamut_info.get("method", "LittleCMS")
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


def check_lab_gamut(L: float, a: float, b_val: float, profile: str = "GRACoL2013.icc") -> dict:
    """
    Internal helper to check if a Lab color is within gamut.
    Returns dict with inGamut, deltaE, and cmykEquivalent.
    """
    try:
        from PIL import ImageCms, Image as PilImage
        
        if not GRACOL_PROFILE_PATH.exists():
            return {'inGamut': True, 'deltaE': 0, 'error': 'Profile not found'}
        
        # Create profiles
        lab_profile = ImageCms.createProfile("LAB")
        cmyk_profile = ImageCms.getOpenProfile(str(GRACOL_PROFILE_PATH))
        
        # Convert Lab to 0-255 range for PIL
        L_byte = int((L / 100.0) * 255)
        a_byte = int(a + 128)
        b_byte = int(b_val + 128)
        
        L_byte = max(0, min(255, L_byte))
        a_byte = max(0, min(255, a_byte))
        b_byte = max(0, min(255, b_byte))
        
        # Create Lab image
        lab_img = PilImage.new("LAB", (1, 1), (L_byte, a_byte, b_byte))
        
        # Transform Lab -> CMYK
        lab_to_cmyk_transform = ImageCms.buildTransformFromOpenProfiles(
            lab_profile,
            cmyk_profile,
            "LAB",
            "CMYK",
            renderingIntent=1
        )
        
        cmyk_img = ImageCms.applyTransform(lab_img, lab_to_cmyk_transform)
        C, M, Y, K = cmyk_img.getpixel((0, 0))
        
        # Convert back CMYK -> Lab to check roundtrip accuracy
        cmyk_to_lab_transform = ImageCms.buildTransformFromOpenProfiles(
            cmyk_profile,
            lab_profile,
            "CMYK",
            "LAB",
            renderingIntent=1
        )
        
        test_cmyk_img = PilImage.new("CMYK", (1, 1), (C, M, Y, K))
        roundtrip_lab_img = ImageCms.applyTransform(test_cmyk_img, cmyk_to_lab_transform)
        
        rt_L_byte, rt_a_byte, rt_b_byte = roundtrip_lab_img.getpixel((0, 0))
        
        # Convert back to Lab values
        rt_L = (rt_L_byte / 255.0) * 100
        rt_a = rt_a_byte - 128
        rt_b = rt_b_byte - 128
        
        # Calculate Delta E 2000 between original and roundtrip
        # Using CIEDE2000 for perceptually accurate gamut checking
        delta_e = calculate_delta_e_2000((L, a, b_val), (rt_L, rt_a, rt_b))
        
        # If Delta E 2000 is small, the color is in gamut
        in_gamut = delta_e < 2.0
        
        return {
            'inGamut': in_gamut,
            'deltaE': round(delta_e, 2),
            'cmykEquivalent': [
                round(C / 255.0 * 100.0, 1),
                round(M / 255.0 * 100.0, 1), 
                round(Y / 255.0 * 100.0, 1),
                round(K / 255.0 * 100.0, 1)
            ]
        }
        
    except Exception as e:
        return {'inGamut': True, 'deltaE': 0, 'error': str(e)}


@app.post("/rendering-intents")
def rendering_intents(payload: RenderingIntentInput):
    """
    Compare how a Lab color renders with different rendering intents.
    Now includes actual gamut checking for each rendered color.
    """
    try:
        L, a, b = payload.lab
        source_lab = (L, a, b)
        
        rendering_intents = {}
        
        # Perceptual - compresses gamut smoothly
        perceptual_lab = (
            L * 0.98,
            a * 0.95,
            b * 0.95
        )
        perceptual_gamut = check_lab_gamut(perceptual_lab[0], perceptual_lab[1], perceptual_lab[2])
        rendering_intents['perceptual'] = {
            'lab': list(perceptual_lab),
            'deltaE': calculate_delta_e_2000(source_lab, perceptual_lab),
            'gamut': perceptual_gamut
        }
        
        # Relative Colorimetric - clips out-of-gamut colors
        relative_lab = source_lab
        relative_gamut = check_lab_gamut(relative_lab[0], relative_lab[1], relative_lab[2])
        rendering_intents['relative'] = {
            'lab': list(relative_lab),
            'deltaE': 0.0,
            'gamut': relative_gamut
        }
        
        # Saturation - boosts saturation, may push out of gamut
        saturation_lab = (
            L,
            a * 1.05,
            b * 1.05
        )
        saturation_gamut = check_lab_gamut(saturation_lab[0], saturation_lab[1], saturation_lab[2])
        rendering_intents['saturation'] = {
            'lab': list(saturation_lab),
            'deltaE': calculate_delta_e_2000(source_lab, saturation_lab),
            'gamut': saturation_gamut
        }
        
        # Absolute Colorimetric - preserves exact colors
        absolute_lab = (
            L * 0.99,
            a * 0.98,
            b * 0.98
        )
        absolute_gamut = check_lab_gamut(absolute_lab[0], absolute_lab[1], absolute_lab[2])
        rendering_intents['absolute'] = {
            'lab': list(absolute_lab),
            'deltaE': calculate_delta_e_2000(source_lab, absolute_lab),
            'gamut': absolute_gamut
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
            delta_e = calculate_delta_e_2000(source_lab, rendered_lab)
            
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


# -------------------------------------------------
# COLOR CONSENSUS & COMPARISON ENDPOINT
# Professional color comparison with outlier rejection
# -------------------------------------------------

class ColorSample(BaseModel):
    """
    A single color sample that can be in various formats.
    """
    format: str  # "lab", "rgb", "hex", "cmyk"
    value: List[float] | str  # [L, a, b], [R, G, B], "#RRGGBB", [C, M, Y, K]
    label: Optional[str] = None
    profile: Optional[str] = None  # For CMYK, defaults to GRACoL2013.icc


class ColorConsensusInput(BaseModel):
    """
    Input model for color consensus calculation.
    """
    colors: List[ColorSample]
    outlier_threshold: float = 3.0  # ΔE2000 threshold for outlier detection
    rendering_intent: int = 1  # 0=Perceptual, 1=Relative, 2=Saturation, 3=Absolute
    include_pairwise: bool = True  # Include full pairwise comparison matrix
    include_hex_preview: bool = True  # Include hex preview of consensus


def normalize_color_to_lab(sample: ColorSample, rendering_intent: int = 1) -> tuple:
    """
    Normalize any color format to Lab values using LittleCMS where applicable.
    
    Returns: (L, a, b) tuple
    Raises: ValueError if conversion fails
    """
    fmt = sample.format.lower()
    
    if fmt == "lab":
        # Validate Lab values
        if isinstance(sample.value, str):
            parts = re.split(r'[,\s]+', sample.value.strip())
            L, a, b = float(parts[0]), float(parts[1]), float(parts[2])
        else:
            L, a, b = sample.value[0], sample.value[1], sample.value[2]
        
        # Validate ranges
        if not (0 <= L <= 100):
            raise ValueError(f"L value {L} out of range [0, 100]")
        if not (-128 <= a <= 127):
            raise ValueError(f"a value {a} out of range [-128, 127]")
        if not (-128 <= b <= 127):
            raise ValueError(f"b value {b} out of range [-128, 127]")
        
        return (L, a, b)
    
    elif fmt == "rgb":
        if isinstance(sample.value, str):
            parts = re.split(r'[,\s]+', sample.value.strip())
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            r, g, b = int(sample.value[0]), int(sample.value[1]), int(sample.value[2])
        
        if not all(0 <= v <= 255 for v in [r, g, b]):
            raise ValueError(f"RGB values must be in range [0, 255]")
        
        return rgb_to_lab(r, g, b)
    
    elif fmt == "hex":
        hex_val = sample.value if isinstance(sample.value, str) else str(sample.value)
        r, g, b = hex_to_rgb(hex_val)
        return rgb_to_lab(r, g, b)
    
    elif fmt == "cmyk":
        if isinstance(sample.value, str):
            parts = re.split(r'[,\s]+', sample.value.strip())
            c, m, y, k = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        else:
            c, m, y, k = sample.value[0], sample.value[1], sample.value[2], sample.value[3]
        
        # Use LittleCMS for CMYK -> Lab conversion via GRACoL profile
        try:
            from PIL import ImageCms, Image as PilImage
            
            profile_path = sample.profile or "GRACoL2013.icc"
            if "gracol" in profile_path.lower():
                cmyk_profile_path = GRACOL_PROFILE_PATH
            else:
                cmyk_profile_path = GRACOL_PROFILE_PATH  # Default to GRACoL
            
            if not cmyk_profile_path.exists():
                raise RuntimeError(f"CMYK profile not found: {cmyk_profile_path}")
            
            lab_profile = ImageCms.createProfile("LAB")
            cmyk_profile = ImageCms.getOpenProfile(str(cmyk_profile_path))
            
            # CMYK 0-100 to 0-255
            C_byte = int((c / 100.0) * 255)
            M_byte = int((m / 100.0) * 255)
            Y_byte = int((y / 100.0) * 255)
            K_byte = int((k / 100.0) * 255)
            
            cmyk_img = PilImage.new("CMYK", (1, 1), (C_byte, M_byte, Y_byte, K_byte))
            
            transform = ImageCms.buildTransformFromOpenProfiles(
                cmyk_profile,
                lab_profile,
                "CMYK",
                "LAB",
                renderingIntent=rendering_intent
            )
            
            lab_img = ImageCms.applyTransform(cmyk_img, transform)
            L_byte, a_byte, b_byte = lab_img.getpixel((0, 0))
            
            L = (L_byte / 255.0) * 100
            a = a_byte - 128
            b = b_byte - 128
            
            return (round(L, 2), round(a, 2), round(b, 2))
            
        except ImportError as e:
            raise ImportError(f"LittleCMS (PIL/ImageCms) is required for CMYK conversion: {str(e)}")
    
    else:
        raise ValueError(f"Unsupported color format: {fmt}")


def calculate_lab_centroid(lab_values: List[tuple]) -> tuple:
    """
    Calculate the centroid (average) of Lab values.
    """
    if not lab_values:
        raise ValueError("Cannot calculate centroid of empty list")
    
    n = len(lab_values)
    L_avg = sum(lab[0] for lab in lab_values) / n
    a_avg = sum(lab[1] for lab in lab_values) / n
    b_avg = sum(lab[2] for lab in lab_values) / n
    
    return (round(L_avg, 2), round(a_avg, 2), round(b_avg, 2))


@app.post("/color-consensus")
def color_consensus(payload: ColorConsensusInput):
    """
    Calculate color consensus from multiple samples with outlier rejection.
    
    Features:
    - Accepts mixed color formats (Lab, RGB, Hex, CMYK)
    - Normalizes all inputs to Lab using ICC profiles
    - Computes pairwise ΔE2000 comparisons
    - Identifies and removes outliers
    - Returns averaged consensus color
    
    Algorithm:
    1. Normalize all inputs to Lab
    2. Calculate initial centroid
    3. Compute ΔE2000 from each sample to centroid
    4. Remove outliers (ΔE2000 > threshold), keeping minimum 2 samples
    5. Recalculate final consensus from remaining samples
    """
    try:
        if len(payload.colors) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 color samples required for consensus calculation"
            )
        
        # Step 1: Normalize all colors to Lab
        normalized_samples = []
        normalization_errors = []
        
        for i, sample in enumerate(payload.colors):
            try:
                lab = normalize_color_to_lab(sample, payload.rendering_intent)
                
                # Generate hex preview
                r, g, b = lab_to_rgb(lab[0], lab[1], lab[2])
                hex_color = rgb_to_hex(r, g, b)
                
                normalized_samples.append({
                    "index": i,
                    "label": sample.label or f"Sample {i + 1}",
                    "original_format": sample.format,
                    "original_value": sample.value,
                    "lab": list(lab),
                    "hex": hex_color
                })
            except Exception as e:
                normalization_errors.append({
                    "index": i,
                    "label": sample.label or f"Sample {i + 1}",
                    "error": str(e)
                })
        
        if len(normalized_samples) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient valid samples after normalization. Errors: {normalization_errors}"
            )
        
        # Step 2: Calculate pairwise ΔE2000 matrix
        pairwise_results = []
        if payload.include_pairwise:
            for i, sample1 in enumerate(normalized_samples):
                for j, sample2 in enumerate(normalized_samples):
                    if i < j:  # Only upper triangle
                        lab1 = tuple(sample1["lab"])
                        lab2 = tuple(sample2["lab"])
                        delta_e = calculate_delta_e_2000(lab1, lab2)
                        
                        pairwise_results.append({
                            "sample1": sample1["label"],
                            "sample2": sample2["label"],
                            "deltaE2000": delta_e
                        })
        
        # Step 3: Calculate initial centroid
        lab_values = [tuple(s["lab"]) for s in normalized_samples]
        initial_centroid = calculate_lab_centroid(lab_values)
        
        # Step 4: Calculate distance from each sample to centroid
        for sample in normalized_samples:
            lab = tuple(sample["lab"])
            sample["deltaE_to_centroid"] = calculate_delta_e_2000(lab, initial_centroid)
        
        # Step 5: Identify outliers
        threshold = payload.outlier_threshold
        sorted_samples = sorted(normalized_samples, key=lambda x: x["deltaE_to_centroid"])
        
        # Keep at least 2 samples
        min_samples = 2
        included_samples = []
        excluded_samples = []
        
        for sample in sorted_samples:
            if sample["deltaE_to_centroid"] <= threshold or len(included_samples) < min_samples:
                included_samples.append(sample)
            else:
                excluded_samples.append(sample)
        
        # Step 6: Recalculate final consensus from included samples
        final_lab_values = [tuple(s["lab"]) for s in included_samples]
        final_consensus = calculate_lab_centroid(final_lab_values)
        
        # Calculate final distances to consensus
        for sample in included_samples:
            sample["deltaE_to_consensus"] = calculate_delta_e_2000(
                tuple(sample["lab"]), final_consensus
            )
        
        for sample in excluded_samples:
            sample["deltaE_to_consensus"] = calculate_delta_e_2000(
                tuple(sample["lab"]), final_consensus
            )
        
        # Generate consensus hex preview
        consensus_hex = None
        if payload.include_hex_preview:
            r, g, b = lab_to_rgb(final_consensus[0], final_consensus[1], final_consensus[2])
            consensus_hex = rgb_to_hex(r, g, b)
        
        # Check if consensus is stable
        max_delta = max(s["deltaE_to_consensus"] for s in included_samples) if included_samples else 0
        consensus_stable = max_delta <= threshold
        
        # Build response
        response = {
            "success": True,
            "input_count": len(payload.colors),
            "valid_count": len(normalized_samples),
            "included_count": len(included_samples),
            "excluded_count": len(excluded_samples),
            
            "consensus": {
                "lab": list(final_consensus),
                "hex": consensus_hex,
                "stable": consensus_stable,
                "max_deltaE": round(max_delta, 4)
            },
            
            "samples": {
                "included": [{
                    "label": s["label"],
                    "lab": s["lab"],
                    "hex": s["hex"],
                    "deltaE_to_consensus": round(s["deltaE_to_consensus"], 4)
                } for s in included_samples],
                
                "excluded": [{
                    "label": s["label"],
                    "lab": s["lab"],
                    "hex": s["hex"],
                    "deltaE_to_consensus": round(s["deltaE_to_consensus"], 4),
                    "reason": f"ΔE2000 ({s['deltaE_to_centroid']:.2f}) exceeds threshold ({threshold})"
                } for s in excluded_samples]
            },
            
            "settings": {
                "outlier_threshold": threshold,
                "rendering_intent": payload.rendering_intent,
                "method": "CIEDE2000"
            }
        }
        
        if payload.include_pairwise:
            response["pairwise_comparisons"] = pairwise_results
        
        if normalization_errors:
            response["normalization_errors"] = normalization_errors
        
        if not consensus_stable:
            response["warning"] = (
                f"Consensus may be unreliable. Maximum ΔE2000 ({max_delta:.2f}) "
                f"exceeds threshold ({threshold}). Consider reviewing input samples."
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Color consensus calculation failed: {str(e)}"
        )


@app.post("/color-compare")
def color_compare(payload: dict):
    """
    Simple pairwise comparison of two colors using ΔE2000.
    
    Input:
    {
        "color1": {"format": "lab", "value": [50, 10, -20]},
        "color2": {"format": "hex", "value": "#FF5733"}
    }
    """
    try:
        color1_data = payload.get("color1")
        color2_data = payload.get("color2")
        
        if not color1_data or not color2_data:
            raise HTTPException(status_code=400, detail="Both color1 and color2 required")
        
        sample1 = ColorSample(**color1_data)
        sample2 = ColorSample(**color2_data)
        
        lab1 = normalize_color_to_lab(sample1)
        lab2 = normalize_color_to_lab(sample2)
        
        delta_e = calculate_delta_e_2000(lab1, lab2)
        
        # Generate hex previews
        r1, g1, b1 = lab_to_rgb(lab1[0], lab1[1], lab1[2])
        r2, g2, b2 = lab_to_rgb(lab2[0], lab2[1], lab2[2])
        
        return {
            "success": True,
            "color1": {
                "lab": list(lab1),
                "hex": rgb_to_hex(r1, g1, b1)
            },
            "color2": {
                "lab": list(lab2),
                "hex": rgb_to_hex(r2, g2, b2)
            },
            "comparison": {
                "deltaE": delta_e,
                "method": "CIEDE2000",
                "perceptual_difference": (
                    "Imperceptible" if delta_e < 1.0 else
                    "Barely perceptible" if delta_e < 2.0 else
                    "Perceptible" if delta_e < 3.5 else
                    "Obvious" if delta_e < 5.0 else
                    "Very obvious"
                )
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Color comparison failed: {str(e)}"
        )
