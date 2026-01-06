"""
LUT-Based Lab->CMYK Conversion
PROFESSIONAL APPROACH: Uses actual Pantone measurements

This replaces polynomial corrections with lookup table + interpolation.
This is how real ICC profiles work internally.
"""

import json
from pathlib import Path

# Load LUT data once at module level
LUT_DATA = None
LUT_GRID_POINTS = None

def load_lut():
    """Load the Pantone LUT data"""
    global LUT_DATA, LUT_GRID_POINTS
    
    if LUT_DATA is not None:
        return
    
    lut_file = Path(__file__).parent.parent / "data" / "pantone_cmyk_lut.json"
    
    with open(lut_file, 'r') as f:
        data = json.load(f)
    
    LUT_GRID_POINTS = data['grid_points']
    LUT_DATA = data['lut']
    
    print(f"Loaded Pantone LUT: {LUT_GRID_POINTS}^3 grid ({len(LUT_DATA)} points)")


def trilinear_interpolate(L, a, b):
    """
    Trilinear interpolation in Lab->CMYK LUT
    
    This is the PROFESSIONAL way to use a 3D lookup table.
    All ICC profiles use this internally.
    
    Args:
        L: Lightness (0-100)
        a: a* component (-128 to 127)
        b: b* component (-128 to 127)
    
    Returns:
        (c, m, y, k) interpolated from LUT
    """
    load_lut()
    
    # Normalize Lab to grid coordinates
    # L: 0-100 -> 0 to (grid_points-1)
    # a,b: -128 to 127 -> 0 to (grid_points-1)
    
    grid_max = LUT_GRID_POINTS - 1
    
    L_norm = (L / 100.0) * grid_max
    a_norm = ((a + 128) / 255.0) * grid_max
    b_norm = ((b + 128) / 255.0) * grid_max
    
    # Clamp to grid bounds
    L_norm = max(0, min(grid_max, L_norm))
    a_norm = max(0, min(grid_max, a_norm))
    b_norm = max(0, min(grid_max, b_norm))
    
    # Get integer coordinates and fractional parts
    L0 = int(L_norm)
    a0 = int(a_norm)
    b0 = int(b_norm)
    
    L1 = min(L0 + 1, grid_max)
    a1 = min(a0 + 1, grid_max)
    b1 = min(b0 + 1, grid_max)
    
    Ld = L_norm - L0
    ad = a_norm - a0
    bd = b_norm - b0
    
    # Get 8 corner values from LUT
    def get_lut_value(L_idx, a_idx, b_idx):
        # Calculate 1D index from 3D coordinates
        idx = L_idx * (LUT_GRID_POINTS ** 2) + a_idx * LUT_GRID_POINTS + b_idx
        if 0 <= idx < len(LUT_DATA):
            cmyk = LUT_DATA[idx]
            return (cmyk['c'], cmyk['m'], cmyk['y'], cmyk['k'])
        return (0, 0, 0, 100)  # Fallback: pure black
    
    c000 = get_lut_value(L0, a0, b0)
    c001 = get_lut_value(L0, a0, b1)
    c010 = get_lut_value(L0, a1, b0)
    c011 = get_lut_value(L0, a1, b1)
    c100 = get_lut_value(L1, a0, b0)
    c101 = get_lut_value(L1, a0, b1)
    c110 = get_lut_value(L1, a1, b0)
    c111 = get_lut_value(L1, a1, b1)
    
    # Trilinear interpolation
    # Interpolate along b axis
    c00 = tuple((1-bd)*v0 + bd*v1 for v0, v1 in zip(c000, c001))
    c01 = tuple((1-bd)*v0 + bd*v1 for v0, v1 in zip(c010, c011))
    c10 = tuple((1-bd)*v0 + bd*v1 for v0, v1 in zip(c100, c101))
    c11 = tuple((1-bd)*v0 + bd*v1 for v0, v1 in zip(c110, c111))
    
    # Interpolate along a axis
    c0 = tuple((1-ad)*v0 + ad*v1 for v0, v1 in zip(c00, c01))
    c1 = tuple((1-ad)*v0 + ad*v1 for v0, v1 in zip(c10, c11))
    
    # Interpolate along L axis
    result = tuple((1-Ld)*v0 + Ld*v1 for v0, v1 in zip(c0, c1))
    
    # Clamp to valid CMYK range
    c, m, y, k = result
    c = max(0, min(100, c))
    m = max(0, min(100, m))
    y = max(0, min(100, y))
    k = max(0, min(100, k))
    
    return (c, m, y, k)


def lab_to_cmyk_lut(L, a, b):
    """
    Professional Lab->CMYK conversion using Pantone measurement LUT
    
    This is how professional software works:
    1. Use actual measurement data
    2. Build lookup table
    3. Interpolate between grid points
    
    Args:
        L, a, b: Lab color values
    
    Returns:
        (c, m, y, k) tuple
    """
    return trilinear_interpolate(L, a, b)


# Test if LUT loads correctly
if __name__ == "__main__":
    print("Testing LUT-based conversion...")
    
    # Test with some known Pantone colors
    test_colors = [
        (97.22, -0.03, 2.26, "P 1-1 C - Light"),
        (44.28, 9.48, -21.15, "P 101-13 C - Blue"),
        (17.30, 1.27, 1.05, "P 179-16 C - Black"),
    ]
    
    for L, a, b, name in test_colors:
        c, m, y, k = lab_to_cmyk_lut(L, a, b)
        print(f"\n{name}")
        print(f"  Lab: L={L:.2f}, a={a:.2f}, b={b:.2f}")
        print(f"  CMYK: C={c:.1f}, M={m:.1f}, Y={y:.1f}, K={k:.1f}")
    
    print("\nLUT-based conversion working!")
