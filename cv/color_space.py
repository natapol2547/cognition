import numpy as np

# Making all colors perceptually uniform
# Oklab color space: https://bottosson.github.io/posts/oklab/
# Copied from https://gist.github.com/earthbound19/e7fe15fdf8ca3ef814750a61bc75b5ce


def gamma_to_linear(c):
    """Convert sRGB gamma-corrected value to linear RGB."""
    return np.where(c >= 0.04045, np.power((c + 0.055) / 1.055, 2.4), c / 12.92)


def linear_to_gamma(c):
    """Convert linear RGB value to sRGB gamma-corrected."""
    return np.where(c >= 0.0031308, 1.055 * np.power(c, 1 / 2.4) - 0.055, 12.92 * c)


def rgb_to_oklab(rgb):
    """Convert RGB (0-255) to Oklab color space.
    
    Args:
        rgb: numpy array of shape (..., 3) with RGB values in range 0-255
    
    Returns:
        Oklab array of shape (..., 3) with L (0-1), a (~-0.5 to +0.5), b (~-0.5 to +0.5)
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    
    # Convert sRGB to linear RGB
    r = gamma_to_linear(rgb[..., 0] / 255)
    g = gamma_to_linear(rgb[..., 1] / 255)
    b = gamma_to_linear(rgb[..., 2] / 255)
    
    # Linear RGB to LMS
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    
    # Cube root
    l = np.cbrt(l)
    m = np.cbrt(m)
    s = np.cbrt(s)
    
    # LMS to Oklab
    L = l * +0.2104542553 + m * +0.7936177850 + s * -0.0040720468
    a = l * +1.9779984951 + m * -2.4285922050 + s * +0.4505937099
    b = l * +0.0259040371 + m * +0.7827717662 + s * -0.8086757660
    
    return np.stack([L, a, b], axis=-1)


def oklab_to_rgb(lab):
    """Convert Oklab color space to RGB (0-255).
    
    Args:
        lab: numpy array of shape (..., 3) with Oklab values (L, a, b)
    
    Returns:
        RGB array of shape (..., 3) with values in range 0-255
    """
    lab = np.asarray(lab, dtype=np.float64)
    
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    
    # Oklab to LMS
    l = L + a * +0.3963377774 + b * +0.2158037573
    m = L + a * -0.1055613458 + b * -0.0638541728
    s = L + a * -0.0894841775 + b * -1.2914855480
    
    # Cube
    l = l ** 3
    m = m ** 3
    s = s ** 3
    
    # LMS to linear RGB
    r = l * +4.0767416621 + m * -3.3077115913 + s * +0.2309699292
    g = l * -1.2684380046 + m * +2.6097574011 + s * -0.3413193965
    b = l * -0.0041960863 + m * -0.7034186147 + s * +1.7076147010
    
    # Linear RGB to sRGB
    r = 255 * linear_to_gamma(r)
    g = 255 * linear_to_gamma(g)
    b = 255 * linear_to_gamma(b)
    
    # Clamp and round
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 255)
    rgb = np.round(rgb).astype(np.uint8)
    
    return rgb

# From https://gist.github.com/dkaraush/65d19d61396f5f3cd8ba7d1b4b3c9432
def oklab_to_oklch(lab):
    """Convert Oklab (L, a, b) to Oklch (L, c, h_degrees)
    """
    lab = np.asarray(lab, dtype=np.float64)

    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    c = np.sqrt(a**2 + b**2)
    h = np.degrees(np.arctan2(b, a))
    h = np.mod(h, 360.0)

    neutral = (np.abs(a) < 0.0002) & (np.abs(b) < 0.0002)
    h = np.where(neutral, 0, h)

    return np.stack([L, c, h], axis=-1)


def oklch_to_oklab(lch):
    """Convert Oklch (L, c, h_degrees) to Oklab (L, a, b)
    """
    lch = np.asarray(lch, dtype=np.float64)

    L = lch[..., 0]
    c = lch[..., 1]
    h = lch[..., 2]

    h_rad = np.deg2rad(h)

    a = c * np.cos(h_rad)
    b = c * np.sin(h_rad)

    return np.stack([L, a, b], axis=-1)

def rgb_to_oklch(rgb):
    oklab = rgb_to_oklab(rgb)
    return oklab_to_oklch(oklab)

def oklch_to_rgb(lch):
    oklab = oklch_to_oklab(lch)
    return oklab_to_rgb(oklab)