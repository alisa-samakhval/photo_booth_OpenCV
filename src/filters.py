import cv2
import numpy as np

def to_grayscale(frame):
    """
    Convert a color frame to grayscale.
    
    Args:
        frame: Color image as numpy array (BGR format)
        
    Returns:
        Grayscale image as numpy array
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def to_negative(frame):
    """
    Convert image to its negative by inverting all colors.
    
    Args:
        frame: Color image as numpy array (BGR format)
        
    Returns:
        Negative image as numpy array
    """
    return 255 - frame

def apply_lut(frame, lut):
    """
    Apply a lookup table to adjust image properties.
    
    Args:
        frame: Color image as numpy array (BGR format)
        lut: Lookup table array of shape (256,) for single channel
             or (256, 1, 3) for color image
             
    Returns:
        Adjusted image as numpy array
    """
    # Make sure LUT is properly shaped for color images
    if len(frame.shape) == 3 and len(lut.shape) == 1:
        lut = np.dstack([lut] * 3)
    
    # Apply the lookup table
    return cv2.LUT(frame, lut)

def build_sample_lut():
    """
    Build a sample lookup table for brightness/contrast adjustment.
    This creates an S-curve that increases contrast in the mid-tones
    while preserving highlights and shadows.
    
    Returns:
        Lookup table as numpy array
    """
    # Create base array (0-255)
    x = np.arange(256, dtype=np.float32)
    
    # Apply sigmoid function for contrast
    alpha = 2.0  # Controls contrast
    beta = 128.0  # Controls brightness
    lut = 255 / (1 + np.exp(-alpha * (x - beta) / 255)) 
    
    # Convert to proper format for cv2.LUT
    return np.clip(lut, 0, 255).astype(np.uint8)

def load_stickers(paths):
    """
    Load stickers from a list of file paths. Returns a list of loaded images (or None if failed).
    """
    stickers = []
    for i, path in enumerate(paths):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Sticker {i+1} at '{path}' failed to load!")
        else:
            print(f"Sticker {i+1} loaded from '{path}': shape {img.shape}")
        stickers.append(img)
    return stickers 