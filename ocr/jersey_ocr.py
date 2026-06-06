import cv2
import numpy as np

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=True)
    return _reader


def crop_jersey_region(frame, bbox):
    """Crop upper-body region where jersey number typically appears."""
    x1, y1, x2, y2 = map(int, bbox)
    h = y2 - y1
    w = x2 - x1
    # Number is usually in upper 40-60% of bbox
    crop_top = y1 + int(h * 0.05)
    crop_bottom = y1 + int(h * 0.55)
    crop_left = x1 + int(w * 0.1)
    crop_right = x2 - int(w * 0.1)
    if crop_bottom <= crop_top or crop_right <= crop_left:
        return None
    crop = frame[crop_top:crop_bottom, crop_left:crop_right]
    return crop


def preprocess_crop(crop):
    """Preprocess for OCR: grayscale, enhance contrast, threshold."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Adaptive threshold to isolate text
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8)
    # Denoise
    denoised = cv2.medianBlur(binary, 3)
    return denoised


def read_jersey_number(frame, bbox, cache=None, track_id=None):
    """
    Read jersey number from player crop.

    Args:
        frame: video frame
        bbox: [x1, y1, x2, y2] player bounding box
        cache: dict mapping track_id -> number (optional)
        track_id: player track ID

    Returns:
        int or None: detected jersey number
    """
    if cache is not None and track_id is not None and track_id in cache:
        return cache[track_id]

    crop = crop_jersey_region(frame, bbox)
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 10:
        return None

    processed = preprocess_crop(crop)
    reader = _get_reader()
    results = reader.readtext(processed, allowlist='0123456789')

    for (_, text), conf in [(r[1], r[2]) for r in results]:
        text = text.strip()
        if text and conf > 0.3:
            try:
                num = int(text)
                if 1 <= num <= 99:
                    if cache is not None and track_id is not None:
                        cache[track_id] = num
                    return num
            except ValueError:
                # Try multi-digit parsing
                digits = ''.join(c for c in text if c.isdigit())
                if digits and len(digits) <= 2:
                    num = int(digits)
                    if 1 <= num <= 99:
                        if cache is not None and track_id is not None:
                            cache[track_id] = num
                        return num
    return None


def draw_jersey_number(frame, bbox, number, track_id=None):
    """Draw jersey number above the player."""
    x1, y1, x2, y2 = map(int, bbox)
    cx = (x1 + x2) // 2
    label = str(number)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2, 3)
    cv2.rectangle(frame, (cx - tw // 2 - 4, y1 - th - 8),
                  (cx + tw // 2 + 4, y1 - 2),
                  (0, 0, 0), -1)
    cv2.putText(frame, label,
                (cx - tw // 2, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 3)
