import cv2
import easyocr
import os
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

def crop_borders(img, border_size):
    """Crop N pixels from all sides of the image."""
    h, w = img.shape[:2]
    return img[border_size:h - border_size, border_size:w - border_size]

def cell_to_letter_easyocr(hsv):
    # Use V channel from HSV (brightness)
    gray = hsv[:, :, 2]
    
    # Threshold to binary (invert to make letter white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    invert = 255 - thresh

    # EasyOCR expects grayscale or 3-channel images

    letters = []
    border_sizes = [0, 4, 8, 16]  # Pixels to crop from each side

    for border in border_sizes:
        cropped = crop_borders(invert, border)
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            continue  # Skip too-small crops

        results = reader.readtext(
            cropped,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            detail=1,
            paragraph=False,
            min_size=1
        )

        if results:
            letter = results[0][1]
            letters.append((letter, results[0][2]))  # (text, confidence)

    if letters:
        # Return highest-confidence letter
        best = max(letters, key=lambda x: x[1])
        return best[0], invert  # best letter, and processed image
    else:
        return '', invert
    
def detect_circle_as_O(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter**2))
        if 0.75 < circularity < 1.2:
            return True
    return False

def cell_to_letter_easyocr_2(cell_img):
    # Convert to grayscale
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Dilate to strengthen thin letters
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    letters = []
    border_sizes = [0, 2, 4, 6]  # pixels to crop from each side

    for border in border_sizes:
        cropped = crop_borders(dilated, border)
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            continue

        # Agrandissement pour OCR
        scale = 3
        resized = cv2.resize(cropped, (cropped.shape[1]*scale, cropped.shape[0]*scale),
                             interpolation=cv2.INTER_CUBIC)

        results = reader.readtext(
            resized,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZIOF',
            detail=1,
            paragraph=False,
            min_size=1
        )

        if results:
            letter = results[0][1].upper()
            conf = results[0][2]
            letters.append((letter, conf))

    if letters:
        best = max(letters, key=lambda x: x[1])
        letter_final = best[0]

        # Vérification circulaire pour corriger les O mal détectés
        if letter_final not in ['O', 'I', 'F'] and detect_circle_as_O(dilated):
            letter_final = 'O'

        return letter_final, dilated
    else:
        return '', dilated