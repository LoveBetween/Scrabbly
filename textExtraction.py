import cv2
import pytesseract
import easyocr
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
reader = easyocr.Reader(['en'], gpu=False)

# def cell_to_letter(img):
#     #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = img
#     blur = cv2.GaussianBlur(gray, (3,3), 0)
#     thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
#     invert = 255 - opening

#     # Perform text extraction
#     cfg = r'-l eng --oem 3 --psm 10 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ" '
#     data = pytesseract.image_to_string(invert, config = cfg)
#     return data

def cell_to_letter_easyocr(hsv):
    #    black_thresh = cv2.bitwise_not(black_thresh)
    #    black_thresh = cv2.inRange(hsv, (0, 0, 0) , (360, 255, 120))
    #gray = cv2.cvtColor(hsv, cv2.COLOR_HSV2GRAY)  # assuming img is already grayscale
    gray = hsv[:, :, 2]
    #blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(gray, cv2.THRESH_OTSU, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - thresh

    # EasyOCR expects 3-channel or grayscale images as numpy arrays (uint8)
    # If grayscale, it's fine as is.
    
    letters = []
    for contrast in [0.5, 0.7, 0.9]:
        result = reader.readtext(
            invert,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            detail=1,
            paragraph=False,
            min_size=1,
            adjust_contrast=contrast
        )
        if result:
            letters.append(result[0][1])
    if letters:
        # Since psm 10 is single character, return first detected letter
        return letters, invert
    else:
        return '', invert

def adjust_contrast_brightness(img, alpha=1.0, beta=0):
    """
    alpha: contrast control (1.0 = original)
    beta: brightness control (0 = original)
    """
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return new_img

def cell_to_letter_easyocr2(hsv):
    gray = hsv[:, :, 2]
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    invert = 255 - thresh

    letters = []
    for alpha in [0.9, 1.2, 1.5]:  # Adjust contrast
        contrast_img = adjust_contrast_brightness(invert, alpha=alpha)
        
        result = reader.readtext(
            contrast_img,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            detail=1,
            paragraph=False,
            min_size=1
        )
        
        if result:
            letters.append(result[0][1])  # append the letter

    if letters:
        return letters, invert
    else:
        return '', invert

def crop_borders(img, border_size):
    """Crop N pixels from all sides of the image."""
    h, w = img.shape[:2]
    return img[border_size:h - border_size, border_size:w - border_size]

def cell_to_letter_easyocr3(hsv):
    # Use V channel from HSV (brightness)
    gray = hsv[:, :, 2]
    
    # Threshold to binary (invert to make letter white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    invert = 255 - thresh

    # EasyOCR expects grayscale or 3-channel images

    letters = []
    border_sizes = [0, 2, 4, 6]  # Pixels to crop from each side

    for border in border_sizes:
        cropped = crop_borders(invert, border)
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            continue  # Skip too-small crops

        results = reader.readtext(
            cropped,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZl',
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
        print(letters)
        return best[0], invert  # best letter, and processed image
    else:
        return '', invert