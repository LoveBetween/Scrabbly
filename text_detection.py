import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from matplotlib import pyplot as plt

image_path = "board2.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]

image_used = thresh
image_with_boxes = image.copy()

custom_config = r'-l eng --psm 11 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ" '
extracted_text = pytesseract.image_to_string(image_used, config=custom_config)
print(" Extracted Text:\n")
print(extracted_text)

data = pytesseract.image_to_data(image_used, output_type=pytesseract.Output.DICT, config=custom_config)

n_boxes = len(data['level'])
for i in range(n_boxes):
    (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
    text = data['text'][i]
    if text.strip():  # Only draw if text is non-empty
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_with_boxes, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        print(i, text)

plt.figure(figsize=(10, 6))
plt.imshow(image_used, cmap="gray")
#plt.imshow(image_with_boxes)
plt.title("Image with Text Bounding Boxes")
plt.axis("off")
plt.show()