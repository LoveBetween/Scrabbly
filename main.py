import cv2
import numpy as np
import math
import sys
import io

from cell_detection import extract_grid_cells
from textExtraction import cell_to_letter_easyocr3 as cell_to_letter
from edgeDetection import find_edges

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# https://github.com/andreanlay/handwritten-character-recognition-deep-learning
# https://www.geeksforgeeks.org/python/line-detection-python-opencv-houghline-method/
# https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python

file_path = "board2.jpg"
img = cv2.imread(file_path)
filtered_lines, debug_img = find_edges(img, "edges")

cells = extract_grid_cells(img, filtered_lines)
print("number of cells :", len(cells))

for cell in enumerate(cells):
    index = cell[0]
    cell_img = cell[1]

    #test if cell has some beige
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    beige_thresh = cv2.inRange(hsv, (14.5-5, 70-30, 217-40) , (14.5+5, 70+20, 217+30))
    # kernel = np.ones((3,3),np.uint8)
    # black_thresh = cv2.dilate(black_thresh,kernel,iterations = 1)
    # kernel = np.ones((5,5),np.uint8)
    # black_thresh = cv2.erode(black_thresh,kernel,iterations = 1)
    height, width = cell_img.shape[:2]
    ratio = np.count_nonzero(beige_thresh)/(height*width)
    
    if ratio > 0.15:
        letter, image_edited = cell_to_letter(hsv)
        # match_letter(hsv, "O")
        resized = cv2.resize(cell_img, (height*5, width*5), interpolation = cv2.INTER_AREA)
        cv2.imshow(f"{letter} {ratio:.2f}", resized)
        cv2.waitKey(0)
        print(f"ratio {ratio:.2f}, col {index//15 + 1}, line {index%15 + 1}, letter", index,  letter)

cv2.destroyAllWindows()