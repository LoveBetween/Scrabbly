# https://www.geeksforgeeks.org/python/line-detection-python-opencv-houghline-method/
# https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python
# Python program to illustrate HoughLine
# method for line detection
import cv2
import numpy as np

def filter_similar_lines(lines_segments, cos_sim_thresh=0.98, disp_thresh=20):
    lines_segments = np.array(lines_segments, dtype=np.float32)

    directions = lines_segments[:, 2:4] - lines_segments[:, 0:2]
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_directions = directions / norms

    cos_sim_matrix = unit_directions @ unit_directions.T
    midpoints = (lines_segments[:, 0:2] + lines_segments[:, 2:4]) / 2
    diffs = midpoints[:, np.newaxis, :] - midpoints[np.newaxis, :, :]
    displacements = np.linalg.norm(diffs, axis=2)

    N = len(lines_segments)
    keep_mask = np.ones(N, dtype=bool)
    for i in range(N):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, N):
            if keep_mask[j]:
                if cos_sim_matrix[i, j] > cos_sim_thresh and displacements[i, j] < disp_thresh:
                    keep_mask[j] = False

    filtered_lines = lines_segments[keep_mask]
    return filtered_lines

def similarity(lines_segments):
    lines_segments = np.array(lines_segments, dtype=np.float32)
    directions = lines_segments[:, 2:4] - lines_segments[:, 0:2]
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_directions = directions / norms
    cos_sim_matrix = unit_directions @ unit_directions.T
    midpoints = (lines_segments[:, 0:2] + lines_segments[:, 2:4])/2
    diffs = midpoints[:, np.newaxis, :] - midpoints[np.newaxis, :, :]
    displacements = np.linalg.norm(diffs, axis=2)

    print("\nDisplacement Matrix:")
    print(displacements)

img = cv2.imread("board2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
thresh = cv2.inRange(hsv, (0, 0, 0) , (179, 80, 360))
invert = 255 - thresh

cv2.imshow("detection de lignes", thresh)
cv2.waitKey(0)

edges = cv2.Canny(invert, 60, 160, apertureSize=5)
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

line_segments = []
for r_theta in lines:
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*r
    y0 = b*r
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    line_segments.append([x1, y1, x2, y2])
    print((x1, y1), (x2, y2))

#similarity(line_segments)
filtered_lines = filter_similar_lines(line_segments, cos_sim_thresh=0.98, disp_thresh=20)

for x1, y1, x2, y2 in filtered_lines.astype(int):
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite('linesDetected.jpg', img)
cv2.imshow("detection de lignes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()