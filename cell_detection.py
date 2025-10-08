import cv2
import numpy as np
import math
import os

def extract_grid_cells(image, lines, save_dir=None, angle_tol=0.1, debug=False):
    """
    Extracts grid cells from an image using horizontal and vertical Hough lines.

    Parameters:
        image (np.ndarray): Input image.
        lines (list): List of lines in (rho, theta) format (from cv2.HoughLines).
        save_dir (str): Directory to save the cells. If None, doesn't save.
        angle_tol (float): Tolerance for detecting vertical/horizontal lines (in radians).
        debug (bool): If True, draws lines and intersections on a copy of the image.

    Returns:
        cells (list of np.ndarray): Cropped cell images.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def line_to_points(rho, theta):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        return (x1, y1), (x2, y2)

    def compute_intersection(line1, line2):
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]

        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if denom == 0:
            return None
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return int(round(px)), int(round(py))

    # Classify lines
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        rho, theta = line[0]
        if abs(theta - np.pi/2) < angle_tol:  # Horizontal (theta ~ 90°)
            horizontal_lines.append(line)
        elif abs(theta) < angle_tol or abs(theta - np.pi) < angle_tol:  # Vertical (theta ~ 0° or 180°)
            vertical_lines.append(line)

    if debug:
        debug_img = image.copy()
        for l in horizontal_lines + vertical_lines:
            pt1, pt2 = line_to_points(*l[0])
            cv2.line(debug_img, pt1, pt2, (0, 255, 0), 1)

    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        print("Not enough lines to form a grid.")
        return []

    # Sort lines by position (rho)
    horizontal_lines.sort(key=lambda l: l[0][0])
    vertical_lines.sort(key=lambda l: l[0][0])

    # Extract cells
    cells = []
    h, w = image.shape[:2]

    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            h_top = line_to_points(*horizontal_lines[i][0])
            h_bottom = line_to_points(*horizontal_lines[i+1][0])
            v_left = line_to_points(*vertical_lines[j][0])
            v_right = line_to_points(*vertical_lines[j+1][0])

            tl = compute_intersection(h_top, v_left)
            tr = compute_intersection(h_top, v_right)
            bl = compute_intersection(h_bottom, v_left)
            br = compute_intersection(h_bottom, v_right)

            if None in (tl, tr, bl, br):
                continue

            x_min = max(min(tl[0], bl[0]), 0)
            x_max = min(max(tr[0], br[0]), w)
            y_min = max(min(tl[1], tr[1]), 0)
            y_max = min(max(bl[1], br[1]), h)

            if x_max - x_min < 5 or y_max - y_min < 5:
                continue  # skip very small/broken cells

            cell = image[y_min:y_max, x_min:x_max].copy()
            cells.append(cell)

            if save_dir:
                filename = os.path.join(save_dir, f"cell_{i}_{j}.png")
                cv2.imwrite(filename, cell)

            if debug:
                cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)

    if debug:
        cv2.imshow("Grid Debug", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cells