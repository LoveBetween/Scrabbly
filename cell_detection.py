import cv2
import numpy as np
import math
import os

def extract_grid_cells(image, lines, save_dir=None, angle_tol=0.1, debug=True):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def order_corners(pts):
        """Given 4 points, return them in top-left, top-right, bottom-right, bottom-left order."""
        pts = np.array(pts, dtype=np.float32)
        s = pts.sum(axis=1)            # tl has smallest sum, br has largest
        diff = np.diff(pts, axis=1)    # tr has smallest diff, bl has largest

        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]       # Top-left
        ordered[2] = pts[np.argmax(s)]       # Bottom-right
        ordered[1] = pts[np.argmin(diff)]    # Top-right
        ordered[3] = pts[np.argmax(diff)]    # Bottom-left

        return ordered

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
    print(f"Horizontal: {len(horizontal_lines)} | Vertical: {len(vertical_lines)}")
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
    normalized_size = (50, 50)  # Width x Height of output cell

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
                print(f"Skipping cell ({i}, {j}) due to None corner.")
                continue

            # Convert to float32 for perspective transform
            src_pts = order_corners([tl, tr, br, bl])
            dst_pts = np.array([
                [0, 0],
                [normalized_size[0] - 1, 0],
                [normalized_size[0] - 1, normalized_size[1] - 1],
                [0, normalized_size[1] - 1]
            ], dtype=np.float32)

            try:
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(image, M, normalized_size)
                cells.append(warped)

                if save_dir:
                    filename = os.path.join(save_dir, f"cell_{i}_{j}.png")
                    cv2.imwrite(filename, warped)

                if debug:
                    for pt in [tl, tr, br, bl]:
                        cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
                    cv2.polylines(debug_img, [np.array([tl, tr, br, bl], dtype=np.int32)], True, (0, 0, 255), 1)

            except cv2.error as e:
                print(f"OpenCV error for cell ({i}, {j}): {e}")
                continue

    if debug:
        cv2.imshow("Grid Debug", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cells