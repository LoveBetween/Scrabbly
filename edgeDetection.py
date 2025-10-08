import cv2
import numpy as np
import math

def is_diagonal(angle_rad, tol=0.1):
    # Normalize the angle to [0, 2π)
    angle_rad = angle_rad % (2 * math.pi)

    multiple_pi_4 = angle_rad / (math.pi / 4)
    multiple_pi_2 = angle_rad / (math.pi / 2)

    is_multiple_pi_4 = math.isclose(multiple_pi_4, round(multiple_pi_4), abs_tol=tol)
    is_multiple_pi_2 = math.isclose(multiple_pi_2, round(multiple_pi_2), abs_tol=tol)

    return is_multiple_pi_4 and not is_multiple_pi_2

def find_edges(img, type_edge):
    filter = True
    cannyApertureSize = 7

    if type_edge == "board":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, (0, 0, 0) , (179, 90, 120))
        gray = thresh
    else:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cannyApertureSize = 3

    edges = cv2.Canny(gray,140,200,apertureSize = cannyApertureSize)
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations = 1)
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.erode(edges,kernel,iterations = 1)
    cv2.imwrite('canny.jpg',edges)
    cv2.imshow("canny", edges)

    cv2.waitKey(0)

    lines = cv2.HoughLines(edges,1,np.pi/180,140)

    if not lines.any():
        print('No lines were found')
        exit()

    if filter:
        rho_threshold = 15
        theta_threshold = 0.1

        # how many lines are similar to a given one
        similar_lines = {i : [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i,theta_i = lines[i][0]
                rho_j,theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        # ordering the INDECES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x : len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines)*[True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
                continue

            for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
                if not line_flags[indices[j]]: # and only if we have not disregarded them already
                    continue

                rho_i,theta_i = lines[indices[i]][0]
                rho_j,theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now



    print('number of Hough lines:', len(lines))

    filtered_lines = []

    if filter:
        for i in range(len(lines)): # filtering
            if line_flags[i]:
                rho, theta = lines[i][0]
                if not type_edge=="edges" or (not is_diagonal(theta, tol=0.1)):  # Only keep diagonals
                    filtered_lines.append(lines[i])

        print('Number of filtered lines:', len(filtered_lines))
    else:
        filtered_lines = lines

    for line in filtered_lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
     
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        

    cv2.imwrite('hough.jpg',img)
    cv2.imshow("img_contour", img)

    cv2.waitKey(0)
    return filtered_lines

file_path = "board2.jpg"
img = cv2.imread(file_path)
filtered_lines = find_edges(img, "edges")

from cell_detection import extract_grid_cells

cells = extract_grid_cells(img, filtered_lines)
print("number of cells :", len(cells))
from textExtraction import cell_to_letter

for cell in enumerate(cells):
    index = cell[0]
    cell_img = cell[1]

    #test if cell has some beige
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    beige_thresh = cv2.inRange(hsv, (14.5-5, 70-30, 217-10) , (14.5+5, 70+20, 217+20))
    black_thresh = cv2.inRange(hsv, (0, 0, 0) , (360, 255, 120))
    

    # kernel = np.ones((3,3),np.uint8)
    # black_thresh = cv2.dilate(black_thresh,kernel,iterations = 1)
    # kernel = np.ones((5,5),np.uint8)
    # black_thresh = cv2.erode(black_thresh,kernel,iterations = 1)

    black_thresh = cv2.bitwise_not(black_thresh)


    height, width = cell_img.shape[:2]
    ratio = np.count_nonzero(beige_thresh)/(height*width)

    if ratio > 0.1:
        letter = cell_to_letter(black_thresh)
        resized = cv2.resize(black_thresh, (height*5, width*5), interpolation = cv2.INTER_AREA)
        cv2.imshow(letter, resized)
        cv2.waitKey(0)
        print(index//15,index%15, letter)

cv2.destroyAllWindows()