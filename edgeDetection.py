import cv2
import numpy as np
import math

def is_diagonal(angle_rad, tol=0.1):
    angle_rad = angle_rad % (2 * math.pi)

    multiple_pi_4 = angle_rad / (math.pi / 4)
    multiple_pi_2 = angle_rad / (math.pi / 2)

    is_multiple_pi_4 = math.isclose(multiple_pi_4, round(multiple_pi_4), abs_tol=tol)
    is_multiple_pi_2 = math.isclose(multiple_pi_2, round(multiple_pi_2), abs_tol=tol)

    return is_multiple_pi_4 and not is_multiple_pi_2
    cv2.waitKey(0)
    return filtered_lines


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

        similar_lines = {i : [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i,theta_i = lines[i][0]
                rho_j,theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x : len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines)*[True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[i]]:
                continue

            for j in range(i + 1, len(lines)):
                if not line_flags[indices[j]]:
                    continue

                rho_i,theta_i = lines[indices[i]][0]
                rho_j,theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[indices[j]] = False


    print('number of Hough lines:', len(lines))

    filtered_lines = []

    if filter:
        for i in range(len(lines)):
            if line_flags[i]:
                rho, theta = lines[i][0]
                if not type_edge=="edges" or (not is_diagonal(theta, tol=0.1)):
                    filtered_lines.append(lines[i])

        print('Number of filtered lines:', len(filtered_lines))
    else:
        filtered_lines = lines
    img_copy = img.copy()
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
     
        cv2.line(img_copy,(x1,y1),(x2,y2),(0,0,255),2)
    #cv2.imwrite('hough.jpg',img)
    #cv2.imshow("img_contour", img)

    return filtered_lines, img_copy