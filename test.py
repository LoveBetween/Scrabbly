import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

def sobel(src_image, kernel_size):
    grad_x = cv.Sobel(src_image, cv.CV_16S, 1, 0, ksize=kernel_size, scale=1,
                      delta=0, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(src_image, cv.CV_16S, 0, 1, ksize=kernel_size, scale=1, 
                      delta=0, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad

def process_image(src_image_path):
    # load the image
    src_image = cv.imread(src_image_path)

    src_gray = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)

    # standard technique to eliminate noise
    blur_image = cv.blur(src_gray,(3,3))

    # strengthen the appearance of lines in the image
    sobel_image = sobel(blur_image, 3)

    # detect corners
    corners = cv.cornerHarris(sobel_image, 17, 21, 0.01)
    # for visualization to make corners easier to see
    corners = cv.dilate(corners, None)

    # overlay on a copy of the source image
    dest_image = np.copy(src_image)
    dest_image[corners>0.1*corners.max()]=[0,0,255]
    return dest_image 

src_image_path = "board2.jpg"
print(src_image_path)
dest_image = process_image(src_image_path)
plt.imshow(dest_image)
plt.show()