import cv2 as cv
import numpy as np


def display_image(title: str, image):
    """
    display image

    :param title: displayed window title
    :param image: image to be displayed
    :return:
    """
    cv.imshow(title, image)
    cv.waitKey(0)


# loading grayscale image
src_img = cv.imread('source_images/Ian_Berry_A young_Russian soldier.Jpeg', 0)
assert src_img is not None, "File could not be read"
# display_image('Original image', src_img)

disk = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
disk2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
ret, bin_image = cv.threshold(src_img[0:735, 200:1080], 180, 255, cv.THRESH_BINARY)
erosion = cv.erode(cv.erode(bin_image, disk, iterations=8), disk2, iterations=4)

display_image('Erosion', erosion)
