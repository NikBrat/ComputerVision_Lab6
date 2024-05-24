import cv2 as cv
import numpy as np


def display_image(title: str, image, option: int = 0):
    """
    display and save image

    :param title: displayed window title
    :param image: image to be displayed
    :param option: switch to save image (0 - off, 1 - on)
    :return:
    """
    cv.imshow(title, image)
    cv.waitKey(0)
    if option:
        cv.imwrite(f'transformed_images/{title}.jpg', image)


# loading grayscale image
src_img = cv.imread('source_images/Ian_Berry_A young_Russian soldier.Jpeg', 0)
assert src_img is not None, "File could not be read"
# display_image('Original image', src_img)
result = np.copy(src_img)
# selecting specific area
spec_area = src_img[0:735, 200:1080]
# morphological operations
disk = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
erosion = cv.erode(spec_area, disk, iterations=1)
dilation = cv.dilate(erosion, disk, iterations=7)
erosion2 = cv.erode(dilation, disk, iterations=5)
# displaying transformed image
result[0:735, 200:1080] = erosion2
display_image('Result', result, 1)


