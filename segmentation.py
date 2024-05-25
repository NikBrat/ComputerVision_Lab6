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


def bwareaopen(image, dim, conn=8):
    assert image.ndim == 2, None
    # Find all connected components
    num, labels, stats, centers = cv.connectedComponentsWithStats(image, connectivity=conn)
    # Check size of all connected components
    for i in range(num):
        if stats[i, cv.CC_STAT_AREA] < dim:
            image[labels == i] = 0
    return image


# loading an image
src_img = cv.imread('source_images/snooker.jpg')
# converting it to binary
gray_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
ret, i_bw = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# deleting tiny connected components
i_bw = bwareaopen(i_bw, 20, 8)
disk = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
i_bw = cv.morphologyEx(i_bw, cv.MORPH_CLOSE, disk)
# display_image('Binary_closed', i_bw, 1)

# finding foreground
i_fg = cv.distanceTransform(i_bw, cv.DIST_L2, 5)
ret1, i_fg = cv.threshold(i_fg, 0.6 * i_fg.max(), 255, 0)
i_fg = i_fg.astype(np.uint8)
ret2, markers = cv.connectedComponents(i_fg)
# display_image('Fg', i_fg, 1)


# Find background location
i_bg = np.zeros_like(i_bw)
markers_bg = markers.copy()
markers_bg = cv.watershed(src_img, markers_bg)
i_bg[markers_bg == -1] = 255
# display_image('Bg', i_bg, 1)

# Define undefined area
i_unk = cv.subtract(i_bg, i_fg)
# Define all markers
markers = markers + 1
markers[i_unk == 255] = 0
# Do watershed
# Prepare for visualization
markers = cv.watershed(src_img, markers)
# display_image('UND', i_unk, 1)
markers_jet = cv.applyColorMap((markers.astype(np.float32)*255/(ret2 + 1)).astype(np.uint8), cv.COLORMAP_JET)
display_image('Jet_fg_markers', markers_jet, 1)
src_img[markers == -1] = (0, 0, 255)
# display_image('result', src_img, 1)
