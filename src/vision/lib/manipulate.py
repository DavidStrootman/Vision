"""
File provides multiple image manipulation techniques which together hopefully do something useful

FIXME: many of these "functions" have pretty bad side effects.
"""
import numpy as np
import cv2 as cv


def adjust_gamma(image, gamma=1.0):
    # Taken from pyimagesearch.com: opencv-gamma-correction (2015-10-05)
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(image, table)


def dilate(in_image, kernel, iterations: int):
    kernel = np.ones(kernel, np.uint8)
    return cv.dilate(in_image, kernel, iterations=iterations)


def cluster(in_image, k):
    # Clustering
    z = in_image.reshape((-1, 1))
    # convert to np.float32
    z = np.float32(z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(in_image.shape)
    return res2


def canny(in_image):
    edges = cv.Canny(in_image, 50, 150)
    return edges


def contour(in_image, new_image: bool, out_image=None):
    image_copy = np.zeros(in_image.shape) if new_image else in_image.copy()
    ret, thresh = cv.threshold(in_image, 127, 255, 0)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    output = out_image if out_image is not None else image_copy
    cv.drawContours(output, contours, -1, (0, 255, 0), thickness=1)

    return output, contours


def hough(in_image, threshold: int, min_line_length: int, max_line_gap: int):
    lines = cv.HoughLinesP(image=in_image,
                           rho=1,
                           theta=np.pi / 180,
                           threshold=threshold,
                           minLineLength=min_line_length,
                           maxLineGap=max_line_gap)

    lines = [] if lines is None else lines

    blank_image = np.zeros(in_image.shape)

    # Iterate over points
    for points in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines join the points
        # On the original image
        cv.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Maintain a simples lookup list for points
        # lines_list.append([(x1,y1),(x2,y2)])

    return blank_image
