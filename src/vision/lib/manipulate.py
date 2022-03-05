"""
File provides multiple image manipulation techniques which together hopefully do something useful

FIXME: many of these "functions" have pretty bad side effects.
"""
import numpy as np
import cv2 as cv


def cluster(in_image, k):
    # Clustering
    z = in_image.reshape((-1, 3))
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
    edges = cv.Canny(in_image, 100, 200)
    return edges


def contour(in_image):
    blank_image = np.zeros(in_image.shape)
    ret, thresh = cv.threshold(in_image, 127, 255, 0)
    thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(blank_image, contours, -1, (0, 255, 0), 3)

    return blank_image


def hough(in_image):
    lines = cv.HoughLinesP(in_image, 1, np.pi / 180, threshold=60, minLineLength=1, maxLineGap=20)

    lines = [] if lines is None else lines

    # Iterate over points
    for points in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines join the points
        # On the original image
        cv.line(in_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Maintain a simples lookup list for points
        # lines_list.append([(x1,y1),(x2,y2)])

    return in_image