"""
File provides multiple image manipulation techniques which together hopefully do something useful

FIXME: many of these "functions" have pretty bad side effects.
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from lib.network import images_with_labels, XraySequence
from lib.util import display_image


def full_manipulation(images: XraySequence, plot_images: bool = False) -> XraySequence:
    manipulated_images = []

    for image, _ in images:
        # ----- Read image

        # ----- convert to grayscale
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # ----- Equalize histogram
        equalized = cv.equalizeHist(image)

        # ----- Increase gamma
        gamma = adjust_gamma(equalized, 5.0)

        # ----- Threshold
        _, thresholded = cv.threshold(gamma, 208, 255, cv.THRESH_BINARY)

        # ----- Remove photo artifacts from some images:
        erode_kernel = np.ones((2, 2), np.uint8)
        eroded = cv.erode(src=thresholded, kernel=erode_kernel)
        # ----- Strip small components in the mask (smaller than 1/10 of the image)
        min_size = eroded.shape[0] * eroded.shape[1] // 10
        n_components, labels, stats, _ = cv.connectedComponentsWithStats(eroded, connectivity=8)
        sizes = stats[1:, -1]

        filtered_components = np.zeros(eroded.shape)
        for c in range(0, n_components - 1):
            if sizes[c] >= min_size:
                filtered_components[labels == c + 1] = 255

        # ----- Dilate to connect "small" gaps
        blur_kernel = (11, 11)
        dilated = cv.GaussianBlur(filtered_components, blur_kernel, 2, 2)
        dilate_kernel = (10, 10)
        dilated = dilate(dilated, dilate_kernel, 5)

        # ----- Flood sides
        flood_mask = np.zeros(dilated.shape, dtype=np.uint8)
        flood_mask = cv.copyMakeBorder(flood_mask, 1, 1, 1, 1, borderType=cv.BORDER_CONSTANT)
        # ----- Flood left side
        flood_input = dilated.astype('uint8')
        cv.floodFill(image=flood_input,
                     mask=flood_mask,
                     seedPoint=(1, flood_input.shape[0] // 2),
                     newVal=(255, 0, 0),
                     flags=cv.FLOODFILL_MASK_ONLY
                     )

        # ----- Flood right side
        cv.floodFill(image=flood_input,
                     mask=flood_mask,
                     seedPoint=(flood_input.shape[1] - 1, flood_input.shape[0] // 2),
                     newVal=(255, 0, 0),
                     flags=cv.FLOODFILL_MASK_ONLY
                     )

        # Remove the 1 pixel border required for masking in flood from all sides
        flood_mask = flood_mask[1:flood_mask.shape[0] - 1, 1: flood_mask.shape[1] - 1]
        _, flood_mask = cv.threshold(flood_mask, 0, 255, cv.THRESH_BINARY_INV)

        output_image = cv.bitwise_and(image, image, mask=flood_mask)
        manipulated_images.append(cv.resize(output_image, (1000, 1000), interpolation=cv.INTER_LINEAR))
        if plot_images is True:
            plt.axis("off")
            plt.imshow(output_image)
            plt.show()

    return XraySequence(manipulated_images, images.y_set, images.batch_size)


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
