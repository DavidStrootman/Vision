#!/usr/bin/env python3.10
#  #!/usr/bin/env python3
from pathlib import Path
import cv2 as cv
import sys
import numpy as np
from lib.util import display_image


def main():
    image_paths = [  # "../image_set/train/NORMAL/IM-0115-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0117-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0119-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0122-0001.jpeg",
        "../image_set/train/NORMAL/IM-0125-0001.jpeg",
    ]

    for image_path in image_paths:
        image = cv.imread(image_path)

        image = cv.imread(image_path)
        display_image(image)
        z = image.reshape((-1, 3))
        # convert to np.float32
        z = np.float32(z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 2
        ret, label, center = cv.kmeans(z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape(image.shape)
        display_image(res2)

        edges = cv.Canny(image, 100, 200)
        display_image(edges)


        blank_image = np.zeros(image.shape)
        ret, thresh = cv.threshold(image, 127, 255, 0)
        thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)

        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(blank_image, contours, -1, (0, 255, 0), 3)

        display_image(thresh)

        # lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=1, maxLineGap=20)
        #
        # lines = [] if lines is None else lines
        #
        # # Iterate over points
        # for points in lines:
        #     # Extracted points nested in the list
        #     x1, y1, x2, y2 = points[0]
        #     # Draw the lines join the points
        #     # On the original image
        #     cv.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     # Maintain a simples lookup list for points
        #     # lines_list.append([(x1,y1),(x2,y2)])
        #
        # display_image(img_copy)
        # image = cv.medianBlur(image, 7)
        # image = cv.medianBlur(image, 7)
        # image = cv.medianBlur(image, 7)
        # display_image(image)


if __name__ == "__main__":
    print(f"Running on Python version {sys.version}")

    main()
