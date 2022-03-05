#!/usr/bin/env python3.10
#  #!/usr/bin/env python3
from pathlib import Path
import cv2 as cv
import sys
import matplotlib.pyplot as plt
import numpy as np
from lib.util import display_image
from lib.manipulate import cluster, canny, contour, hough


def main():
    image_paths = [
        # "../image_set/train/NORMAL/IM-0115-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0117-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0119-0001.jpeg",
        "../image_set/train/NORMAL/IM-0122-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0125-0001.jpeg",
    ]

    for image_path in image_paths:
        # Read image
        image = cv.imread(image_path)
        display_image("Input image", image)

        clustered = cluster(image, 2)
        display_image("Clustering", clustered)

        cannyed = canny(clustered)
        display_image("Canny", cannyed)

        houghed = hough(cannyed)
        display_image("Hough", houghed)

        contoured = contour(clustered)
        display_image("Cluster Contour", contoured)

    # bug (in pycharm?) causes pyplot buffer to not flush for last plot
    plt.show()


if __name__ == "__main__":
    print(f"Running on Python version {sys.version}")

    main()
