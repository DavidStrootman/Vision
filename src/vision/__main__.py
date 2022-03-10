#!/usr/bin/env python3.10
#  #!/usr/bin/env python3
from pathlib import Path
import cv2 as cv
import sys
import matplotlib.pyplot as plt
import numpy as np
from lib.util import display_image
from lib.manipulate import cluster, canny, contour, hough, dilate


def main():
    image_paths = [
        "../image_set/train/NORMAL/IM-0115-0001.jpeg",
        "../image_set/train/NORMAL/IM-0117-0001.jpeg",
        "../image_set/train/NORMAL/IM-0119-0001.jpeg",
        "../image_set/train/NORMAL/IM-0122-0001.jpeg",
        "../image_set/train/NORMAL/IM-0125-0001.jpeg",
        "../image_set/train/NORMAL/IM-0128-0001.jpeg",
        "../image_set/train/NORMAL/IM-0129-0001.jpeg",
        "../image_set/train/NORMAL/IM-0131-0001.jpeg",
        "../image_set/train/NORMAL/IM-0133-0001.jpeg",
        "../image_set/train/NORMAL/IM-0135-0001.jpeg",
        "../image_set/train/NORMAL/IM-0137-0001.jpeg",
        "../image_set/train/NORMAL/IM-0140-0001.jpeg",
        "../image_set/train/NORMAL/IM-0141-0001.jpeg",
        "../image_set/train/NORMAL/IM-0143-0001.jpeg",
        "../image_set/train/NORMAL/IM-0145-0001.jpeg",
        "../image_set/train/NORMAL/IM-0147-0001.jpeg",
        "../image_set/train/NORMAL/IM-0149-0001.jpeg",
    ]

    for image_path in image_paths:
        # Read image
        image = cv.imread(image_path)
        # display_image("Input image", image)

        clustered = cluster(image, 4)
        # display_image("Clustering", clustered)

        clustered = cluster(clustered, 2)
        # display_image("Clustering", clustered)

        dilated = dilate(clustered, 4)
        # display_image("Dilated", dilated)

        contoured, contours = contour(dilated, True)
        # display_image("Dilated Contour", contoured)

        # cv.fillPoly(contoured, pts=contours, color=(255, 255, 255))
        # display_image("Filled Contour", contoured)

        flood_mask = np.zeros(contoured.shape[:2], dtype=np.uint8)
        flood_mask = cv.copyMakeBorder(flood_mask, 1, 1, 1, 1, borderType=cv.BORDER_CONSTANT)
        # flood_mask = np.zeros(tuple(map(lambda dim: dim + 2, contoured.shape[:2])), dtype=np.uint8)

        flood_input = contoured.astype('uint8')
        cv.floodFill(image=flood_input,
                     mask=flood_mask,
                     seedPoint=(1, flood_input.shape[0] // 2),
                     newVal=(255, 0, 0),
                     flags=cv.FLOODFILL_MASK_ONLY
                     )

        cv.floodFill(image=flood_input,
                     mask=flood_mask,
                     seedPoint=(flood_input.shape[1] - 1, flood_input.shape[0] // 2),
                     newVal=(255, 0, 0),
                     flags=cv.FLOODFILL_MASK_ONLY
                     )

        flood_mask = flood_mask[1:flood_mask.shape[0] - 1, 1: flood_mask.shape[1] - 1]
        flood_mask = cv.bitwise_not(flood_mask)
        flood_mask = cv.threshold(flood_mask, 0, 255, cv.THRESH_OTSU)[1]

        output_image = cv.bitwise_and(image, image, mask=flood_mask)
        display_image("Output", output_image)


        # cannyed = canny(image)
        # display_image("Canny", cannyed)

        # houghed = hough(clustered, threshold=10, min_line_length=1, max_line_gap=10)
        # display_image("Hough", houghed)
        # contoured = contour(clustered, False)
        # display_image("Cluster Contour", contoured)

        # contoured = contour(dilated, False)
        # display_image("Dilated Contour", contoured)

    cv.waitKey()


if __name__ == "__main__":
    print(f"Running on Python version {sys.version}")

    main()
