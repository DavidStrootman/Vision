#!/usr/bin/env python3.10
#  #!/usr/bin/env python3
from pathlib import Path
import cv2 as cv
import sys
import matplotlib.pyplot as plt
import numpy as np
from lib.util import display_image
from lib.manipulate import cluster, canny, contour, hough, dilate, adjust_gamma


def main():
    image_paths = [
        "../image_set/train/NORMAL/IM-0115-0001.jpeg",
        "../image_set/train/NORMAL/IM-0117-0001.jpeg",
        "../image_set/train/NORMAL/IM-0119-0001.jpeg",
        "../image_set/train/NORMAL/IM-0122-0001.jpeg",
        "../image_set/train/NORMAL/IM-0125-0001.jpeg",
        "../image_set/train/NORMAL/IM-0128-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0129-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0131-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0133-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0135-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0137-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0140-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0141-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0143-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0145-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0147-0001.jpeg",
        # "../image_set/train/NORMAL/IM-0149-0001.jpeg",
    ]

    n = 50

    path = Path("../image_set/train/NORMAL/")
    fig, grid = plt.subplots(nrows=n//10, ncols=10, dpi=1000)

    for i, image_path in enumerate(path.iterdir()):
        j = i // 10
        print(image_path)
        if i == n:
            break
        # Read image
        image = cv.imread(image_path.as_posix()).astype("uint8")

        # display_image("Input image", image)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        equalized = cv.equalizeHist(image)

        gamma = adjust_gamma(equalized, 5.0)
        # fig, ((input, output)) = plt.subplots(1, 2)
        # display_image("Equalized", equalized)

        # clustered = cluster(gamma, 7)
        # display_image("Clustering", clustered)

        _, max_clusters = cv.threshold(gamma, 208, 255, cv.THRESH_BINARY)
        # display_image("Max clusters", max_clusters)

        # Remove photo artifacts from some images:
        erode_kernel = np.ones((2, 2), np.uint8)
        eroded = cv.erode(src=max_clusters, kernel=erode_kernel)

        # cv.imshow("Eroded", eroded)
        # cv.waitKey(0)

        # Remove small components in the mask
        min_size = eroded.shape[0] * eroded.shape[1] // 10
        n_components, labels, stats, _ = cv.connectedComponentsWithStats(eroded, connectivity=8)
        sizes = stats[1:, -1]

        filtered_components = np.zeros(eroded.shape)
        for c in range(0, n_components - 1):
            if sizes[c] >= min_size:
                filtered_components[labels == c + 1] = 255

        # Dilate to connect "small" gaps
        blur_kernel = (11, 11)
        dilated = cv.GaussianBlur(filtered_components, blur_kernel, 2, 2)
        dilate_kernel = (10, 10)
        dilated = dilate(dilated, dilate_kernel, 5)
        # display_image("Dilated Mask", dilated)

        # display_image("Filtered Components", filtered_components)

        flood_mask = np.zeros(dilated.shape, dtype=np.uint8)
        flood_mask = cv.copyMakeBorder(flood_mask, 1, 1, 1, 1, borderType=cv.BORDER_CONSTANT)

        flood_input = dilated.astype('uint8')
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

        # Remove 1 pixel border from all sides
        flood_mask = flood_mask[1:flood_mask.shape[0] - 1, 1: flood_mask.shape[1] - 1]
        _, flood_mask = cv.threshold(flood_mask, 0, 255, cv.THRESH_BINARY_INV)
        # display_image("Flood mask", flood_mask)

        # cv.imshow("Flood mask", flood_mask)

        output_image = cv.bitwise_and(image, image, mask=flood_mask)
        # display_image(f"Input Image {i + 1}", image)
        # display_image(f"Output Image {i + 1}", output_image)
        grid[j, i % 10].axis("off")

        grid[j, i % 10].imshow(output_image)
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    print(f"Running on Python version {sys.version}")

    main()
