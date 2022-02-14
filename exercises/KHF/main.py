from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.color.adapt_rgb
from skimage import io


def main():
    fig, ((img1, img2, img3), (hist1, hist2, hist3)) = plt.subplots(2, 3)

    image_path = Path('./beach.jpg')
    image = io.imread(image_path.as_posix()).astype(float)

    img1.imshow(image.astype(int))

    hue = skimage.color.rgb2hsv(image)[:, :, 0]

    red_grayscale = skimage.color.gray2rgb(skimage.color.rgb2gray(image))
    blue_grayscale = red_grayscale.copy()

    # Red mask
    color_mask = np.logical_and(hue > (320 / 360), hue < (360 / 360))
    red_grayscale[color_mask] = image[color_mask]
    img2.imshow(red_grayscale.astype(int))

    # Blue mask
    color_mask = np.logical_and(hue > (150 / 360), hue < (210 / 360))
    blue_grayscale[color_mask] = image[color_mask]
    img3.imshow(blue_grayscale.astype(int))

    bins = 20
    range_ = (0, 1)

    image_hue = skimage.color.rgb2hsv(image)[:, :, 0].flatten()
    hist1.hist(image_hue, bins, range_)
    hist1.set_title("Original")

    red_hue = skimage.color.rgb2hsv(red_grayscale)[:, :, 0].flatten()
    hist2.hist(red_hue, bins, range_)
    hist2.set_title("Red filter")

    blue_hue = skimage.color.rgb2hsv(blue_grayscale)[:, :, 0].flatten()
    hist3.hist(blue_hue, bins, range_)
    hist3.set_title("Blue filter")


    plt.show()


if __name__ == '__main__':
    main()
