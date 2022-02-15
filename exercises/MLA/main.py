import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import data


def main():
    fig, (img1, img2, img3, img4) = plt.subplots(1, 4)
    image = data.chelsea()

    img1.imshow(image)

    rotate_transform = skimage.transform.AffineTransform(rotation=np.radians(20))
    rotated_image = skimage.transform.warp(image, rotate_transform)

    img2.imshow(rotated_image)

    translate_transform = skimage.transform.AffineTransform(translation=[-50, -50])
    translated_image = skimage.transform.warp(image, translate_transform)

    img3.imshow(translated_image)

    stretch_transform = skimage.transform.AffineTransform(scale=[0.5, 1])
    stretched_image = skimage.transform.warp(image, stretch_transform)

    img4.imshow(stretched_image)

    fig.show()


if __name__ == '__main__':
    main()
