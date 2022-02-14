import skimage
from skimage import data
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage


def main1():
    image = data.camera()
    fig, (img1, img2, img3, img4) = plt.subplots(1, 4)

    mask1 = [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]]

    mask2 = [[1, 0, -1],
             [0, 0, 0],
             [-1, 0, 1]]

    mask3 = [[0, -0.25, 0],
             [0.25, 0, 0.25],
             [0, -0.25, 0]]

    newimage1 = scipy.ndimage.convolve(image, mask1)

    newimage2 = scipy.ndimage.convolve(image, mask2)

    newimage3 = scipy.ndimage.convolve(image, mask3)

    img1.imshow(image)
    img2.imshow(newimage1)
    img3.imshow(newimage2)
    img4.imshow(newimage3)

    plt.show()


def main2():
    image = data.camera()
    fig, (img1, img2, img3, img4) = plt.subplots(1, 4)

    # het valt op dat ze deze filter veel beter werken, de edges zijn duidelijker.
    newimage1 = skimage.filters.edges.farid(image)
    newimage2 = skimage.filters.edges.roberts(image)
    newimage3 = skimage.filters.edges.prewitt(image)

    img1.imshow(image)
    img2.imshow(newimage1)
    img3.imshow(newimage2)
    img4.imshow(newimage3)

    plt.show()


def main3():
    plt.imshow(skimage.feature.canny(data.camera(), sigma=2.2))

    plt.show()


if __name__ == '__main__':
    # main1()
    # main2()
    main3()
