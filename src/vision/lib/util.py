"""
File provides image utilities
TODO
"""
from pathlib import Path
import matplotlib.pyplot as plt
import skimage
import numpy as np


def display_image_from_path(title: str, image_path: Path):
    image = skimage.io.imread(image_path.as_posix()).astype(float)
    display_image(title, image)


def display_image(title: str, image: np.ndarray):
    plt.title(title)
    plt.axis("off")
    plt.imshow(image)
    plt.show()
