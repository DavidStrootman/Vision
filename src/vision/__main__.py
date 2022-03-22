#!/usr/bin/env python3.10
#  #!/usr/bin/env python3
import random
from pathlib import Path
import sys
import numpy as np
from lib.manipulate import full_manipulation

from lib.network import equalize_labels, load_data_paths, load_data, create_model, train_model, evaluate, XraySequence


def get_images(path: str) -> XraySequence:
    image_paths: list[tuple[Path, str]] = list(load_data_paths(Path(path)))
    random.shuffle(image_paths)
    image_paths = image_paths[:100]
    image_paths = equalize_labels(image_paths)

    images = load_data(image_paths, 32)

    return full_manipulation(images)


def main():
    train_images = get_images("../image_set/train/")
    validation_images = get_images("../image_set/val/")

    train_images.x_set = np.expand_dims(train_images.x_set, axis=2)
    images_shape: tuple = train_images.x_set[0].shape
    model = create_model(input_shape=images_shape)
    return train_model(model, train_images, validation_images)


if __name__ == "__main__":
    print(f"Running on Python version {sys.version}")

    main()
