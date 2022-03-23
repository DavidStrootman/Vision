#!/usr/bin/env python3.10
#  #!/usr/bin/env python3
import random
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import numpy as np
from lib.manipulate import full_manipulation
from pprint import pprint
from lib.network import equalize_labels, load_data_paths, load_data, create_model, train_model, validate_model, \
    XraySequence


def get_images(path: str, max_count=100, batch_size=4) -> XraySequence:
    assert max_count % batch_size == 0
    image_paths: list[tuple[Path, str]] = list(load_data_paths(Path(path)))
    random.shuffle(image_paths)
    image_paths = equalize_labels(image_paths)
    image_paths = image_paths[:max_count]

    images = load_data(image_paths, batch_size)

    return full_manipulation(images)


def main():
    batch_size = 4
    max_image_count = 400
    train_images = get_images("../image_set/train/", max_image_count, batch_size=batch_size)
    test_images = get_images("../image_set/test/", max_image_count, batch_size=batch_size)
    validation_images = get_images("../image_set/val/", max_image_count, batch_size=batch_size)

    train_images.x_set = np.expand_dims(train_images.x_set, axis=3)
    images_shape: tuple = train_images.x_set.shape[1:]

    plt.imshow(train_images.x_set[0])
    plt.show()

    model = create_model(input_shape=images_shape, num_filters=4, batch_size=batch_size)
    model, history = train_model(model, train_images, test_images)
    validation = validate_model(model, validation_images, batch_size)
    pprint(validation)
    return validation


if __name__ == "__main__":
    print(f"Running on Python version {sys.version}")

    main()
