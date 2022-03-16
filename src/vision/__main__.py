#!/usr/bin/env python3.10
#  #!/usr/bin/env python3
import random
from pathlib import Path
import sys

from pprint import pprint
from lib.manipulate import full_manipulation
from lib.network import equalize_labels, load_data


def main():
    train_images: list[tuple[Path, str]] = list(load_data(Path("../image_set/train/"), data_type="train"))
    random.shuffle(train_images)
    train_images = equalize_labels(train_images)
    manipulated_images = full_manipulation([image[0] for image in train_images])
    manipulated_images_with_labels = [(manipulated_images[i], train_images[i][1]) for i, _ in
                                      enumerate(manipulated_images)]


if __name__ == "__main__":
    print(f"Running on Python version {sys.version}")

    main()
