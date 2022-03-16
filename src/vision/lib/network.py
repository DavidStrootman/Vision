from enum import Enum
from pathlib import Path
from itertools import chain

import numpy as np
from tensorflow.keras.models import Sequential
from typing import TypeVar


def create_model():
    model: Sequential = Sequential()

    return model


class DataType(Enum):
    TEST = "test"
    TRAIN = "train"
    VALIDATION = "val"


class Labels(Enum):
    NORMAL = "normal"
    BACTERIA = "bacteria"
    VIRUS = "virus"


def _get_label_for_image(path: Path) -> str:
    path_str = str(path)
    if "_bacteria_" in path_str:
        return Labels.BACTERIA.value
    if "_virus_" in path_str:
        return Labels.VIRUS.value
    raise AttributeError(f"Could not get label for file: {path}")


images_with_labels = TypeVar("images_with_labels", list[tuple[Path, str]], list[tuple[np.ndarray, str]])


def equalize_labels(images: images_with_labels) -> images_with_labels:
    """
    Get the minimum amount of a single label and remove all images with a label that exceed that amount.
    Does not maintain ordering.
    """
    label_minimum = float("inf")

    # Get the minimum amount of a single label
    for label in Labels:
        label_count = 0
        for image, label_ in images:
            if label_ == label.value:
                label_count += 1

        if label_count < label_minimum:
            label_minimum = label_count

    # Clamp all labels at the minimum size
    for label in Labels:
        count_images_with_label = 0
        images_to_remove = []
        for image_with_label in images:
            if image_with_label[1] == label.value:
                if count_images_with_label >= label_minimum:
                    images_to_remove.append(image_with_label)
                else:
                    count_images_with_label += 1

        for image in images_to_remove:
            images.remove(image)

    return images


def load_data(path: Path, data_type: str) -> chain[tuple[Path, str]]:
    """
    Returns the path to an image and the label
    """
    data_type = DataType(data_type)

    if data_type is DataType.TRAIN:
        normal_dir = path / "NORMAL"
        pneumonia_dir = path / "PNEUMONIA"

        normal_files = normal_dir.iterdir()
        pneumonia_files = list(pneumonia_dir.iterdir())

        normal_data_labels = map(lambda _: Labels.NORMAL.value, normal_files)

        pneumonia_data_labels = map(_get_label_for_image, pneumonia_files)

        return chain(zip(normal_files, normal_data_labels), zip(pneumonia_files, pneumonia_data_labels))

    if data_type is DataType.TEST:
        pass
    if data_type is DataType.VALIDATION:
        pass
