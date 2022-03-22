from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, Optional, TypeVar, Union
import math

import numpy as np
import cv2 as cv
import tensorflow.keras as ks
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical, Sequence

images_with_labels = TypeVar("images_with_labels", list[tuple[Path, str]], list[tuple[np.ndarray, str]])


class Labels(Enum):
    NORMAL = 0
    BACTERIA = 1
    VIRUS = 2


class XraySequence(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x_set = np.asarray(x_set)
        self.y_set = y_set
        self.batch_size = batch_size
        self.shape_ = (len(self.x_set),) + self.x_set[0].shape

    def __len__(self):
        return math.ceil(len(self.x_set) / self.batch_size)

    def __getitem__(self, item):
        batch_x = self.x_set[item * self.batch_size:(item + 1) * self.batch_size]
        batch_y = self.y_set[item * self.batch_size:(item + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)

    def __iter__(self):
        return ((image, label) for image, label in zip(self.x_set, self.y_set))


def _get_label_for_image(path: Path) -> str:
    path_str = str(path)
    if "_bacteria_" in path_str:
        return Labels.BACTERIA.value
    if "_virus_" in path_str:
        return Labels.VIRUS.value
    raise AttributeError(f"Could not get label for file: {path}")


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


def load_data_paths(path: Path) -> chain[tuple[Path, str]]:
    """
    Returns the path to an image and the label
    """

    normal_dir = path / "NORMAL"
    pneumonia_dir = path / "PNEUMONIA"

    normal_files = normal_dir.iterdir()
    pneumonia_files = list(pneumonia_dir.iterdir())

    normal_data_labels = map(lambda _: Labels.NORMAL.value, normal_files)

    pneumonia_data_labels = map(_get_label_for_image, pneumonia_files)

    return chain(zip(normal_files, normal_data_labels), zip(pneumonia_files, pneumonia_data_labels))


def load_data(paths: chain[tuple[Path, str]], batch_size) -> XraySequence:
    images = []
    labels = []
    for path, label in paths:
        images.append(cv.imread(str(path)).astype("uint8"))
        labels.append(label)

    if len(images) != len(labels):
        raise RuntimeError("Got different lengths for images and labels.")

    images = np.asarray(images)
    labels = np.asarray(labels)

    return XraySequence(images, labels, batch_size)


def create_model(input_shape: tuple) -> ks.models.Model:
    model: ks.models.Sequential = ks.models.Sequential()
    # input
    # model.add(Input(shape=input_shape))
    # Network
    model.add(Conv2D(filters=32, kernel_size=3, padding="same", input_shape=input_shape, activation="relu"))
    model.add(MaxPooling2D(pool_size=2, padding="same"))

    # Output
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))

    # Compile
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(model: ks.models.Model, train_data: XraySequence, validation_data: XraySequence) -> \
        tuple[ks.models.Model, Optional[ks.callbacks.History]]:
    """
    Trains a model and returns the trained model along with the training history, if at least one epoch has been run.
    """
    history: ks.callbacks.History = model.fit(x=train_data.x_set, y=to_categorical(train_data.y_set), epochs=10)

    return model, history


def evaluate(model: ks.models.Model, images: images_with_labels) -> Union[Any, list[Any]]:
    return model.evaluate()
