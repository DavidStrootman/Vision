import math
import random
from typing import Any

from sklearn import datasets
from sklearn import svm


def shuffled_digits() -> tuple[Any, Any]:
    digits = datasets.load_digits()
    zipped_digits = list(zip(digits.data, digits.target))

    random.shuffle(zipped_digits)

    data_, target_ = zip(*zipped_digits)
    return data_, target_


data, target = shuffled_digits()

clf = svm.SVC(gamma=0.001, C=100)
twothirds = math.floor(len(data) / 3 * 2)

X, y = data[:twothirds], target[:twothirds]
clf.fit(X, y)

test_data = data[twothirds:]
test_targets = target[twothirds:]

predictions = clf.predict(test_data)

matches = [prediction == test_targets[i] for i, prediction in enumerate(predictions)]
accuracy = sum(matches) / len(matches)

print(accuracy)

