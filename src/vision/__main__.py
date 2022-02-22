#!/usr/bin/env python3.10
#  #!/usr/bin/env python3
from pathlib import Path
import sys

from lib.util import display_image


def main():
    image_path: Path = Path("../image_set/train/NORMAL/IM-0115-0001.jpeg")
    display_image(image_path)


if __name__ == "__main__":
    print(f"Running on Python version {sys.version}")

    main()
