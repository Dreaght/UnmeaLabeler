from pathlib import Path
import argparse

from core.converter import reinitialize
from gui.labeler import run_labeling


def main(food_dataset_path: Path):
    reinitialize(food_dataset_path)
    run_labeling()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UnmeaLabeler Studio')
    parser.add_argument('-d', '--dataset', type=str, default='./food-101/',
                        help='Food-101 dataset path')
    args = parser.parse_args()

    main(Path(args.dataset))
