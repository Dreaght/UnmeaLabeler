from pathlib import Path
import argparse

from core.converter import reinitialize, initialize
from gui.labeler import run_labeling


def main(food_dataset_path: Path, purge: int = 1):
    if purge:
        reinitialize(food_dataset_path)
    else:
        initialize(food_dataset_path)
    run_labeling()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UnmeaLabeler Studio')
    parser.add_argument('-d', '--dataset', type=str, default='./food-101/',
                        help='Food-101 dataset path')
    parser.add_argument('-t', '--train-sample-size', type=int, default=15, help='Training sample size')
    parser.add_argument('-v', '--val-sample-size', type=int, default=5, help='Validation sample size')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Seed')
    parser.add_argument('-p', '--purge', type=int, default=1, help='Purge dataset before starting (0 / 1)')
    args = parser.parse_args()

    main(food_dataset_path=Path(args.dataset), purge=args.purge)
