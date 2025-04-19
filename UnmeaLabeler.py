from pathlib import Path
import argparse

from core.converter import initialize
from core.migrate_from_label_studio import convert_from_label_studio
from gui.labeler import run_labeling


def main():
    initialize(
        Path(args.dataset),
        train_sample_size=args.train_sample_size,
        val_sample_size=args.val_sample_size,
        seed=42,
        should_purge=args.purge
    )

    if args.label_studio_path:
        convert_from_label_studio(Path(args.label_studio_path))

    run_labeling()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UnmeaLabeler Studio')
    parser.add_argument('-d', '--dataset', type=str, default='./food-101/',
                        help='Food-101 dataset path')
    parser.add_argument('-m', '--label-studio-path', default="", help='Migrate from label studio (Specify a path)')
    parser.add_argument('-t', '--train-sample-size', type=int, default=15, help='Training sample size')
    parser.add_argument('-v', '--val-sample-size', type=int, default=5, help='Validation sample size')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Seed')
    parser.add_argument('-p', '--purge', type=int, default=1, help='Purge dataset before starting (0 / 1)')
    args = parser.parse_args()

    main()
