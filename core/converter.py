import shutil
import random
from pathlib import Path


base_path = Path().resolve()


def reinitialize(dataset_path: Path, train_sample_size: int = 15, val_sample_size: int = 5, seed: int = 42):
    purge()
    initialize(dataset_path, train_sample_size, val_sample_size, seed)


def initialize(dataset_path: Path, train_sample_size: int = 15, val_sample_size: int = 5, seed: int = 42):
    images_path = dataset_path / 'images'
    meta_path = dataset_path / 'meta'

    generate_dataset_structure()
    generate_data_yaml(get_classes_id(meta_path))
    populate_images(images_path, meta_path, train_sample_size, val_sample_size, seed)


def purge():
    if Path(base_path / "dataset").exists():
        shutil.rmtree(base_path / "dataset")


def generate_dataset_structure():
    structure = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/images/unlabeled",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/labels/pseudo",
    ]

    for folder in structure:
        (base_path / folder).mkdir(parents=True, exist_ok=True)


def generate_data_yaml(classes_id: dict[int, str]):
    content_text = """\
path: dataset
train: images/train
val: images/val
names:
"""
    for i, class_name in classes_id.items():
        content_text += f"  {i}: {class_name}\n"

    (base_path / "dataset/data.yaml").write_text(content_text)


def get_classes_id(meta_folder: Path) -> dict[int, str]:
    with open(meta_folder / "classes.txt") as f:
        class_names = [line.strip() for line in f]
        return {i: name for i, name in enumerate(class_names)}


def populate_images(images_path: Path, meta_path: Path, train_sample_size: int, val_sample_size: int, seed: int):
    random.seed(seed)

    train_paths = open(meta_path / "train.txt").read().splitlines()
    val_paths = open(meta_path / "test.txt").read().splitlines()

    sampled_train = random.sample(train_paths, min(train_sample_size, len(train_paths)))
    sampled_val = random.sample(val_paths, min(val_sample_size, len(val_paths)))

    copy_images(images_path, sampled_train, "dataset/images/train")
    copy_images(images_path, sampled_val, "dataset/images/val")


def copy_images(images_path: Path, paths: list[str], dst_folder: str):
    map_file = base_path / "dataset" / "paths.txt"

    with open(map_file, "a") as f:
        for path in paths:
            src = images_path / (path + ".jpg")
            dst = base_path / dst_folder
            shutil.copy(src, dst)

            f.write(f"{src} -> {dst / (path.split("/")[1] + ".jpg")}\n")
