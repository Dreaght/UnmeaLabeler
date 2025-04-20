from pathlib import Path
import random
import shutil

from core.util.image_names_normalizer import normalize_all

base_path = Path().resolve()


def convert_from_label_studio(datasets_folder: Path):
    datasets = list(datasets_folder.glob("*.zip"))

    extract_dir = base_path / "dataset/extracted"
    extract_dir.mkdir(exist_ok=True)

    for zip_file in datasets:
        shutil.unpack_archive(zip_file, extract_dir / zip_file.stem)

    for dataset_dir in extract_dir.iterdir():
        image_path = dataset_dir / "images"
        labels_path = dataset_dir / "labels"

        if not image_path.exists() or not labels_path.exists():
            print(f"Skipping {dataset_dir}, missing 'images' or 'labels' folder.")
            continue

        images = list(image_path.iterdir())
        labels = list(labels_path.iterdir())

        if len(images) != len(labels):
            print(f"Warning: Image/label count mismatch in {dataset_dir}")

        paired = list(zip(images, labels))
        random.shuffle(paired)

        split_index = int(len(paired) * 0.75)
        train_pairs = paired[:split_index]
        val_pairs = paired[split_index:]

        for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
            images_dst = base_path / f"dataset/images/{split}"
            labels_dst = base_path / f"dataset/labels/{split}"
            images_dst.mkdir(parents=True, exist_ok=True)
            labels_dst.mkdir(parents=True, exist_ok=True)

            for img_path, lbl_path in pairs:
                # Rename/copy the original image/label (normalize later)
                shutil.copy(img_path, images_dst / img_path.name)
                shutil.copy(lbl_path, labels_dst / lbl_path.name)

    normalize_all(base_path)
