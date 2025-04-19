from pathlib import Path


def normalize_all(base_path: Path) -> None:
    folders = [
        base_path / "dataset/images/train",
        base_path / "dataset/images/val",
        base_path / "dataset/labels/train",
        base_path / "dataset/labels/val",
    ]

    for folder in folders:
        for file in folder.iterdir():
            if file.is_file():
                parts = file.name.split("-", 1)
                if len(parts) == 2:
                    new_name = parts[1]
                    new_path = folder / new_name
                    file.rename(new_path)