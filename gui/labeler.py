from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt6.QtGui import QPixmap, QPainter, QPen, QFont, QColor, QImage
from PyQt6.QtCore import Qt, QRect, QPoint
import sys
import cv2
import numpy as np
from pathlib import Path

base_path = Path(__file__).parent.parent

# Settings
images_dir = Path("dataset/images/train")
labels_dir = Path("dataset/labels/train")

path_map = {}

class_names = []
current_index = 0
image_paths = []
class_id = 0
boxes = []

class LabelingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UnmeaLabeler")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.bbox_start = None
        self.bbox_end = None
        self.drawing = False
        self.current_image = None
        self.scale = 1.0
        self.offset = QPoint(0, 0)

        self.load_image()

    def load_image(self):
        global current_index, image_paths, labels_dir, boxes, class_id

        # Skip non-existent files
        while current_index < len(image_paths) and not image_paths[current_index].exists():
            current_index += 1

        # Check if we've reached the end of the current split
        if current_index >= len(image_paths):
            if "train" in str(labels_dir):
                print("Switching to validation set...")
                image_paths[:] = load_image_paths(Path("dataset/images/val"))
                labels_dir = Path("dataset/labels/val")
                current_index = 0
                class_id = 0
                boxes.clear()
            else:
                print("All validation images processed. Exiting.")
                QApplication.quit()
                return

        # If no images available even after switching
        if not image_paths:
            print("No images found! Exiting.")
            QApplication.quit()
            return

        img_path = image_paths[current_index]
        current_index += 1  # Prepare for next call

        # Try to remap path if available
        line = str(path_map.get(str(base_path / img_path), str(img_path)))
        class_id = get_class_id_from_path(line)

        # Load and convert image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path}")
            self.load_image()  # Skip to next
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_image = img

        h, w, _ = img.shape
        boxes.clear()

        # Load bounding boxes
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, cx, cy, bw, bh = map(float, parts)
                    cls_id = int(cls_id)
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)

                    print(x1, y1, x2, y2, " | ",  h, w)

                    boxes.append((x1, y1, x2, y2, cls_id))

        # Scale image to fit screen
        screen_rect = QApplication.primaryScreen().availableGeometry()
        max_width, max_height = int(screen_rect.width() * 0.9), int(screen_rect.height() * 0.9)

        self.scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * self.scale), int(h * self.scale)
        self.offset = QPoint((max_width - new_w) // 2, (max_height - new_h) // 2)

        self.setFixedSize(new_w, new_h)
        self.image_label.setFixedSize(new_w, new_h)
        self.update()

    def scaled_to_original(self, point: QPoint) -> QPoint:
        return QPoint(
            int((point.x() - self.offset.x()) / self.scale),
            int((point.y() - self.offset.y()) / self.scale)
        )

    def paintEvent(self, event):
        if self.current_image is None:
            return

        painter = QPainter(self)
        h, w, _ = self.current_image.shape
        qimg = QImage(self.current_image.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        painter.drawPixmap(self.rect(), pixmap)

        for x1, y1, x2, y2, cls_id in boxes:
            color = QColor(0, 255, 0) if cls_id == class_id else QColor(255, 0, 0)
            painter.setPen(QPen(color, 2))
            rect = QRect(QPoint(x1, y1), QPoint(x2, y2))
            scaled_rect = QRect(
                rect.topLeft() * self.scale + self.offset,
                rect.bottomRight() * self.scale + self.offset
            )
            painter.drawRect(scaled_rect)
            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            painter.drawText(scaled_rect.topLeft(), label)

        if self.drawing and self.bbox_start and self.bbox_end:
            r = QRect(self.bbox_start, self.bbox_end)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawRect(r)

        # Draw overlay text
        font = QFont("Arial", 10)
        painter.setFont(font)
        metrics = painter.fontMetrics()

        text_lines = []

        line = str(path_map.get(str(base_path / image_paths[current_index]), "")) # e.x: /home/dreaght/Documents/food-101/images/ceviche/1648055.jpg'
        max_width = self.width() - 10
        for i, name in enumerate(class_names):
            entry = f"[{i}] {name}" if i != class_id else f"→[{i}] {name}←"
            if metrics.horizontalAdvance(line + " " + entry) > max_width:
                text_lines.append(line)
                line = entry
            else:
                line += " " + entry if line else entry
        if line:
            text_lines.append(line)

        box_height = metrics.height() * len(text_lines) + 10
        rect = QRect(0, self.height() - box_height, self.width(), box_height)

        # Transparent background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 30))
        painter.drawRect(rect)

        painter.setPen(QColor(0, 0, 0))
        for i, text in enumerate(text_lines):
            painter.drawText(rect.adjusted(5, 5 + i * metrics.height(), 0, 0), Qt.AlignmentFlag.AlignLeft, text)

        # Image counter in top-left corner
        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        counter_text = f"Image {current_index}/{len(image_paths)}"
        counter_rect = QRect(10, 10, 200, 30)
        painter.setBrush(QColor(0, 0, 0, 100))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(counter_rect)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(counter_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, counter_text)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.bbox_start = event.pos()
            self.drawing = True

    def mouseMoveEvent(self, event):
        print(event.pos())

        if self.drawing:
            pos = event.pos()

            # Only update the box if the cursor is inside the scaled image area
            if 0 <= pos.x() <= self.image_label.width() and \
                    0 <= pos.y() <= self.image_label.height():
                self.bbox_end = pos
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.bbox_start and self.bbox_end:
            self.drawing = False
            p1 = self.scaled_to_original(self.bbox_start)
            p2 = self.scaled_to_original(self.bbox_end)
            x1, y1 = p1.x(), p1.y()
            x2, y2 = p2.x(), p2.y()
            boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), class_id))
            self.bbox_start = None
            self.bbox_end = None
            self.update()

    def keyPressEvent(self, event):
        global current_index, class_id
        key = event.key()
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            num = key - Qt.Key.Key_0
            class_id = (class_id * 10 + num) % 100
        elif key == Qt.Key.Key_Escape:
            class_id = 0
        elif key == Qt.Key.Key_S:
            self.save_labels()
            current_index += 1
            self.load_image()
        elif key == Qt.Key.Key_N:
            current_index += 1
            self.load_image()
        elif key == Qt.Key.Key_Z:
            if boxes:
                boxes.pop()
            self.update()
        elif key == Qt.Key.Key_C:
            boxes.clear()
            self.update()
        elif key == Qt.Key.Key_Q:
            self.close()
        self.update()

    def save_labels(self):
        global image_paths, current_index, labels_dir, class_id

        if not (current_index < len(image_paths) and image_paths[current_index].exists()):
            image_paths = load_image_paths(Path("dataset/images/val"))
            current_index = 0
            class_id = 0  # Reset to default class

            # Also update labels_dir to match the val set
            if "val" in str(image_paths[0]):
                labels_dir = Path("dataset/labels/val")
            else:
                labels_dir = Path("dataset/labels/train")

        img_path = image_paths[current_index]
        h, w, _ = self.current_image.shape
        label_path = labels_dir / (img_path.stem + ".txt")
        with open(label_path, "w") as f:
            for x1, y1, x2, y2, cls_id in boxes:
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = abs(x2 - x1) / w
                bh = abs(y2 - y1) / h
                f.write(f"{cls_id} {cx} {cy} {bw} {bh}\n")


def load_class_names():
    global class_names
    with open("dataset/data.yaml") as f:
        in_names = False
        for line in f:
            line = line.strip()
            if ":" in line and line.split(":")[0].isdigit():
                name = line.split(":")[1].strip()
                class_names.append(name)

def load_image_paths(im_dir: Path = images_dir, review: bool = False):
    related_labels_dir = Path(str(im_dir).replace("images", "labels"))
    if review:
        return sorted([p for p in im_dir.iterdir() if p.suffix == ".jpg"])
    else:
        return sorted([
            p for p in im_dir.rglob("*.jpg")
            if not (related_labels_dir / (p.stem + ".txt")).exists()
        ])

def load_path_mapping():
    global path_map

    path_map = {}
    map_file = Path("dataset/paths.txt")
    if map_file.exists():
        with open(map_file, "r") as f:
            for line in f:
                if "->" in line:
                    src, dst = map(str.strip, line.strip().split("->"))
                    path_map[dst] = src

def get_class_id_from_path(path: str) -> int:
    """
    Extracts the dish name from the path and returns the corresponding class_id.
    Assumes that dish name is the parent directory of the image file.
    """
    for name in class_names:
        if f"/{name}/" in path or f"\\{name}\\" in path:  # Handle both Unix and Windows paths
            return class_names.index(name)
    return 0  # Default class if not found


def run_labeling(review: bool = False):
    global image_paths
    load_class_names()
    image_paths = load_image_paths(review=review)

    load_path_mapping()

    app = QApplication(sys.argv)
    widget = LabelingWidget()
    widget.show()
    app.exec()
