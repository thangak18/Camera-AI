from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageStat

from app.services.embedding import normalize_text


COLOR_PALETTE: dict[str, tuple[int, int, int]] = {
    "black": (20, 20, 20),
    "white": (235, 235, 235),
    "red": (205, 45, 45),
    "yellow": (220, 190, 60),
    "blue": (50, 105, 205),
    "green": (55, 150, 90),
    "gray": (125, 125, 125),
}

COLOR_VI = {
    "black": "den",
    "white": "trang",
    "red": "do",
    "yellow": "vang",
    "blue": "xanh duong",
    "green": "xanh la",
    "gray": "xam",
}

FILENAME_HINTS = {
    "person": ["person", "nguoi", "human"],
    "motorbike": ["motorbike", "xe may", "bike"],
    "car": ["car", "oto", "o to"],
    "helmet": ["helmet", "mu bao hiem", "non bao hiem"],
    "bag": ["bag", "balo", "tui"],
    "gate": ["gate", "cong"],
    "parking": ["parking", "bai xe"],
    "corridor": ["corridor", "hanh lang"],
    "enter": ["enter", "di vao"],
    "exit": ["exit", "di ra"],
}


@dataclass(frozen=True)
class AnalysisResult:
    caption: str
    objects: list[str]
    attributes: dict
    confidence: float


class FrameAnalyzer:
    def analyze(
        self,
        thumbnail_path: Path | None,
        *,
        filename: str,
        location: str,
        start_second: float,
        end_second: float,
    ) -> AnalysisResult:
        attributes: dict = {
            "analyzer": "local_frame_metadata",
            "start_second": round(start_second, 2),
            "end_second": round(end_second, 2),
        }
        objects = self._infer_objects_from_filename(filename)

        color_name = "gray"
        brightness_label = "khong ro"
        if thumbnail_path and thumbnail_path.exists():
            color_name, brightness_label = self._inspect_image(thumbnail_path)
            attributes["dominant_color"] = color_name
            attributes["brightness"] = brightness_label

        color_vi = COLOR_VI.get(color_name, color_name)
        object_text = ", ".join(objects) if objects else "scene"
        caption = (
            f"Doan {start_second:.0f}-{end_second:.0f}s tai {location}. "
            f"Khung hinh chinh co tong mau {color_vi}, do sang {brightness_label}. "
            f"Metadata goi y doi tuong: {object_text}."
        )

        confidence = 0.55 if objects else 0.4
        if color_name != "gray":
            confidence += 0.1

        return AnalysisResult(
            caption=caption,
            objects=objects or ["scene"],
            attributes=attributes,
            confidence=min(confidence, 0.8),
        )

    def _infer_objects_from_filename(self, filename: str) -> list[str]:
        normalized = normalize_text(filename)
        found: list[str] = []
        for canonical, hints in FILENAME_HINTS.items():
            if any(hint in normalized for hint in hints):
                found.append(canonical)
        return sorted(set(found))

    def _inspect_image(self, path: Path) -> tuple[str, str]:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((1, 1))
            rgb = image.getpixel((0, 0))

        color_name = min(
            COLOR_PALETTE,
            key=lambda name: sum((rgb[index] - COLOR_PALETTE[name][index]) ** 2 for index in range(3)),
        )
        mean_brightness = sum(rgb) / 3
        if mean_brightness < 70:
            brightness_label = "thap"
        elif mean_brightness > 185:
            brightness_label = "cao"
        else:
            brightness_label = "trung binh"

        return color_name, brightness_label


def create_placeholder_thumbnail(path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (960, 540), color=(32, 38, 46))
    # Keep the placeholder intentionally simple; real evidence frames come from FFmpeg.
    stat = ImageStat.Stat(image)
    _ = stat.mean
    image.save(path, quality=86)

