from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import math
import re
import shutil
import subprocess
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.orm import Session, selectinload

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models.entities import Camera, Clip, Video, utc_now
from app.services.embedding import HashingEmbedding
from app.services.frame_analyzer import FrameAnalyzer, create_placeholder_thumbnail


SAFE_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def secure_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename).strip("._")
    return cleaned or "camera_video.mp4"


def save_upload_file(file: UploadFile) -> tuple[str, Path]:
    settings = get_settings()
    original_name = secure_filename(file.filename or "camera_video.mp4")
    suffix = Path(original_name).suffix.lower()
    if suffix not in SAFE_SUFFIXES:
        suffix = ".mp4"

    relative_path = f"videos/{uuid4()}{suffix}"
    destination = settings.media_path / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return relative_path, destination


def process_video_job(video_id: str) -> None:
    with SessionLocal() as db:
        VideoProcessor(db).process(video_id)


class VideoProcessor:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()
        self.embedder = HashingEmbedding()
        self.analyzer = FrameAnalyzer()

    def process(self, video_id: str) -> None:
        video = (
            self.db.query(Video)
            .options(selectinload(Video.camera))
            .filter(Video.id == video_id)
            .first()
        )
        if not video:
            return

        video.status = "processing"
        video.error_message = None
        self.db.commit()

        try:
            video_path = self.settings.media_path / video.storage_path
            duration = self._probe_duration(video_path)
            video.duration_seconds = duration

            self.db.query(Clip).filter(Clip.video_id == video.id).delete()
            self.db.flush()

            segment_seconds = max(1, self.settings.clip_seconds)
            segment_count = max(1, math.ceil(duration / segment_seconds))

            for index in range(segment_count):
                start_second = float(index * segment_seconds)
                end_second = float(min(duration, start_second + segment_seconds))
                clip_id = str(uuid4())
                thumbnail_relative = f"thumbnails/{clip_id}.jpg"
                clip_relative = f"clips/{clip_id}.mp4"
                thumbnail_path = self.settings.media_path / thumbnail_relative
                clip_path = self.settings.media_path / clip_relative

                self._extract_thumbnail(video_path, thumbnail_path, start_second, end_second)
                self._extract_clip(video_path, clip_path, start_second, end_second)

                analysis = self.analyzer.analyze(
                    thumbnail_path,
                    filename=video.original_filename,
                    location=video.camera.location if video.camera else "unknown",
                    start_second=start_second,
                    end_second=end_second,
                )
                embedding_text = " ".join(
                    [
                        analysis.caption,
                        video.original_filename,
                        video.camera.name if video.camera else "",
                        video.camera.location if video.camera else "",
                        " ".join(analysis.objects),
                    ]
                )

                start_time = None
                end_time = None
                if video.recorded_at:
                    start_time = video.recorded_at + timedelta(seconds=start_second)
                    end_time = video.recorded_at + timedelta(seconds=end_second)

                self.db.add(
                    Clip(
                        id=clip_id,
                        video_id=video.id,
                        camera_id=video.camera_id,
                        start_second=start_second,
                        end_second=end_second,
                        start_time=start_time,
                        end_time=end_time,
                        caption=analysis.caption,
                        objects=analysis.objects,
                        attributes=analysis.attributes,
                        thumbnail_path=thumbnail_relative,
                        clip_path=clip_relative if clip_path.exists() else None,
                        embedding=self.embedder.embed(embedding_text),
                        confidence=analysis.confidence,
                    )
                )

            video.status = "processed"
            video.processed_at = utc_now()
            self.db.commit()
        except Exception as exc:
            self.db.rollback()
            video = self.db.query(Video).filter(Video.id == video_id).first()
            if video:
                video.status = "failed"
                video.error_message = str(exc)
                self.db.commit()

    def _probe_duration(self, path: Path) -> float:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return max(0.1, float(result.stdout.strip()))

    def _extract_thumbnail(self, source: Path, destination: Path, start: float, end: float) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        seek = max(0.0, start + min(1.0, (end - start) / 2))
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{seek:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            "scale=960:-1",
            "-q:v",
            "4",
            str(destination),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except Exception:
            create_placeholder_thumbnail(destination, source.name)

    def _extract_clip(self, source: Path, destination: Path, start: float, end: float) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        duration = max(0.1, end - start)
        copy_command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            str(destination),
        ]
        try:
            subprocess.run(copy_command, check=True, capture_output=True, text=True)
            return
        except Exception:
            pass

        transcode_command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-an",
            str(destination),
        ]
        subprocess.run(transcode_command, check=True, capture_output=True, text=True)


def get_or_create_camera(db: Session, camera_id: str | None, location: str | None) -> Camera:
    if camera_id:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if camera:
            return camera

    camera = db.query(Camera).order_by(Camera.created_at.asc()).first()
    if camera:
        if location and camera.location != location:
            camera.location = location
            db.commit()
            db.refresh(camera)
        return camera

    camera = Camera(name="CAM_01", location=location or "Cong chinh")
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera

