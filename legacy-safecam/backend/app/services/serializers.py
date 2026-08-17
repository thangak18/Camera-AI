from __future__ import annotations

from app.models.entities import Clip, Video


def media_url(relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    return f"/media/{relative_path.lstrip('/')}"


def video_payload(video: Video) -> dict:
    return {
        "id": video.id,
        "camera_id": video.camera_id,
        "camera_name": video.camera.name if video.camera else None,
        "location": video.camera.location if video.camera else None,
        "original_filename": video.original_filename,
        "status": video.status,
        "duration_seconds": video.duration_seconds,
        "recorded_at": video.recorded_at,
        "created_at": video.created_at,
        "processed_at": video.processed_at,
        "error_message": video.error_message,
        "video_url": media_url(video.storage_path),
        "clip_count": len(video.clips or []),
    }


def clip_payload(clip: Clip) -> dict:
    return {
        "id": clip.id,
        "video_id": clip.video_id,
        "camera_id": clip.camera_id,
        "camera_name": clip.camera.name if clip.camera else None,
        "location": clip.camera.location if clip.camera else None,
        "start_second": clip.start_second,
        "end_second": clip.end_second,
        "start_time": clip.start_time,
        "end_time": clip.end_time,
        "caption": clip.caption,
        "objects": clip.objects or [],
        "attributes": clip.attributes or {},
        "confidence": clip.confidence,
        "thumbnail_url": media_url(clip.thumbnail_path),
        "clip_url": media_url(clip.clip_path),
        "source_video_url": media_url(clip.video.storage_path if clip.video else None),
    }

