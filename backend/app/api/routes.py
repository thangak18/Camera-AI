from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, text
from sqlalchemy.orm import Session, selectinload

from app.core.config import get_settings
from app.db.session import get_db
from app.models.entities import Camera, Clip, Video
from app.schemas import (
    CameraCreate,
    CameraRead,
    ClipRead,
    HealthRead,
    SearchRequest,
    SearchResponse,
    SummaryResponse,
    VideoRead,
)
from app.services.reporting import build_summary
from app.services.search import search_service
from app.services.serializers import clip_payload, video_payload
from app.services.video_processor import get_or_create_camera, process_video_job, save_upload_file


router = APIRouter()


@router.get("/health", response_model=HealthRead)
def health(db: Session = Depends(get_db)) -> HealthRead:
    db.execute(text("SELECT 1"))
    return HealthRead(status="ok", app=get_settings().app_name, database="ok")


@router.get("/stats")
def stats(db: Session = Depends(get_db)) -> dict:
    return {
        "cameras": db.query(func.count(Camera.id)).scalar() or 0,
        "videos": db.query(func.count(Video.id)).scalar() or 0,
        "clips": db.query(func.count(Clip.id)).scalar() or 0,
        "processed_videos": db.query(func.count(Video.id)).filter(Video.status == "processed").scalar() or 0,
    }


@router.get("/cameras", response_model=list[CameraRead])
def list_cameras(db: Session = Depends(get_db)) -> list[Camera]:
    return db.query(Camera).order_by(Camera.created_at.asc()).all()


@router.post("/cameras", response_model=CameraRead)
def create_camera(payload: CameraCreate, db: Session = Depends(get_db)) -> Camera:
    existing = db.query(Camera).filter(Camera.name == payload.name).first()
    if existing:
        raise HTTPException(status_code=409, detail="Camera name already exists")
    camera = Camera(**payload.model_dump())
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera


@router.get("/videos", response_model=list[VideoRead])
def list_videos(db: Session = Depends(get_db)) -> list[dict]:
    videos = (
        db.query(Video)
        .options(selectinload(Video.camera), selectinload(Video.clips))
        .order_by(Video.created_at.desc())
        .all()
    )
    return [video_payload(video) for video in videos]


@router.get("/videos/{video_id}", response_model=VideoRead)
def get_video(video_id: str, db: Session = Depends(get_db)) -> dict:
    video = (
        db.query(Video)
        .options(selectinload(Video.camera), selectinload(Video.clips))
        .filter(Video.id == video_id)
        .first()
    )
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video_payload(video)


@router.post("/videos/upload", response_model=VideoRead)
def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    camera_id: str | None = Form(default=None),
    location: str | None = Form(default=None),
    recorded_at: str | None = Form(default=None),
    process_now: bool = Form(default=False),
    db: Session = Depends(get_db),
) -> dict:
    parsed_recorded_at = _parse_datetime(recorded_at)
    camera = get_or_create_camera(db, camera_id, location)
    relative_path, _ = save_upload_file(file)

    video = Video(
        camera_id=camera.id,
        original_filename=file.filename or "camera_video.mp4",
        storage_path=relative_path,
        recorded_at=parsed_recorded_at,
        status="uploaded",
    )
    db.add(video)
    db.commit()
    db.refresh(video)

    if process_now:
        process_video_job(video.id)
        db.refresh(video)
    else:
        background_tasks.add_task(process_video_job, video.id)

    video = (
        db.query(Video)
        .options(selectinload(Video.camera), selectinload(Video.clips))
        .filter(Video.id == video.id)
        .first()
    )
    return video_payload(video)


@router.post("/videos/{video_id}/process", response_model=VideoRead)
def process_video(video_id: str, db: Session = Depends(get_db)) -> dict:
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    process_video_job(video_id)
    db.expire_all()
    video = (
        db.query(Video)
        .options(selectinload(Video.camera), selectinload(Video.clips))
        .filter(Video.id == video_id)
        .first()
    )
    return video_payload(video)


@router.get("/clips", response_model=list[ClipRead])
def list_clips(video_id: str | None = None, db: Session = Depends(get_db)) -> list[dict]:
    query = db.query(Clip).options(selectinload(Clip.camera), selectinload(Clip.video))
    if video_id:
        query = query.filter(Clip.video_id == video_id)
    clips = query.order_by(Clip.created_at.desc()).limit(200).all()
    return [clip_payload(clip) for clip in clips]


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    return search_service.search(db, payload)


@router.post("/reports/summary", response_model=SummaryResponse)
def summarize(payload: SearchRequest, db: Session = Depends(get_db)) -> SummaryResponse:
    response = search_service.search(db, payload)
    return SummaryResponse(
        query=payload.query,
        summary=build_summary(payload.query, response.results),
        evidence=response.results,
    )


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="recorded_at must be ISO datetime") from exc

