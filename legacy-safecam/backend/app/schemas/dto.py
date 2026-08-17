from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class HealthRead(BaseModel):
    status: str
    app: str
    database: str


class CameraCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    location: str = Field(min_length=1, max_length=255)
    stream_url: str | None = None
    description: str | None = None


class CameraRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    location: str
    stream_url: str | None = None
    description: str | None = None
    created_at: datetime


class VideoRead(BaseModel):
    id: str
    camera_id: str
    camera_name: str | None = None
    location: str | None = None
    original_filename: str
    status: str
    duration_seconds: float | None = None
    recorded_at: datetime | None = None
    created_at: datetime
    processed_at: datetime | None = None
    error_message: str | None = None
    video_url: str | None = None
    clip_count: int = 0


class ClipRead(BaseModel):
    id: str
    video_id: str
    camera_id: str
    camera_name: str | None = None
    location: str | None = None
    start_second: float
    end_second: float
    start_time: datetime | None = None
    end_time: datetime | None = None
    caption: str
    objects: list[str] = Field(default_factory=list)
    attributes: dict = Field(default_factory=dict)
    confidence: float
    thumbnail_url: str | None = None
    clip_url: str | None = None
    source_video_url: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=8, ge=1, le=30)


class SearchResult(ClipRead):
    score: float
    reason: str


class SearchResponse(BaseModel):
    query: str
    answer: str
    results: list[SearchResult]
    latency_ms: int


class SummaryResponse(BaseModel):
    query: str
    summary: str
    evidence: list[SearchResult]

