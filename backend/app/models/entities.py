from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


EMBEDDING_DIM = get_settings().embedding_dim


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True, index=True)
    location: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    stream_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    videos: Mapped[list[Video]] = relationship(back_populates="camera", cascade="all, delete-orphan")
    clips: Mapped[list[Clip]] = relationship(back_populates="camera", cascade="all, delete-orphan")


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="uploaded", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    camera: Mapped[Camera] = relationship(back_populates="videos")
    clips: Mapped[list[Clip]] = relationship(back_populates="video", cascade="all, delete-orphan")


class Clip(Base):
    __tablename__ = "clips"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    start_second: Mapped[float] = mapped_column(Float, nullable=False)
    end_second: Mapped[float] = mapped_column(Float, nullable=False)
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    caption: Mapped[str] = mapped_column(Text, nullable=False)
    objects: Mapped[list[str]] = mapped_column(JSON, default=list)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)
    thumbnail_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    clip_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    video: Mapped[Video] = relationship(back_populates="clips")
    camera: Mapped[Camera] = relationship(back_populates="clips")


class SearchLog(Base):
    __tablename__ = "search_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_query: Mapped[str] = mapped_column(Text, nullable=False)
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

