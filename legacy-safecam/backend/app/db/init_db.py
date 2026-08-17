from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.base import Base
from app.db.session import engine
from app.models.entities import Camera


def init_db() -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)


def seed_default_camera(db: Session) -> None:
    existing = db.query(Camera).first()
    if existing:
        return

    db.add(
        Camera(
            name="CAM_01",
            location="Cong chinh",
            description="Camera demo mac dinh cho upload video dau tien.",
        )
    )
    db.commit()


def ensure_media_dirs() -> None:
    media_root = get_settings().media_path
    for folder in ("videos", "clips", "thumbnails", "reports"):
        (media_root / folder).mkdir(parents=True, exist_ok=True)

