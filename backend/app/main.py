from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings
from app.db.init_db import ensure_media_dirs, init_db, seed_default_camera
from app.db.session import SessionLocal


settings = get_settings()

app = FastAPI(title=settings.app_name, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    ensure_media_dirs()
    init_db()
    with SessionLocal() as db:
        seed_default_camera(db)


app.mount("/media", StaticFiles(directory=settings.media_path, check_dir=False), name="media")
app.include_router(router, prefix="/api")
