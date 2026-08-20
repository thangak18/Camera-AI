from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from contextlib import asynccontextmanager
import logging
from src.reranking import get_reranker
from src.vlm.client import probe_models, shutdown

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Probing VLM models...")
        probe_models()
        logger.info("Warming up SigLIP...")
        get_reranker().encode_text("warmup")
        logger.info("Warmup complete.")
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
    yield
    shutdown()

app = FastAPI(title="Camera AI 2026 - API & Submission", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router, prefix="/search")

@app.get("/")
def read_root():
    return {"message": "AIC 2026 Submission Backend is running."}
