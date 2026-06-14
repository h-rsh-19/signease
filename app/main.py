from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import build_router
from app.config import Trial3Config
from app.services.asset_catalog import build_catalog
from app.services.recognition import RecognitionService
from app.services.renderer import RenderService
from app.services.text_pipeline import TextPipelineService

BASE_DIR = Path(__file__).resolve().parents[1]
config = Trial3Config.from_env(BASE_DIR)
asset_catalog = build_catalog(
    human_media_dir=config.human_media_dir,
    animated_media_dir=config.animated_media_dir,
    sigml_roots=[config.sigml_root_dir],
    manifest_dir=config.manifest_dir,
)

recognition_service = RecognitionService(
    model_path=config.model_path,
    confidence_threshold=config.confidence_threshold,
    smoothing_window=config.smoothing_window,
    commit_threshold=config.commit_threshold,
)
text_service = TextPipelineService(asset_catalog)
render_service = RenderService(asset_catalog)

app = FastAPI(title="SignEase Trial3", version="1.0.0")
app.mount("/static", StaticFiles(directory=config.static_dir), name="static")

# Mount the Avatar engine from the local static/jas folder
jas_loc2021 = config.static_dir / "jas" / "loc2021"
if jas_loc2021.exists():
    app.mount("/jas/loc2021", StaticFiles(directory=jas_loc2021), name="jas-loc2021")

app.include_router(
    build_router(
        recognition_service=recognition_service,
        text_service=text_service,
        render_service=render_service,
        asset_catalog=asset_catalog,
        allowed_asset_roots=[
            config.human_media_dir.resolve(),
            config.animated_media_dir.resolve(),
            config.sigml_root_dir.resolve(),
        ],
    )
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.static_dir / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8010, reload=True)
