from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app.models import (
    RecognitionFrameRequest,
    RecognitionFrameResponse,
    RenderSequenceRequest,
    RenderSequenceResponse,
    TranslationRequest,
    TranslationResponse,
)


def build_router(
    *,
    recognition_service,
    text_service,
    render_service,
    asset_catalog,
    allowed_asset_roots: list[Path],
) -> APIRouter:
    router = APIRouter(prefix="/api")
    allowed_asset_roots = [root.resolve() for root in allowed_asset_roots]
    asset_base_dir = Path(os.path.commonpath([str(root) for root in allowed_asset_roots])).resolve()

    @router.get("/assets/health")
    def assets_health() -> dict[str, object]:
        return {"status": "ok", **asset_catalog.snapshot()}

    @router.get("/assets/file")
    def assets_file(path: str = Query(..., min_length=3)) -> FileResponse:
        path_obj = Path(path)
        if not path_obj.is_absolute():
            resolved = (asset_base_dir / path_obj).resolve()
        else:
            resolved = path_obj.resolve()

        if not resolved.exists() or not resolved.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")

        if not any(root in resolved.parents or resolved == root for root in allowed_asset_roots):
            raise HTTPException(status_code=403, detail="Path outside allowed asset roots")

        return FileResponse(resolved)

    @router.post("/recognition/frame", response_model=RecognitionFrameResponse)
    def recognition_frame(payload: RecognitionFrameRequest) -> dict[str, object]:
        return recognition_service.infer(
            image_base64=payload.image_base64,
            session_id=payload.session_id,
        )

    @router.websocket("/recognition/stream")
    async def recognition_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_json()
                image_base64 = str(message.get("image_base64", ""))
                session_id = str(message.get("session_id", "default"))
                if not image_base64:
                    await websocket.send_json({"error": "image_base64 is required"})
                    continue
                result = recognition_service.infer(
                    image_base64=image_base64,
                    session_id=session_id,
                )
                await websocket.send_json(
                    {
                        "session_id": result["session_id"],
                        "top_prediction": result["top_prediction"].model_dump(),
                        "candidates": [c.model_dump() for c in result["candidates"]],
                        "committed_sequence": result["committed_sequence"],
                        "diagnostics": result["diagnostics"],
                    }
                )
        except WebSocketDisconnect:
            return

    @router.post("/translation/text", response_model=TranslationResponse)
    def translate_text(payload: TranslationRequest) -> dict[str, object]:
        result = text_service.translate(payload.text)
        return {
            "input_text": result.input_text,
            "normalized_text": result.normalized_text,
            "tokens": result.tokens,
            "token_confidence": result.token_confidence,
            "unknown_tokens": result.unknown_tokens,
            "trace": [step.model_dump() for step in result.trace],
        }

    @router.post("/render/sequence", response_model=RenderSequenceResponse)
    def render_sequence(payload: RenderSequenceRequest) -> dict[str, object]:
        result = render_service.render(
            tokens=payload.tokens,
            mode=payload.render_mode,
            token_confidence=payload.token_confidence,
        )
        for artifact in result["artifacts"]:
            if artifact.path:
                try:
                    rel_path = Path(artifact.path).relative_to(asset_base_dir).as_posix()
                    artifact.path = f"/api/assets/file?path={quote(rel_path, safe=':/._-')}"
                except ValueError:
                    artifact.path = f"/api/assets/file?path={quote(artifact.path, safe=':/._-')}"
        return result

    return router
