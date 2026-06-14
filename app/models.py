from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RenderMode = Literal["human_video", "animated_video", "sigml_avatar"]


class RecognitionFrameRequest(BaseModel):
    image_base64: str = Field(..., min_length=8)
    session_id: str = Field(default="default", min_length=1, max_length=128)


class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class RenderSequenceRequest(BaseModel):
    tokens: list[str] = Field(default_factory=list)
    token_confidence: list[dict[str, object]] = Field(default_factory=list)
    render_mode: RenderMode


class CandidatePrediction(BaseModel):
    label: str
    confidence: float
    bbox_xyxy: list[float] | None = None


class RecognitionFrameResponse(BaseModel):
    session_id: str
    top_prediction: CandidatePrediction
    candidates: list[CandidatePrediction]
    committed_sequence: list[str]
    diagnostics: dict[str, object] = Field(default_factory=dict)


class PipelineTraceStep(BaseModel):
    stage: str
    input: object
    output: object


class TranslationResponse(BaseModel):
    input_text: str
    normalized_text: str
    tokens: list[str]
    token_confidence: list[dict[str, object]] = Field(default_factory=list)
    unknown_tokens: list[str]
    trace: list[PipelineTraceStep]


class RenderArtifact(BaseModel):
    token: str
    mode: RenderMode
    kind: Literal["video", "sigml", "missing"]
    path: str | None = None
    source: str | None = None


class RenderSequenceResponse(BaseModel):
    render_mode: RenderMode
    artifacts: list[RenderArtifact]
    missing_tokens: list[str]
    diagnostics: dict[str, object] = Field(default_factory=dict)
