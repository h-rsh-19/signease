from __future__ import annotations

import base64
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.models import CandidatePrediction

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class RecognitionDiagnostics:
    model_loaded: bool
    backend: str
    model_path: str
    note: str | None = None


class YOLORecognizer:
    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._model: Any | None = None
        self._backend = "none"
        self._note: str | None = None

        try:
            from ultralytics import YOLO  # type: ignore

            if model_path.exists():
                self._model = YOLO(str(model_path))
                self._backend = "ultralytics"
            else:
                self._note = f"Model file not found at {model_path}"
        except Exception as exc:  # pragma: no cover
            self._note = f"Ultralytics unavailable: {exc}"

    @property
    def diagnostics(self) -> RecognitionDiagnostics:
        return RecognitionDiagnostics(
            model_loaded=self._model is not None,
            backend=self._backend,
            model_path=str(self.model_path),
            note=self._note,
        )

    def predict(self, image_np: np.ndarray) -> list[CandidatePrediction]:
        if self._model is None:
            return [
                CandidatePrediction(
                    label="unknown",
                    confidence=0.0,
                    bbox_xyxy=None,
                )
            ]

        results = self._model.predict(source=image_np, conf=self.confidence_threshold, verbose=False)
        if not results:
            return [CandidatePrediction(label="no_detection", confidence=0.0, bbox_xyxy=None)]

        result = results[0]
        names = getattr(result, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return [CandidatePrediction(label="no_detection", confidence=0.0, bbox_xyxy=None)]

        candidates: list[CandidatePrediction] = []
        for idx in range(min(5, len(boxes))):
            cls_id = int(boxes.cls[idx].item())
            conf = float(boxes.conf[idx].item())
            xyxy = boxes.xyxy[idx].tolist()
            label = str(names.get(cls_id, f"class_{cls_id}"))
            candidates.append(
                CandidatePrediction(
                    label=label,
                    confidence=round(conf, 4),
                    bbox_xyxy=[float(v) for v in xyxy],
                )
            )

        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates


class TemporalStabilizer:
    def __init__(self, window: int, commit_threshold: int) -> None:
        self.window = max(3, window)
        self.commit_threshold = max(2, commit_threshold)
        self._buffers: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=self.window))
        self._committed: dict[str, list[str]] = defaultdict(list)

    def update(self, session_id: str, label: str) -> list[str]:
        buf = self._buffers[session_id]
        buf.append(label)
        counts = Counter(buf)
        majority_label, count = counts.most_common(1)[0]

        seq = self._committed[session_id]
        if count >= self.commit_threshold:
            if not seq or seq[-1] != majority_label:
                seq.append(majority_label)
        return list(seq)


class RecognitionService:
    def __init__(
        self,
        *,
        model_path: Path,
        confidence_threshold: float,
        smoothing_window: int,
        commit_threshold: int,
    ) -> None:
        self.recognizer = YOLORecognizer(model_path=model_path, confidence_threshold=confidence_threshold)
        self.stabilizer = TemporalStabilizer(window=smoothing_window, commit_threshold=commit_threshold)

    @staticmethod
    def _predict_number_heuristic(image_np: np.ndarray) -> list[CandidatePrediction]:
        if cv2 is None:
            return [CandidatePrediction(label="unknown", confidence=0.0, bbox_xyxy=None)]

        frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [CandidatePrediction(label="no_hand", confidence=0.0, bbox_xyxy=None)]

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 2500:
            return [CandidatePrediction(label="no_hand", confidence=0.0, bbox_xyxy=None)]

        hull = cv2.convexHull(cnt, returnPoints=False)
        fingers = 0
        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(cnt, hull)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = cnt[s][0]
                    end = cnt[e][0]
                    far = cnt[f][0]
                    a = np.linalg.norm(end - start)
                    b = np.linalg.norm(far - start)
                    c = np.linalg.norm(end - far)
                    if b == 0 or c == 0:
                        continue
                    angle = np.degrees(np.arccos((b * b + c * c - a * a) / (2 * b * c)))
                    if angle <= 90 and d > 7000:
                        fingers += 1

        # Convexity defects usually count gaps between fingers; open fingers ~= defects + 1.
        open_fingers = max(0, min(5, fingers + 1))
        x, y, w, h = cv2.boundingRect(cnt)

        if open_fingers == 0:
            label = "0"
            conf = 0.45
        else:
            label = str(open_fingers)
            conf = min(0.85, 0.45 + 0.08 * open_fingers)

        return [
            CandidatePrediction(
                label=label,
                confidence=round(float(conf), 4),
                bbox_xyxy=[float(x), float(y), float(x + w), float(y + h)],
            )
        ]

    @staticmethod
    def decode_image_base64(payload: str) -> np.ndarray:
        raw = payload.split(",", 1)[1] if payload.startswith("data:") and "," in payload else payload
        image_bytes = base64.b64decode(raw)
        with Image.open(BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")
            return np.array(rgb)

    def infer(self, *, image_base64: str, session_id: str) -> dict[str, object]:
        image_np = self.decode_image_base64(image_base64)
        diag = self.recognizer.diagnostics
        if diag.model_loaded:
            candidates = self.recognizer.predict(image_np)
            backend = diag.backend
            note = diag.note
        else:
            candidates = self._predict_number_heuristic(image_np)
            backend = "heuristic_contour"
            note = diag.note or "Using contour fallback. Add YOLO weights for full ISL classes."

        top = candidates[0]
        committed = self.stabilizer.update(session_id, top.label)

        return {
            "session_id": session_id,
            "top_prediction": top,
            "candidates": candidates,
            "committed_sequence": committed,
            "diagnostics": {
                "model_loaded": diag.model_loaded,
                "backend": backend,
                "model_path": diag.model_path,
                "note": note,
            },
        }
