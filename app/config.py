from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Trial3Config:
    base_dir: Path
    static_dir: Path
    data_dir: Path
    manifest_dir: Path
    model_path: Path
    human_media_dir: Path
    animated_media_dir: Path
    sigml_root_dir: Path
    confidence_threshold: float
    smoothing_window: int
    commit_threshold: int

    @staticmethod
    def from_env(base_dir: Path) -> "Trial3Config":
        model_path = Path(
            os.getenv(
                "TRIAL3_YOLO_MODEL_PATH",
                str(base_dir / "models" / "yolov5" / "best.pt"),
            )
        )
        return Trial3Config(
            base_dir=base_dir,
            static_dir=base_dir / "static",
            data_dir=base_dir / "data",
            manifest_dir=base_dir / "data" / "manifests",
            model_path=model_path,
            human_media_dir=base_dir / "data" / "human_videos",
            animated_media_dir=base_dir / "data" / "animated_videos",
            sigml_root_dir=base_dir / "data" / "sigml",
            confidence_threshold=float(os.getenv("TRIAL3_CONFIDENCE_THRESHOLD", "0.35")),
            smoothing_window=int(os.getenv("TRIAL3_SMOOTHING_WINDOW", "10")),
            commit_threshold=int(os.getenv("TRIAL3_COMMIT_THRESHOLD", "4")),
        )
