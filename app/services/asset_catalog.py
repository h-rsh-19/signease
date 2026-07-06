from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


def _norm_token(token: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_")
    return cleaned


@dataclass
class AssetCatalog:
    human_videos: dict[str, str]
    animated_videos: dict[str, str]
    sigml_files: dict[str, str]

    def token_exists_any(self, token: str) -> bool:
        key = _norm_token(token)
        return key in self.human_videos or key in self.animated_videos or key in self.sigml_files

    def snapshot(self) -> dict[str, object]:
        return {
            "human_video_count": len(self.human_videos),
            "animated_video_count": len(self.animated_videos),
            "sigml_count": len(self.sigml_files),
            "human_ready": bool(self.human_videos),
            "animated_ready": bool(self.animated_videos),
            "sigml_ready": bool(self.sigml_files),
        }


def _scan_videos(directory: Path) -> dict[str, str]:
    if not directory.exists():
        return {}
    result: dict[str, str] = {}
    for path in sorted(directory.glob("*.mp4")):
        key = _norm_token(path.stem)
        result[key] = path.resolve().as_posix()
    return result


def _scan_sigml(roots: list[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.sigml")):
            key = _norm_token(path.stem)
            # Project-local files take priority if there's a name collision
            result[key] = path.resolve().as_posix()
    return result


def _manifest_payload(payload: dict[str, str], base_dir: Path) -> dict[str, str]:
    portable: dict[str, str] = {}
    for key, value in payload.items():
        value_path = Path(value)
        try:
            portable[key] = value_path.resolve().relative_to(base_dir).as_posix()
        except ValueError:
            portable[key] = value
    return portable


def _dump_manifest(path: Path, payload: dict[str, str], base_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_manifest_payload(payload, base_dir), indent=2), encoding="utf-8")


def build_catalog(
    *,
    human_media_dir: Path,
    animated_media_dir: Path,
    sigml_roots: list[Path],
    manifest_dir: Path,
) -> AssetCatalog:
    base_dir = manifest_dir.resolve().parents[1]
    human_videos = _scan_videos(human_media_dir)
    animated_videos = _scan_videos(animated_media_dir)
    sigml_files = _scan_sigml(sigml_roots)

    _dump_manifest(manifest_dir / "human_video_manifest.json", human_videos, base_dir)
    _dump_manifest(manifest_dir / "animated_video_manifest.json", animated_videos, base_dir)
    _dump_manifest(manifest_dir / "sigml_manifest.json", sigml_files, base_dir)

    return AssetCatalog(
        human_videos=human_videos,
        animated_videos=animated_videos,
        sigml_files=sigml_files,
    )


def norm_token(token: str) -> str:
    return _norm_token(token)
