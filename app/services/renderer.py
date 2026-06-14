from __future__ import annotations

from app.models import RenderArtifact, RenderMode
from app.services.asset_catalog import AssetCatalog, norm_token


class RenderService:
    def __init__(self, catalog: AssetCatalog) -> None:
        self.catalog = catalog

    def _lookup(self, mode: RenderMode, token: str) -> str | None:
        key = norm_token(token)
        if mode == "human_video":
            return self.catalog.human_videos.get(key)
        if mode == "animated_video":
            return self.catalog.animated_videos.get(key)
        return self.catalog.sigml_files.get(key)

    def _letter_fallback(self, mode: RenderMode, token: str) -> list[RenderArtifact]:
        artifacts: list[RenderArtifact] = []
        if len(token) <= 1:
            return artifacts
        for ch in token:
            mapped = self._lookup(mode, ch)
            if mapped:
                artifacts.append(
                    RenderArtifact(
                        token=ch,
                        mode=mode,
                        kind="sigml" if mode == "sigml_avatar" else "video",
                        path=mapped,
                        source="letter_fallback",
                    )
                )
        return artifacts

    def render(
        self,
        *,
        tokens: list[str],
        mode: RenderMode,
        token_confidence: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        artifacts: list[RenderArtifact] = []
        missing: list[str] = []
        token_confidence = token_confidence or []

        def _resolve_token(tok: str):
            mapped = self._lookup(mode, tok)
            if mapped:
                artifacts.append(
                    RenderArtifact(
                        token=tok,
                        mode=mode,
                        kind="sigml" if mode == "sigml_avatar" else "video",
                        path=mapped,
                        source="direct",
                    )
                )
                return

            if "_" in tok:
                for sub in tok.split("_"):
                    _resolve_token(sub)
                return

            fallback_items = self._letter_fallback(mode, tok)
            if fallback_items:
                artifacts.extend(fallback_items)
                return

            artifacts.append(
                RenderArtifact(
                    token=tok,
                    mode=mode,
                    kind="missing",
                    path=None,
                    source="missing",
                )
            )
            missing.append(tok)

        for token in tokens:
            _resolve_token(token)

        conf_vals = [float(item.get("confidence", 0.0)) for item in token_confidence]
        by_source_count: dict[str, int] = {}
        for item in token_confidence:
            src = str(item.get("source", "unknown"))
            by_source_count[src] = by_source_count.get(src, 0) + 1

        return {
            "render_mode": mode,
            "artifacts": artifacts,
            "missing_tokens": sorted(set(missing)),
            "diagnostics": {
                "requested_tokens": len(tokens),
                "resolved_items": len([a for a in artifacts if a.kind != "missing"]),
                "missing_items": len(missing),
                "mean_confidence": round(sum(conf_vals) / len(conf_vals), 3) if conf_vals else None,
                "min_confidence": round(min(conf_vals), 3) if conf_vals else None,
                "by_source_count": by_source_count,
            },
        }
