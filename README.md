# trial3 - SignEase Multimodal (Standalone)

This is a standalone project for:
1. Camera ISL recognition (YOLOv5/Ultralytics-compatible path)
2. Text -> ISL pipeline with 3 render modes:
   - human_video
   - animated_video
   - sigml_avatar

## Fixed Data Sources
- Animated videos: `E:\archive\INDIAN SIGN LANGUAGE ANIMATED VIDEOS_`
- SiGML corpus: `E:\text_to_isl-main\text_to_isl-main\static`
- Human videos fallback: `phase2_code/data/media/raw`

## Setup
```bash
cd E:\projects\MAJOR Project\SignEase3\phase2_code\trial3
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
uvicorn app.main:app --reload --port 8010
```
Open `http://127.0.0.1:8010`

## Optional YOLO model
Place model at:
`trial3/models/yolov5/best.pt`

Or set env var:
`TRIAL3_YOLO_MODEL_PATH=...`

If model/package is not available, recognition endpoints still run and report diagnostics with `model_loaded=false`.

## API
- `GET /api/assets/health`
- `GET /api/assets/file?path=...`
- `POST /api/recognition/frame`
- `WS /api/recognition/stream`
- `POST /api/translation/text`
- `POST /api/render/sequence`

## Notes
- Manifests are generated at startup in `trial3/data/manifests`.
- Build-first approach applied: full automated test suite can be added after feature stabilization.
