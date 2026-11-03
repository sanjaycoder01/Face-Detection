# Face Detection Service (Phase 1)

Event photo platform **Phase 1**: detect faces in uploaded images with SCRFD (via InsightFace), draw bounding boxes, and return metadata plus an annotated JPEG. No face recognition, embeddings, database, or auth.

## Project layout

| Path | Responsibility |
|------|----------------|
| `app/main.py` | FastAPI app, startup model load, `/health`, global error handling |
| `app/routes/detection.py` | `POST /api/v1/faces/detect`, `GET /api/v1/faces/model-info` |
| `app/services/face_detector.py` | InsightFace SCRFD wrapper, singleton load, `detect_faces()` |
| `app/schemas/detection_schema.py` | Pydantic response/error models |
| `app/utils/image_utils.py` | Validate/decode uploads, draw boxes, encode JPEG |
| `uploads/` | Optional scratch (API does not persist uploads by default) |
| `outputs/` | Optional local output folder for manual experiments |
| `tests/` | Pytest + mocked detector |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |

## Architecture and request flow

```mermaid
flowchart LR
  Client -->|multipart image| API[FastAPI route]
  API --> Utils[image_utils]
  Utils --> Detector[face_detector]
  Detector -->|SCRFD| InsightFace[InsightFace CPU]
  Detector --> Utils
  Utils -->|annotate + JPEG| API
  API -->|multipart/mixed JSON + image| Client
```

1. Client uploads JPEG/PNG to `POST /api/v1/faces/detect`.
2. Route validates size/format and decodes with OpenCV.
3. `FaceDetectorService.detect_faces()` runs SCRFD once per request (model weights stay loaded).
4. Utils draw green boxes, IDs, and confidence labels.
5. Response is **multipart/mixed**: part 1 = JSON metadata, part 2 = annotated JPEG.

**Why multipart/mixed?** One round trip returns both structured data and the visual result. Swagger still documents the JSON shape; use curl or a small script to split parts and save the image. A separate image-only endpoint can be added in Phase 2 if clients prefer pure JSON.

## SCRFD (high level)

SCRFD is a single-stage face detector that predicts face boxes (and optional landmarks) at multiple feature scales. It is designed for speed and accuracy on crowded scenes—well suited to group event photos. InsightFace ships SCRFD weights inside model packs such as `buffalo_l`.

## InsightFace model loading

On startup, the app creates `FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"])`, then calls `prepare(ctx_id=-1, det_size=(640, 640))`. InsightFace downloads ONNX weights into `~/.insightface/models/` on first run. Only the detection graph runs; recognition/ArcFace modules are not loaded.

**Licensing:** Review [InsightFace](https://github.com/deepinsight/insightface) and the buffalo model license before commercial deployment.

## Setup

```bash
cd /Users/sanjay/Documents/Face_Dectection
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional
```

First startup downloads models (network required, may take a few minutes).

## Run the server

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health: `GET /health`
- Model info: `GET /api/v1/faces/model-info`

## API examples

**Detect faces (curl):**

```bash
curl -s -D /tmp/headers.txt -o /tmp/body.bin \
  -F "image=@/path/to/group_photo.jpg" \
  http://127.0.0.1:8000/api/v1/faces/detect
```

Inspect headers for `Content-Type: multipart/mixed; boundary=...`, then split parts with the boundary string to extract JSON and JPEG.

**Example JSON metadata:**

```json
{
  "success": true,
  "total_faces": 3,
  "faces": [
    {
      "face_id": 1,
      "bounding_box": { "x1": 100, "y1": 80, "x2": 200, "y2": 220 },
      "confidence": 0.98
    }
  ],
  "message": "Face detection completed"
}
```

**Bounding boxes:** Coordinates are in **pixel space** of the decoded image: `(x1, y1)` top-left, `(x2, y2)` bottom-right, origin at the top-left corner. `face_id` is assigned left-to-right in detector output order (1-based).

## Manual test with a real group photo

1. Start the server (see above).
2. Open `/docs` → **POST /api/v1/faces/detect** → Try it out → choose a group JPEG/PNG (≤ 10 MB).
3. Execute; download the response body and parse multipart, or use curl as above and save the JPEG part to `outputs/annotated.jpg`.
4. Confirm `total_faces` matches visible people and boxes align with faces.

## Tests

```bash
source .venv/bin/activate
pytest -q
```

Unit tests mock the detector so CI does not need model weights.

## Common issues

| Symptom | Likely cause |
|---------|----------------|
| `503` model not loaded | First download failed or offline; check logs and `~/.insightface/models/` |
| Slow first request | One-time ONNX/model init; subsequent requests reuse loaded weights |
| `413` | File larger than `MAX_UPLOAD_BYTES` (default 10 MB) |
| `415` | Not JPEG/PNG or wrong extension |
| Import errors for `onnxruntime` | Reinstall deps inside the venv; use Python 3.11+ |

## Request lifecycle (summary)

Upload → validate filename/size → decode BGR → SCRFD inference → build face list → draw annotations → encode JPEG → return multipart/mixed (JSON + image). Images with zero faces still return **HTTP 200** with `total_faces: 0`.
