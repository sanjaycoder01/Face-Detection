# Face Detection Service (Phase 1)

Event photo platform **Phase 1**: detect faces in uploaded images using [CompreFace](https://github.com/exadel-inc/CompreFace), draw bounding boxes, and return metadata plus an annotated JPEG. Face matching, history, media URLs, and embedding dumps are also implemented. CompreFace runs separately for inference; this FastAPI app is the only API the frontend should call.

**Full technical documentation for developers:** [`docs/TECHNICAL.md`](docs/TECHNICAL.md)

## Project layout

| Path | Responsibility |
|------|----------------|
| `app/main.py` | FastAPI app, CompreFace config at startup, `/health`, global error handling |
| `app/routes/detection.py` | `POST /api/v1/faces/detect`, `GET /api/v1/faces/model-info` |
| `app/routes/match.py` | `POST /api/v1/faces/match` (reference photo → find same person in target) |
| `app/routes/history.py` | `GET /api/v1/faces/history` |
| `app/routes/media.py` | `GET /api/v1/media/outputs/{filename}` (safe image serving) |
| `app/services/face_matcher.py` | Temporary CompreFace subject enroll + recognize + cleanup |
| `app/services/face_detector.py` | CompreFace-backed detector singleton |
| `app/services/compreface_client.py` | HTTP client for CompreFace detect/recognize |
| `app/services/history_store.py` | Local JSON history for match operations |
| `app/schemas/detection_schema.py` | Pydantic response/error models |
| `app/utils/image_utils.py` | Validate/decode uploads, draw boxes, encode JPEG |
| `docker-compose.compreface.yml` | Local CompreFace stack (UI on port **8080**) |
| `uploads/` | Optional scratch |
| `outputs/` | Annotated images saved from the API |
| `data/history.json` | Match history (created on first match; not invented) |
| `tests/` | Pytest with mocked detector |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |

## Architecture

```mermaid
flowchart LR
  Client -->|multipart image| API[FastAPI :8000]
  API --> Utils[image_utils]
  Utils --> Detector[face_detector]
  Detector -->|HTTP + x-api-key| CF[CompreFace :8080]
  CF --> Core[compreface-core ML]
  Detector --> Utils
  Utils -->|annotate + JPEG| API
  API -->|multipart/mixed JSON + image| Client
```

1. Client uploads JPEG/PNG to `POST /api/v1/faces/detect`.
2. Route validates size/format and decodes with OpenCV.
3. `FaceDetectorService` encodes the image and calls CompreFace detection API.
4. CompreFace returns boxes as `x_min`, `y_min`, `x_max`, `y_max`, and `probability`.
5. This service maps them to your API schema, draws annotations, and returns **multipart/mixed** (JSON + JPEG).

Your FastAPI app stays on port **8000**. CompreFace UI/API gateway uses **8080** in the bundled compose file to avoid a port clash.

## CompreFace setup

### 1. Start CompreFace (Docker)

Requires [Docker](https://docs.docker.com/get-docker/) and Docker Compose.

```bash
cd /Users/sanjay/Documents/Face_Dectection
docker compose -f docker-compose.compreface.yml up -d
```

Wait until containers are healthy (first pull can take several minutes). Open [http://localhost:8080](http://localhost:8080).

### 2. Create a Face Detection service and API key

1. Sign up / log in in the CompreFace UI.
2. Create an application (if prompted).
3. Add a **Face detection** service (not recognition).
4. Copy the service **API key**.

CompreFace documents the detection endpoint as:

`POST /api/v1/detection/detect` with header `x-api-key` and multipart field `file`.

See [Rest API description](https://github.com/exadel-inc/CompreFace/blob/master/docs/Rest-API-description.md#face-detection-service).

### 3. Configure this app

```bash
cp .env.example .env
# Set COMPREFACE_API_KEY=your-key-here
```

## Python setup

```bash
cd /Users/sanjay/Documents/Face_Dectection
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set COMPREFACE_API_KEY and COMPREFACE_SERVICE_TYPE
```

The app loads `.env` automatically on startup (`python-dotenv`).

## Run the server

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health: `GET /health`
- Model info: `GET /api/v1/faces/model-info`

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CORS_ALLOWED_ORIGINS` | localhost + Lovable preview URL | Comma-separated browser origins |
| `COMPREFACE_URL` | `http://localhost:8080` | CompreFace UI/gateway base URL |
| `COMPREFACE_API_KEY` | (required) | API key from the CompreFace service (Dashboard) |
| `COMPREFACE_SERVICE_TYPE` | `detection` | `detection` or `recognition` (`recognition` required for match) |
| `COMPREFACE_DET_PROB_THRESHOLD` | (optional) | Min face confidence 0.0–1.0 |
| `COMPREFACE_FACE_LIMIT` | `0` | Max faces (0 = no limit) |
| `COMPREFACE_MAX_DETECT_SIZE` | `640` | Shown in model-info; core uses `IMG_LENGTH_LIMIT` in compose |
| `COMPREFACE_TIMEOUT_SECONDS` | `60` | HTTP timeout to CompreFace |
| `MAX_UPLOAD_BYTES` | `10485760` | Max upload size for this API |
| `OUTPUT_DIR` | `outputs` | Annotated image directory |
| `HISTORY_FILE` | `data/history.json` | Match history JSON |

CompreFace itself limits uploads (default **5 MB** per image in their stack).

## Frontend / Lovable integration

- Frontend must call **this FastAPI app only** (never CompreFace directly).
- Set frontend `VITE_API_BASE_URL=http://127.0.0.1:8000` for local development.
- Hosted Lovable preview **cannot** reach your Mac `localhost`. CORS fixes browser policy only. Use a tunnel (ngrok/cloudflared) or a deployed HTTPS API URL for hosted preview.
- Configure `CORS_ALLOWED_ORIGINS` with your Lovable preview origin.

### Response contract change (detect + match)

**Default is now `application/json`** so the React app can use `response.json()`.

| Field | Meaning |
|-------|---------|
| `matches` / `faces` | Results with bounding boxes and scores |
| `annotated_image_url` | Relative URL, e.g. `/api/v1/media/outputs/Fullimage_matched_abcd1234.jpg` |
| `operation_id` | History id (match only) |
| `person_name` | Optional name from form field `name` (match only) |

Display images as: `` `${VITE_API_BASE_URL}${annotated_image_url}` ``

Legacy **multipart/mixed** (JSON part + JPEG bytes) is still available:

`?response_format=multipart`

## Face matching (reference → event photo)

Requires **`COMPREFACE_SERVICE_TYPE=recognition`** and a Face **Recognition** service API key.

1. Upload a **reference** image (one person — a selfie or a crop from the event photo).
2. Upload the **target** group/event image.
3. Optional form field **`name`** (person name for history).
4. The API enrolls the reference face temporarily in CompreFace, runs recognition on the target, returns matches with **similarity** scores, saves an annotated JPEG, and appends a history record.

If the reference contains multiple faces, the **largest face** is used automatically.

**Swagger:** `POST /api/v1/faces/match` — fields `reference`, `target`, optional `name`.

**curl (JSON, frontend-friendly):**

```bash
curl -s \
  -F "reference=@/path/to/person.jpg" \
  -F "target=@/path/to/event_group.jpg" \
  -F "name=Alex" \
  http://127.0.0.1:8000/api/v1/faces/match
```

Defaults: **`COMPREFACE_MATCH_SIMILARITY_THRESHOLD=0.75`** and **`COMPREFACE_MATCH_MAX_RESULTS=1`**. Optional query: `?max_matches=3`.

## History and media

```bash
curl -s http://127.0.0.1:8000/api/v1/faces/history
curl -s -o /tmp/annotated.jpg http://127.0.0.1:8000/api/v1/media/outputs/FILENAME.jpg
curl -s http://127.0.0.1:8000/health
```

History starts empty and only grows when match operations succeed. Existing `outputs/*.jpg` files are not turned into fake history rows.

## API examples

**Detect faces (curl):**

```bash
curl -s \
  -F "image=@/path/to/group_photo.jpg" \
  http://127.0.0.1:8000/api/v1/faces/detect
```

**Example detect JSON:**

```json
{
  "success": true,
  "total_faces": 3,
  "faces": [
    {
      "face_id": 1,
      "bounding_box": { "x1": 68, "y1": 77, "x2": 376, "y2": 479 },
      "confidence": 0.9988
    }
  ],
  "message": "Face detection completed",
  "annotated_image_url": "/api/v1/media/outputs/group_detected_abcd1234.jpg"
}
```

Bounding boxes use pixel coordinates: `(x1, y1)` top-left, `(x2, y2)` bottom-right. `face_id` is assigned left-to-right by `x_min`.

## Tests

```bash
source .venv/bin/activate
pytest -q
```

Tests mock the detector so CI does not need Docker or CompreFace.

## Common issues

| Symptom | Likely cause |
|---------|----------------|
| `503` model not loaded | Missing `COMPREFACE_API_KEY` in `.env` |
| `503` / invalid API key | Wrong key or recognition key used instead of detection |
| Connection errors | CompreFace not running; run `docker compose -f docker-compose.compreface.yml ps` |
| CompreFace `413` / size errors | Image over CompreFace limit (~5 MB); resize or raise limits in their compose |
| Port conflict on 8080 | Change CompreFace mapping in `docker-compose.compreface.yml` and set `COMPREFACE_URL` |

## Licensing

CompreFace is open source (Apache 2.0). Review [CompreFace](https://github.com/exadel-inc/CompreFace) and embedded model licenses before commercial use.
