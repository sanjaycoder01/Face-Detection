# Face Detection & Matching Service — Technical Documentation

**Audience:** backend and frontend developers joining the project  
**Stack:** Python 3.11+, FastAPI, CompreFace (Docker), OpenCV  
**Frontend:** React app (Lovable) — talks only to this FastAPI API, never to CompreFace directly  
**API base (local):** `http://127.0.0.1:8000`  
**CompreFace (local):** `http://localhost:8080`

---

## 1. Purpose

This service is Phase 1+ of an **event photo distribution** platform:

1. **Detect** faces in a group / event photograph.
2. **Match** a reference selfie (person) against faces in a group photograph.
3. Return **JSON** results plus **URLs** to annotated images for a web UI.
4. Keep a simple **local history** of match operations for demos.

Face ML inference runs in **CompreFace**. This FastAPI app is an orchestration / API layer: validation, annotation, storage, CORS, and frontend-friendly contracts.

---

## 2. High-level architecture

```text
┌─────────────────────┐         ┌──────────────────────────┐
│  Lovable / React UI │  HTTP   │  FastAPI (:8000)         │
│  VITE_API_BASE_URL  │ ──────► │  routes + services        │
└─────────────────────┘         └────────────┬─────────────┘
                                             │ HTTP + x-api-key
                                             ▼
                                   ┌──────────────────────┐
                                   │  CompreFace (:8080)  │
                                   │  detect / recognize  │
                                   │  embeddings (calc.)  │
                                   └──────────────────────┘
```

| Component | Responsibility |
|-----------|----------------|
| FastAPI app | Auth-less HTTP API, CORS, image I/O, history, media URLs |
| CompreFace | Face detection, recognition, embedding vectors |
| `outputs/` | Annotated JPEGs served to the frontend |
| `outputs/embeddings/` | JSON dumps of embedding vectors (debug / inspection) |
| `data/history.json` | Match operation history for the UI |

**Important:** Hosted Lovable preview (`https://*.lovable.app`) cannot reach a developer’s Mac `localhost`. CORS alone does not fix that; use a tunnel or a deployed API URL for hosted preview.

---

## 3. Repository layout

```text
Face_Dectection/
├── app/
│   ├── main.py                 # FastAPI app, CORS, lifespan, /health
│   ├── routes/
│   │   ├── detection.py        # POST /api/v1/faces/detect
│   │   ├── match.py            # POST /api/v1/faces/match
│   │   ├── history.py          # GET  /api/v1/faces/history
│   │   └── media.py            # GET  /api/v1/media/outputs/{filename}
│   ├── services/
│   │   ├── compreface_client.py  # HTTP client to CompreFace
│   │   ├── face_detector.py      # Detection singleton
│   │   ├── face_matcher.py       # Enroll → recognize → cleanup
│   │   ├── embedding_store.py    # Write embedding JSON files
│   │   └── history_store.py      # Append/list match history
│   ├── schemas/                # Pydantic response models
│   └── utils/
│       ├── image_utils.py      # Decode, annotate, encode, save JPEG
│       └── media_urls.py       # Safe output path + public URL helper
├── tests/                      # Pytest (CompreFace mocked)
├── uploads/                    # Sample / manual test images
├── outputs/                    # Annotated images
│   └── embeddings/             # target_photo / group_photo embedding dumps
├── data/                       # history.json (created at runtime)
├── docker-compose.compreface.yml
├── .env.example
├── requirements.txt
├── README.md
└── docs/TECHNICAL.md           # This document
```

---

## 4. Runtime dependencies

### 4.1 Python packages (`requirements.txt`)

- `fastapi`, `uvicorn` — API server  
- `httpx` — CompreFace HTTP client  
- `opencv-python-headless`, `numpy` — image decode / draw / encode  
- `python-multipart` — file uploads  
- `python-dotenv` — `.env` loading  
- `pytest` — tests  

ML models are **not** loaded in this process; CompreFace owns them.

### 4.2 CompreFace (Docker)

Start with:

```bash
docker compose -f docker-compose.compreface.yml up -d
```

Then in the CompreFace UI (`http://localhost:8080`):

1. Create an application.
2. Create a **Face Recognition** service (required for `/match`).
3. Copy the service **API key** into `.env` as `COMPREFACE_API_KEY`.
4. Set `COMPREFACE_SERVICE_TYPE=recognition`.

Detection-only keys work for `/detect` when `COMPREFACE_SERVICE_TYPE=detection`, but **matching requires recognition**.

---

## 5. Configuration (`.env`)

Copy `.env.example` → `.env`. Key variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `COMPREFACE_URL` | `http://localhost:8080` | CompreFace gateway |
| `COMPREFACE_API_KEY` | _(required)_ | Service API key |
| `COMPREFACE_SERVICE_TYPE` | `detection` | `detection` or `recognition` |
| `COMPREFACE_MATCH_SIMILARITY_THRESHOLD` | `0.75` | Min similarity (0–1) to count as match |
| `COMPREFACE_MATCH_MAX_RESULTS` | `1` | Max matches returned/drawn (0 = all above threshold) |
| `CORS_ALLOWED_ORIGINS` | localhost + Lovable URLs | Comma-separated browser origins |
| `MAX_UPLOAD_BYTES` | `10485760` | Max upload size for this API (10 MB) |
| `OUTPUT_DIR` | `outputs` | Annotated JPEG directory |
| `HISTORY_FILE` | `data/history.json` | Match history store |
| `EMBEDDINGS_DIR` | `outputs/embeddings` | Embedding JSON dumps |
| `LOG_FACE_EMBEDDINGS` | `true` | Print full vectors to uvicorn console |

CompreFace itself often enforces a ~**5 MB** image limit independently of `MAX_UPLOAD_BYTES`.

---

## 6. Application startup

On startup (`app/main.py` lifespan):

1. Load `.env`.
2. Build `DetectorConfig` from env.
3. `FaceDetectorService.load()` — validates API key; pings CompreFace (non-fatal if down).
4. `FaceMatcherService.load()` — only enables matching when `SERVICE_TYPE=recognition` and API key is set.
5. Register singletons via `set_detector` / `set_matcher`.

On shutdown, HTTP clients are closed.

---

## 7. API reference

Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 7.1 `GET /health`

Returns API + CompreFace status for the frontend “Backend online” indicator.

Example:

```json
{
  "status": "healthy",
  "service": "face-recognition-api",
  "api": "running",
  "compreface": "connected",
  "model_loaded": true,
  "face_matching_enabled": true
}
```

`compreface` values: `connected` | `unreachable` | `not_configured`.

---

### 7.2 `POST /api/v1/faces/detect`

Detect all faces in one image.

**Request:** `multipart/form-data`

| Field | Required | Notes |
|-------|----------|-------|
| `image` | yes | JPEG / PNG |

**Query:**

| Param | Default | Notes |
|-------|---------|-------|
| `response_format` | `json` | `json` or `multipart` (legacy) |

**Default response:** `application/json`

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
  "annotated_image_url": "/api/v1/media/outputs/group_detected_abcd1234.jpg",
  "output_image_path": "/abs/path/outputs/...."
}
```

**Flow:**

1. Validate filename / size / magic bytes.
2. Decode with OpenCV (BGR).
3. Call CompreFace detect (with `face_plugins=calculator`).
4. Draw green boxes + labels.
5. Save JPEG under `outputs/`.
6. Return JSON with relative `annotated_image_url`.

---

### 7.3 `POST /api/v1/faces/match`

Find the reference person inside a group photograph.

**Request:** `multipart/form-data`

| Role | Preferred field | Also accepted (Lovable aliases) |
|------|-----------------|----------------------------------|
| Person / selfie | `reference` | `reference_image`, `referenceImage`, `selfie`, `person`, `personImage`, … |
| Group photo | `target` | `group_image`, `groupImage`, `group`, `groupPhoto`, … |
| Display name | `name` | `personName`, `person_name`, `targetName`, … |

Current Lovable UI uses: `reference_image`, `group_image`, `name`.

**Query:**

| Param | Default | Notes |
|-------|---------|-------|
| `max_matches` | env default | Cap returned/drawn matches |
| `response_format` | `json` | `json` or `multipart` |

**Default response:** `application/json` (frontend-friendly; use `response.json()`).

```json
{
  "success": true,
  "operation_id": "uuid",
  "person_name": "Sanjay",
  "match_count": 1,
  "total_faces_in_target": 26,
  "similarity_threshold": 0.75,
  "max_matches": 1,
  "candidates_above_threshold": 3,
  "matches": [
    {
      "face_id": 1,
      "bounding_box": { "x1": 5442, "y1": 3503, "x2": 6337, "y2": 4557 },
      "detection_confidence": 0.9999,
      "similarity": 0.9999,
      "is_match": true
    }
  ],
  "all_candidates": [ "... all faces ≥ threshold ..." ],
  "message": "Best match for Sanjay: 100% similar (face #1); ...",
  "annotated_image_url": "/api/v1/media/outputs/Fullimage_matched_xxxx.jpg",
  "output_image_path": "..."
}
```

**Match pipeline (`FaceMatcherService`):**

1. Detect faces on the reference image; if multiple, **crop the largest** face (CompreFace rejects multi-face enrollment).
2. Request embeddings for the **target/person photo** → write `target_photo_embeddings.json`.
3. Temporarily enroll cropped face as subject `match-<uuid>`.
4. Run CompreFace **recognize** on the group photo (`face_plugins=calculator`).
5. Write **group** embeddings → `group_photo_embeddings.json`.
6. Keep faces whose similarity to the temp subject ≥ `COMPREFACE_MATCH_SIMILARITY_THRESHOLD`.
7. Apply `max_matches` (usually top 1).
8. Delete temporary subject.
9. Draw orange “BEST MATCH” annotations; save JPEG; append history; return JSON.

**Similarity threshold:** float 0–1. Example: `0.75` means only faces ≥ 75% similar are candidates.

---

### 7.4 `GET /api/v1/faces/history`

Returns newest-first match operations from `data/history.json`.

- Starts **empty**.
- Only real successful match calls are stored (no invented history).
- Includes `person_name`, scores, boxes, `annotated_image_url`, timestamps.

---

### 7.5 `GET /api/v1/media/outputs/{filename}`

Serves an annotated JPEG from `OUTPUT_DIR`.

- Filename allowlist: `[A-Za-z0-9._-]+` only.
- Path traversal rejected.
- Frontend should display:  
  `` `${VITE_API_BASE_URL}${annotated_image_url}` ``

---

### 7.6 `GET /api/v1/faces/model-info`

Returns configured backend metadata (CompreFace pack info / readiness). Used for diagnostics.

---

## 8. Response format history (important for clients)

Originally, detect/match returned **`multipart/mixed`** (JSON part + JPEG bytes). That broke simple `response.json()` usage in React.

**Current default:** `application/json` + `annotated_image_url`.

**Legacy:** still available via `?response_format=multipart`.

Do not assume absolute filesystem paths are usable in the browser; always use `annotated_image_url`.

---

## 9. Embeddings

CompreFace returns embedding vectors when called with:

```text
face_plugins=calculator
```

### Console logging

If `LOG_FACE_EMBEDDINGS=true`, uvicorn prints each face vector:

```text
[embedding] recognize face #1 dim=512 values=[...]
```

### Files (written on every match)

| File | Contents |
|------|----------|
| `outputs/embeddings/target_photo_embeddings.json` | Faces + embeddings from the **person/selfie** upload |
| `outputs/embeddings/group_photo_embeddings.json` | Faces + embeddings from the **group** upload |

Each file includes `source_filename`, `created_at`, per-face `bounding_box`, `embedding_dim`, and full `embedding` array. Files are **overwritten** on each match.

Embeddings are for inspection / future search (e.g. FAISS). They are **not** yet a primary matching path; matching still uses CompreFace temporary subject + recognize.

---

## 10. CORS & frontend integration

CORS middleware allows origins from `CORS_ALLOWED_ORIGINS` (`allow_credentials=False`).

Known frontend origins include:

- Local Vite/React hosts
- `https://image-locked-recreation.lovable.app`
- Preview URLs under `*.lovable.app` (add each new preview to env if needed)

Frontend env:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Expected FormData for match (as used by current Lovable UI):

```ts
formData.append("reference_image", selfieFile);
formData.append("group_image", groupFile);
formData.append("name", personName);
```

---

## 11. Error handling (selected)

| HTTP | When |
|------|------|
| 400 | Empty / corrupted image |
| 413 | Upload exceeds `MAX_UPLOAD_BYTES` |
| 415 | Unsupported format (non JPEG/PNG) |
| 422 | Missing required match image fields |
| 503 | Detector/matcher not ready, or CompreFace unreachable / bad key |
| 500 | Unexpected server errors |

Images with **zero faces** on detect still return **200** with `total_faces: 0` (valid result).

---

## 12. Local development

```bash
cd /path/to/Face_Dectection
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set COMPREFACE_API_KEY + SERVICE_TYPE=recognition

docker compose -f docker-compose.compreface.yml up -d

uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Smoke tests:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/api/v1/faces/history

curl -s \
  -F "reference_image=@uploads/Target_image.jpg" \
  -F "group_image=@uploads/Fullimage.jpg" \
  -F "name=Demo" \
  http://127.0.0.1:8000/api/v1/faces/match
```

Unit tests (mocked CompreFace):

```bash
pytest -q
```

---

## 13. Bounding boxes

- Coordinates are in **pixel space** of the decoded image.
- `(x1, y1)` = top-left, `(x2, y2)` = bottom-right.
- Origin is the image top-left.
- Detect annotations: green boxes.  
- Match annotations: orange “BEST MATCH {pct}%” on selected faces.

On very large images, CompreFace boxes may be scaled to fit decoded dimensions (`face_matcher._scale_box_to_image`).

---

## 14. What is intentionally out of scope (so far)

- No user authentication / multi-tenant auth in FastAPI.
- No production database (history is a local JSON file).
- No permanent CompreFace subject gallery (match subjects are temporary and deleted).
- No direct FAISS / vector DB search yet (embeddings are dumped for inspection).
- Frontend does not call CompreFace; only FastAPI.

---

## 15. Suggested next steps for new developers

1. Read this doc + `README.md`, then open Swagger at `/docs`.
2. Confirm CompreFace is healthy and `.env` uses a **recognition** API key.
3. Run a match with files under `uploads/`; inspect `outputs/` and `outputs/embeddings/`.
4. If integrating a new frontend origin, add it to `CORS_ALLOWED_ORIGINS` and restart.
5. If FormData field names differ, extend aliases in `app/routes/match.py` (`REFERENCE_FIELD_NAMES` / `TARGET_FIELD_NAMES` / `NAME_FIELD_NAMES`).

---

## 16. Version note

API version reported by FastAPI: **1.2.0** (see `app/main.py`).  
This document reflects the codebase as of the embedding-file + Lovable integration work (detect, match, history, media, CORS, embedding dumps).
