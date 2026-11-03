# Project: Face Detection Service – Phase 1

Act as a senior Python backend developer and computer vision engineer.

I want to build Phase 1 of an event photo distribution platform. For now, focus ONLY on face detection. Do not implement face recognition, embeddings, similarity matching, databases, or authentication.

## 1. Objective

Build a Python FastAPI application that:

1. Accepts an uploaded image containing one or multiple people.
2. Detects all visible faces in the image.
3. Identifies the bounding box coordinates of each detected face.
4. Draws a rectangle around every detected face.
5. Returns the processed image and face detection metadata.
6. Handles invalid images and images with no detectable faces.

The application should work with group photographs containing multiple people.

## 2. Technology stack

- Python 3.11+
- FastAPI
- Uvicorn
- InsightFace
- SCRFD face detection model
- OpenCV
- NumPy
- python-multipart

Use a pretrained SCRFD model through InsightFace.

Important:
- Do not train a model from scratch.
- Do not implement ArcFace or face recognition in this phase.
- Do not generate or store face embeddings.
- Check the pretrained model weights' licensing before commercial use.
- Keep model initialization separate from API request handling.
- Load the model only once when the application starts.

## 3. Project structure

Create the following structure:

face-detection-service/
│
├── app/
│   ├── main.py
│   │
│   ├── routes/
│   │   └── detection.py
│   │
│   ├── services/
│   │   └── face_detector.py
│   │
│   ├── schemas/
│   │   └── detection_schema.py
│   │
│   └── utils/
│       └── image_utils.py
│
├── uploads/
├── outputs/
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md

Explain the responsibility of each file.

## 4. Face detection service

Implement face_detector.py.

Requirements:

- Initialize InsightFace FaceAnalysis with an appropriate SCRFD model.
- Use CPU execution initially so that it runs on ordinary laptops.
- Configure detection size appropriately, such as 640x640.
- Load the model once during application startup.
- Implement a function:

detect_faces(image)

This function should:

1. Receive an OpenCV image.
2. Detect all faces.
3. Extract bounding box coordinates.
4. Extract detection confidence scores.
5. Return structured detection results.

Each detected face should contain:

- face_id
- bounding_box:
  - x1
  - y1
  - x2
  - y2
- confidence

Do not include identity or person names because this is only face detection.

## 5. Image processing

Implement image_utils.py.

Requirements:

- Validate uploaded image format.
- Support JPEG, JPG, and PNG.
- Decode image using OpenCV.
- Draw green bounding boxes around detected faces.
- Display face ID and confidence above each bounding box.
- Preserve the original image dimensions.
- Encode the processed image as JPEG for API response.

Do not unnecessarily save uploaded images permanently.

## 6. API endpoint

Implement:

POST /api/v1/faces/detect

Accept:
- multipart/form-data
- image: uploaded image file

Example response:

{
  "success": true,
  "total_faces": 3,
  "faces": [
    {
      "face_id": 1,
      "bounding_box": {
        "x1": 100,
        "y1": 80,
        "x2": 200,
        "y2": 220
      },
      "confidence": 0.98
    },
    {
      "face_id": 2,
      "bounding_box": {
        "x1": 300,
        "y1": 90,
        "x2": 390,
        "y2": 210
      },
      "confidence": 0.95
    }
  ],
  "message": "Face detection completed"
}

Return the processed image alongside the JSON metadata using a well-documented response format. Prefer a multipart/mixed response with a JSON metadata part and an image part, or provide a separate processed-image endpoint. Explain your choice and make it easy to test in Swagger UI.

Also provide:
- GET /health
- GET /api/v1/faces/model-info

## 7. Error handling

Handle:

- Unsupported image format.
- Empty image upload.
- Corrupted image.
- Image too large.
- No faces detected.
- Model initialization failure.
- Unexpected internal errors.

Use appropriate HTTP status codes and clear error messages.

For an image with no faces, return HTTP 200 with total_faces = 0. This is a valid detection result, not an API failure.

Set a reasonable upload size limit, such as 10 MB.

## 8. Performance and code quality

- Use modular code.
- Add type hints.
- Add meaningful comments.
- Avoid loading the model on every API request.
- Keep image processing separate from route logic.
- Avoid unnecessary global mutable state.
- Include useful logging.
- Ensure multiple simultaneous requests do not cause unsafe model initialization.
- Do not log image contents or sensitive personal data.

## 9. Testing

Create basic tests using pytest and FastAPI TestClient.

Test:
- Valid image containing faces.
- Image with no faces.
- Invalid file format.
- Corrupted image.
- Empty upload.

If actual model weights are unavailable in the test environment, mock the detector for unit tests. Also provide a manual test using a real photograph.

## 10. Documentation

Create README.md explaining:

1. Project overview.
2. Architecture and request flow.
3. Python virtual environment setup.
4. Dependency installation.
5. Model download and initialization.
6. Running the FastAPI server.
7. Swagger UI testing.
8. API request and response examples.
9. How bounding box coordinates work.
10. Common installation and model-loading errors.

## 11. Expected result

When I run the application and upload a group photograph:

- The API detects all visible faces.
- Each face receives a bounding box.
- Each bounding box displays a face number and confidence score.
- The API returns detection metadata.
- I can view the processed image.
- I can test everything through Swagger UI at /docs.

## 12. Development instructions

Before writing code:

1. Explain the architecture briefly.
2. Explain how SCRFD works at a high level.
3. Explain how InsightFace loads pretrained detection models.
4. Then create the complete project.

After implementation:

1. Give me exact terminal commands to install and run.
2. Explain how to test with a real group photograph.
3. Explain the complete request lifecycle from image upload to detection response.
4. Verify that the code is runnable and fix implementation issues.

Keep the code beginner-friendly but production-conscious.

Do not implement Phase 2 until Phase 1 is working successfully.