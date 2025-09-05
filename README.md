# Signature Cropper API

This API extracts signature zones from contract documents using the exact same logic from the MOHRE OCR Colab notebook.

## Features

- Accepts PDF files or image files (JPG, PNG, TIFF, BMP)
- Extracts the second page from PDF documents
- Uses advanced OCR and computer vision to detect signature zones
- Returns cropped signature images as JPG files
- Includes multiple fallback detection methods for robust performance

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Tesseract OCR**:
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng`
   - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

3. **Set Environment Variable** (optional):
   ```bash
   export GEMINI_API_KEY="your_gemini_api_key_here"
   ```
   Note: The API key is currently hardcoded in the script, but you can override it with this environment variable.

4. **Run the API**:
   ```bash
   python signature_api.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn signature_api:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints

### POST /crop-signature

Extracts and returns the cropped signature from a contract document.

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: Form field named `file` containing the contract document

**Response**:
- Success (200): Returns the cropped signature as a JPG image
- Error (400): Invalid file type
- Error (404): No signature zone detected
- Error (500): Internal server error

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/crop-signature" \
     -H "accept: image/jpeg" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@contract.pdf" \
     --output signature.jpg
```

**Example using Python requests**:
```python
import requests

with open('contract.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/crop-signature',
        files={'file': f}
    )

if response.status_code == 200:
    with open('signature.jpg', 'wb') as output:
        output.write(response.content)
    print("Signature saved as signature.jpg")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### GET /

Health check endpoint that returns basic status information.

### GET /health

Detailed health check that shows configuration status.

## Supported File Types

- PDF files (.pdf) - Extracts from the second page
- Image files (.jpg, .jpeg, .png, .tiff, .bmp)

## Algorithm Overview

The API uses the exact same signature detection algorithm from the MOHRE OCR Colab notebook:

1. **Document Processing**: Extracts the second page from PDF or loads image file
2. **Region Focusing**: Crops to bottom 2/3 of the document where signatures typically appear
3. **Text Detection**: Uses Tesseract OCR to find text elements and signature-related keywords
4. **Zone Detection**: Multiple methods:
   - Primary: Text-based detection using signature keywords
   - Backup 1: Enhanced image processing with edge detection
   - Backup 2: Grey boundary detection using document structure analysis
5. **Zone Refinement**: Applies 30% horizontal shrinkage and 5% height reduction for precision
6. **Signature Extraction**: Crops the detected zone and returns as JPG

## Configuration

The API uses the same parameters as the original notebook:
- Gemini 2.5 Pro for advanced classification (when needed)
- 30% horizontal zone shrinkage for precision
- 5% height reduction to avoid document boundaries
- Multiple fallback detection methods for robustness

## Notes

- The API preserves the exact logic from the original Colab notebook
- All signature detection parameters and methods are unchanged
- The only modification is the interface - from notebook cells to HTTP API endpoints
- Tesseract OCR must be installed separately on the system
