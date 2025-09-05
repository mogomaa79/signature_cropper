#!/bin/bash

# Signature Cropper API Startup Script

echo "🚀 Starting Signature Cropper API..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if requirements are installed
if ! python3 -c "import fastapi, uvicorn, cv2, pytesseract, fitz, google.generativeai" &> /dev/null; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Check if Tesseract is available
if ! command -v tesseract &> /dev/null; then
    echo "⚠️  Tesseract OCR is not installed. Please install it:"
    echo "   macOS: brew install tesseract"
    echo "   Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng"
    echo ""
    echo "The API will still start, but OCR functionality may be limited."
fi

# Set environment variable if not already set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "ℹ️  GEMINI_API_KEY environment variable not set. Using hardcoded API key."
fi

echo "🌐 Starting API server on http://localhost:8000"
echo "📖 API documentation available at http://localhost:8000/docs"
echo "❤️  Health check available at http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the API server
python3 signature_api.py
