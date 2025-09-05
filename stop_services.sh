#!/bin/bash

# Signature Cropper Services Stop Script

cd /root/maids/signature_cropper

echo "🛑 Stopping Signature Cropper Services..."

# Read PIDs if they exist
if [ -f api_server.pid ]; then
    API_PID=$(cat api_server.pid)
    if kill -0 $API_PID 2>/dev/null; then
        echo "📡 Stopping API Server (PID: $API_PID)..."
        kill $API_PID
    else
        echo "📡 API Server not running (PID file stale)"
    fi
    rm -f api_server.pid
else
    echo "📡 Stopping any running API Server processes..."
    pkill -f "signature_api.py" || echo "No API Server processes found"
fi

if [ -f web_server.pid ]; then
    WEB_PID=$(cat web_server.pid)
    if kill -0 $WEB_PID 2>/dev/null; then
        echo "🌐 Stopping Web Server (PID: $WEB_PID)..."
        kill $WEB_PID
    else
        echo "🌐 Web Server not running (PID file stale)"
    fi
    rm -f web_server.pid
else
    echo "🌐 Stopping any running Web Server processes..."
    pkill -f "http.server.*8081" || echo "No Web Server processes found"
fi

echo ""
echo "✅ Services Stopped Successfully!"
