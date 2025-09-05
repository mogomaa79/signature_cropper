#!/bin/bash

# Signature Cropper Services Startup Script
# This script starts both the API and web server as persistent background services

cd /root/maids/signature_cropper

echo "🚀 Starting Signature Cropper Services..."

# Kill any existing processes
pkill -f "signature_api.py" 2>/dev/null || true
pkill -f "http.server" 2>/dev/null || true
sleep 2

# Start API Server with nohup (persistent background process)
echo "📡 Starting API Server on port 8000..."
nohup python3 signature_api.py > api_server.log 2>&1 &
API_PID=$!
echo "API Server PID: $API_PID"

# Wait a moment for API to start
sleep 3

# Start Web Server with nohup (persistent background process)
echo "🌐 Starting Web Server on port 8081..."
nohup python3 -m http.server 8081 --bind 0.0.0.0 > web_server.log 2>&1 &
WEB_PID=$!
echo "Web Server PID: $WEB_PID"

# Save PIDs for later reference
echo $API_PID > api_server.pid
echo $WEB_PID > web_server.pid

echo ""
echo "✅ Services Started Successfully!"
echo "📡 API Server: http://203.161.46.120:8000"
echo "🌐 Web App: http://203.161.46.120:8081/signature_extractor_app.html"
echo ""
echo "📋 Process Information:"
echo "API Server PID: $API_PID (log: api_server.log)"
echo "Web Server PID: $WEB_PID (log: web_server.log)"
echo ""
echo "💡 These services will continue running even after you close the terminal!"
echo "💡 Use './stop_services.sh' to stop them"
echo "💡 Use './restart_services.sh' to restart them"
echo "💡 Use './status_services.sh' to check their status"
