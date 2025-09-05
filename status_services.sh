#!/bin/bash

# Signature Cropper Services Status Script

cd /root/maids/signature_cropper

echo "📊 Signature Cropper Services Status"
echo "===================================="
echo ""

# Check API Server
echo "📡 API Server Status:"
if [ -f api_server.pid ]; then
    API_PID=$(cat api_server.pid)
    if kill -0 $API_PID 2>/dev/null; then
        echo "   ✅ Running (PID: $API_PID)"
        echo "   🔗 URL: http://203.161.46.120:8000"
        echo "   📄 Log: api_server.log"
    else
        echo "   ❌ Not running (PID file stale)"
        rm -f api_server.pid
    fi
else
    if pgrep -f "signature_api.py" > /dev/null; then
        echo "   ⚠️  Running but no PID file found"
        echo "   PID: $(pgrep -f signature_api.py)"
    else
        echo "   ❌ Not running"
    fi
fi

echo ""

# Check Web Server
echo "🌐 Web Server Status:"
if [ -f web_server.pid ]; then
    WEB_PID=$(cat web_server.pid)
    if kill -0 $WEB_PID 2>/dev/null; then
        echo "   ✅ Running (PID: $WEB_PID)"
        echo "   🔗 URL: http://203.161.46.120:8081/signature_extractor_app.html"
        echo "   📄 Log: web_server.log"
    else
        echo "   ❌ Not running (PID file stale)"
        rm -f web_server.pid
    fi
else
    if pgrep -f "http.server.*8081" > /dev/null; then
        echo "   ⚠️  Running but no PID file found"
        echo "   PID: $(pgrep -f 'http.server.*8081')"
    else
        echo "   ❌ Not running"
    fi
fi

echo ""

# Check ports
echo "🔌 Port Status:"
echo "   Port 8000 (API): $(netstat -ln | grep ':8000 ' > /dev/null && echo '✅ Open' || echo '❌ Closed')"
echo "   Port 8081 (Web): $(netstat -ln | grep ':8081 ' > /dev/null && echo '✅ Open' || echo '❌ Closed')"

echo ""

# Recent logs
if [ -f api_server.log ]; then
    echo "📄 Recent API Server Log (last 5 lines):"
    tail -5 api_server.log | sed 's/^/   /'
    echo ""
fi

if [ -f web_server.log ]; then
    echo "📄 Recent Web Server Log (last 5 lines):"
    tail -5 web_server.log | sed 's/^/   /'
fi
