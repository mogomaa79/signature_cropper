#!/bin/bash

# Signature Cropper Services Restart Script

echo "🔄 Restarting Signature Cropper Services..."
echo ""

# Stop services
./stop_services.sh

echo ""
echo "⏳ Waiting 3 seconds..."
sleep 3
echo ""

# Start services
./start_services.sh
