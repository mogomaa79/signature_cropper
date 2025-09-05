# 🚀 Signature Cropper - Persistent Service Management

Your Signature Cropper is now configured to run **persistently** even when you close terminals and applications!

## 📋 Current Status
✅ **Services are RUNNING and PERSISTENT**
- 📡 **API Server**: http://203.161.46.120:8000
- 🌐 **Web App**: http://203.161.46.120:8081/signature_extractor_app.html

## 🛠️ Service Management Options

### **Option 1: Simple Scripts (Recommended)**
Use the convenient management scripts I created:

```bash
# Start both services (persistent background)
./start_services.sh

# Check status of both services
./status_services.sh

# Stop both services
./stop_services.sh

# Restart both services
./restart_services.sh
```

### **Option 2: Systemd Services (Advanced)**
For enterprise-grade service management:

```bash
# Start services
sudo systemctl start signature-api signature-web

# Check status
sudo systemctl status signature-api signature-web

# Stop services
sudo systemctl stop signature-api signature-web

# Restart services
sudo systemctl restart signature-api signature-web

# Enable auto-start on boot
sudo systemctl enable signature-api signature-web
```

## 📄 Log Files

### Script-based Services:
- **API Logs**: `api_server.log`
- **Web Logs**: `web_server.log`

### Systemd Services:
```bash
# View API logs
sudo journalctl -u signature-api -f

# View Web logs
sudo journalctl -u signature-web -f
```

## 🔧 Service Details

### API Server (Port 8000)
- **Endpoints**: `/health`, `/crop-signature`
- **Features**: CORS enabled, fuzzy text matching
- **Technology**: FastAPI + Uvicorn

### Web Server (Port 8081)
- **Files**: `signature_extractor_app.html`
- **Features**: Drag & drop upload, batch processing, ZIP download
- **Technology**: Python HTTP server

## 🎯 Quick Commands

```bash
# Check if services are running
ps aux | grep -E "(signature_api|http.server)"

# Check ports
netstat -tlnp | grep -E ":8000|:8081"

# Quick health check
curl -s http://localhost:8000/health

# View recent API activity
tail -f api_server.log
```

## 🔄 What Happens Now?

✅ **Services will continue running** even if you:
- Close this terminal
- Close Cursor
- Log out of SSH
- Restart your laptop

✅ **Services will automatically restart** if they crash (systemd only)

✅ **Services will start on server boot** (systemd enabled)

## 🚨 Important Notes

1. **Services are already running** - you don't need to do anything else!
2. **Use `./status_services.sh`** to check if everything is working
3. **Logs are continuously written** to the log files
4. **Both approaches work** - use scripts for simplicity or systemd for robustness

## 🌐 Access URLs

- **Web App**: http://203.161.46.120:8081/signature_extractor_app.html
- **API Health**: http://203.161.46.120:8000/health
- **API Docs**: http://203.161.46.120:8000/docs

---

**Your Signature Cropper is now production-ready and will run 24/7!** 🎉
