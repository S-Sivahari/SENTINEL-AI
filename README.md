# 🛡️ SentinelAI — Border Threat Detection System

A real-time AI-powered surveillance system for detecting threats at borders using YOLO v8 object detection, persistent tracking, polygon zone intrusion detection, tripwire crossing alerts, and intelligent threat scoring.

[![Powered by Hugging Face Spaces](https://img.shields.io/badge/Powered%20by-Hugging%20Face%20Spaces-blue?logo=huggingface)](https://huggingface.co/spaces)

## ✨ Features

- **🎯 YOLO v8 Detection** — Real-time object detection (person, car, truck, motorcycle, bus)
- **👁️ ByteTrack** — Persistent multi-object tracking across frames
- **🚨 Zone Intrusion Detection** — Flag objects entering restricted polygon zones (any part touching)
- **━ Tripwire Crossing** — Detect crossing direction (ENTRY/EXIT)
- **⏱️ Loitering Detection** — Alert on objects dwelling >8 seconds
- **📊 Threat Scoring** — LOW/MEDIUM/HIGH/CRITICAL based on event type, confidence, and context
- **🗺️ Heatmap** — Visual density map of all intrusion events
- **📋 Alert Log** — Timestamped alerts with export to CSV
- **📹 Frame Scrubbing** — Play/pause, frame-by-frame review, click-to-jump alerts

## 🚀 Quick Start

### Local Development

```bash
# Clone and setup
git clone <your-repo-url>
cd sentinel-ai
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Flask server (port 7860 for HF compatibility)
python app.py
```

Visit `http://localhost:7860` in your browser.

### Docker (Recommended for Deployment)

```bash
docker build -t sentinel-ai .
docker run -p 7860:7860 sentinel-ai
```

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Detection** | YOLOv8 (Ultralytics) |
| **Tracking** | ByteTrack |
| **Backend** | Flask (Python) |
| **Frontend** | HTML5 + JavaScript + Canvas |
| **Deployment** | Docker on Hugging Face Spaces |

## 🎮 How to Use

### Step 1: Upload Video
- Drag & drop or select a surveillance video (mp4, avi, mov, mkv)
- System extracts and processes the first frame

### Step 2: Define Restricted Zone & Tripwire
- **Polygon Zone** (🔺): Click to place corners, double-click to close → defines restricted area
- **Tripwire Line** (━): Draw a line for entry/exit detection → crossing triggers alerts
- Click **"Confirm & Preview"** to verify placement

### Step 3: Configure & Process
- Adjust confidence threshold (0-1): ignore low-confidence detections
- Click **"Process Video"** to analyze entire video
- System runs YOLO + ByteTrack on every frame

### Step 4: Review Results
- **Video Playback**: Frame-by-frame review with bounding boxes and alert banners
- **Alert Log**: Click any alert to jump to that frame
- **Heatmap**: See density of intrusions with breakdown statistics
- **Export**: Download all alerts as CSV

## 📊 Threat Scoring System

Alerts are scored 0-100 based on multiple factors:

| Factor | Points | Notes |
|--------|--------|-------|
| **Line Crossing** | +60 | Crossing tripwire into threat zone |
| **Zone Intrusion** | +30 | ANY part of object in zone |
| **Loitering (≥8s)** | +25 | Object dwelling in zone |
| **Person Detection** | +10 | Human presence |
| **Vehicles** | +15 | Car, truck, bus, motorcycle |
| **Group Size (≥3)** | +20 | Multiple people together |
| **Confidence × 100** | 0-100% | Multiplier on final score |

**Final Thresholds:**
- **CRITICAL**: ≥ 90 points
- **HIGH**: 60-89 points
- **MEDIUM**: 30-59 points
- **LOW**: < 30 points

## 🔍 Detection Algorithm Details

### Zone Intrusion (Bbox Intersection)
Objects are flagged if **ANY part** of their bounding box intersects the zone:
- Check if any bbox corner is inside zone
- Check if any zone vertex is inside bbox
- Detect edge-to-edge intersections using computational geometry
- **Not** just the foot point

### Tripwire Crossing
- Uses 2D cross-product to determine which side of line the object is on
- Frame-to-frame sign change = crossing detected
- Negative→Positive = **ENTRY** (high threat, +60 pts)
- Positive→Negative = **EXIT** (low threat, +0 pts)

### Loitering/Dwell
- Counts consecutive frames object stays in zone
- Threshold: 8 seconds (customizable)
- Single alert per loiter event

## 📁 Project Structure

```
sentinel-ai/
├── app.py                    # Flask backend + routes
├── threat_engine.py          # Threat logic engine
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker build config
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── static/
│   └── index.html          # Web UI (HTML5 + Canvas)
├── yolov8n.pt              # YOLO nano model (~40MB)
├── yolov8m.pt              # YOLO medium model (~50MB)
└── videos/                 # Test videos folder
```

## 🔧 Configuration

### In `threat_engine.py`:
```python
DWELL_FRAMES = self.fps * 8   # Change loitering threshold (8 seconds)

# Adjust threat scoring weights:
def _score(self, label, conf, crossed, in_zone, dwell_secs, group_size=1) -> str:
    s = 0
    if crossed:           s += 60  # ← change line crossing weight
    if in_zone:           s += 30  # ← change zone intrusion weight
    # ... more weights
```

### In `app.py`:
```python
model = YOLO("yolov8n.pt")  # Change to yolov8m.pt for better accuracy

CLASS_COLORS = {            # Customize detection colors
    "person":     (0,   255,   0),
    # ... more classes
}
```

## 📡 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve web UI |
| `/upload` | POST | Upload video file (multipart form-data) |
| `/analysis` | GET | Run threat analysis on uploaded video |
| `/frames` | GET | Get processed frame (query param: `frame=idx`) |
| `/export` | GET | Download alerts as CSV file |

### Example: Get Frame 42
```
GET /frames?frame=42
```

## 🐳 Hugging Face Spaces Deployment

This repo is configured for seamless deployment to HF Spaces:

1. **Setup**: Push to GitHub repo
2. **Create Space**: 
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Select **Docker** as SDK
   - Link your GitHub repo
3. **Auto-deploy**: HF Spaces will:
   - Build Docker image from `Dockerfile`
   - Install dependencies from `requirements.txt`
   - Download YOLO models on first startup (cached after)
   - Expose app on port 7860

**First startup takes 2-3 minutes** (downloading models). Subsequent restarts are faster.

### Environment Variables
HF Spaces automatically injects:
- `PORT=7860` — Web server port
- `HF_SPACE_ID`, `HF_SPACE_PERSISTENT_STORAGE` — Storage path

## ⚙️ Requirements

- Python 3.10+
- 2GB+ RAM (for video processing)
- Optional: CUDA GPU for faster inference

**Pip packages** (see `requirements.txt`):
```
ultralytics>=8.0.0
opencv-python-headless
Flask>=2.0.0
numpy
pandas
Pillow
lapx
```

## 🎓 How It Works (Technical)

```
Video Input
    ↓
[Frame Extraction]
    ↓
[YOLO Detection]  → Bounding boxes + confidence
    ↓
[ByteTrack]       → Assign persistent track IDs
    ↓
[Threat Engine]   → Check zone/line + score
    ├─ Geometry checks (point-in-polygon, bbox intersection)
    ├─ Crossing detection (cross-product method)
    ├─ Group detection (multi-person alerts)
    └─ Threat scoring (weighted combination)
    ↓
[Alert Generation] → Alert objects with metadata
    ↓
[Rendering]       → Draw boxes, zones, alert banners
    ↓
[Export/Review]   → Playback, heatmap, CSV
```

## 📝 License

MIT License — Free for research, commercial use requires licensing.

## 🤝 Contributing

Issues and PRs welcome! Please follow the project's code style.

## 👨‍💻 Author

Built for border security, surveillance, and crowded area monitoring.

---

**Status**: ✅ Production Ready | **Last Updated**: March 2026


## License

MIT
