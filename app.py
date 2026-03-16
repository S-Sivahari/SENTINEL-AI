"""
SentinelAI - Flask Backend
Run: python app.py
"""
import os, csv, base64, tempfile, threading, io as io_module, mimetypes
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, Response
from ultralytics import YOLO

from threat_engine import ThreatEngine

app   = Flask(__name__, static_folder="static")
MODELS = {
    "yolov8n": YOLO("yolov8n.pt"),
    "yolov8s": None,
    "yolov8m": None,
    "yolov8l": None,
}
current_model_key = "yolov8n"


CLASS_COLORS = {
    "person":     (0,   255,   0),
    "car":        (255, 165,   0),
    "truck":      (0,   165, 255),
    "motorcycle": (255,   0, 255),
    "bus":        (0,   255, 255),
}

STATE = {
    "video_path":   None,
    "media_type":   None,
    "first_frame":  None,
    "frame_shape":  None,
    "fps":          25,
    "total_frames": 0,
    "zone_points":  [],
    "tripwire":     None,
    "frame_data":   {},
    "all_alerts":   [],
    "heatmap_img":  None,
    "processing":   False,
    "progress":     0,
    "progress_text":"",
    "frame_skip":   3,
    "selected_model":"yolov8n",
    "conf_threshold": 0.25,
}


def _safe_model_key(raw_key) -> str:
    key = (raw_key or "yolov8n").strip().lower()
    return key if key in MODELS else "yolov8n"


def _safe_conf_threshold(raw_value, default: float = 0.25) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = float(default)
    return max(0.0, min(1.0, value))

def get_model(key=None):
    global current_model_key
    if key is None: key = STATE.get("selected_model", "yolov8n")
    key = _safe_model_key(key)
    
    if MODELS[key] is None:
        STATE["progress_text"] = f"Loading model {key}..."
        MODELS[key] = YOLO(f"{key}.pt")
    
    current_model_key = key
    return MODELS[key]



# ── HELPERS ────────────────────────────────────────────────────────────────────

def to_b64(frame_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def render_frame(frame: np.ndarray, idx: int) -> np.ndarray:
    out = frame.copy()

    if len(STATE["zone_points"]) >= 3:
        za  = np.array(STATE["zone_points"], dtype=np.int32)
        ovl = out.copy()
        cv2.fillPoly(ovl, [za], (0, 0, 200))
        cv2.addWeighted(ovl, 0.18, out, 0.82, 0, out)
        cv2.polylines(out, [za], True, (0, 0, 255), 2)
        cv2.putText(out, "RESTRICTED ZONE",
                    (STATE["zone_points"][0][0]+6, STATE["zone_points"][0][1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

    if STATE["tripwire"]:
        p1, p2 = tuple(STATE["tripwire"][0]), tuple(STATE["tripwire"][1])
        cv2.line(out, p1, p2, (0, 50, 255), 2)
        cv2.putText(out, "TRIPWIRE", (p1[0]+6, p1[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 50, 255), 1)

    fdata    = STATE["frame_data"].get(idx, {})
    dets     = fdata.get("dets",   [])
    falerts  = fdata.get("alerts", [])
    alerted  = {a["track_id"] for a in falerts}

    for (tid, lbl, conf, x1, y1, x2, y2) in dets:
        color = (0, 0, 255) if tid in alerted else CLASS_COLORS.get(lbl, (200, 200, 200))
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        txt = f"{lbl} #{tid} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(out, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    if falerts:
        levels = ["LOW","MEDIUM","HIGH","CRITICAL"]
        top    = max(falerts, key=lambda a: levels.index(a["threat_level"]))
        banner = f"  ALERT  {top['threat_level']}  —  {top['event']}  ({top['direction']})"
        cv2.rectangle(out, (0,0), (out.shape[1], 44), (0,0,180), -1)
        cv2.putText(out, banner, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)

    cv2.putText(out, f"Frame {idx}  |  {idx/STATE['fps']:.1f}s",
                (10, out.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)
    return out


def read_raw_frame(idx: int) -> np.ndarray:
    if STATE.get("media_type") == "image":
        if STATE["first_frame"] is None:
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        return STATE["first_frame"].copy()

    cap = cv2.VideoCapture(STATE["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else np.zeros(STATE["frame_shape"], dtype=np.uint8)


def _is_image_filename(name: str) -> bool:
    ext = os.path.splitext(name.lower())[1]
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_allowed_image_suffix(name: str) -> bool:
    ext = os.path.splitext((name or "").lower())[1]
    return ext in {".jpg", ".png"}


def _is_allowed_video_suffix(name: str) -> bool:
    ext = os.path.splitext((name or "").lower())[1]
    return ext == ".mp4"


def _reset_media_state(
    *,
    first_frame: np.ndarray,
    fps: int,
    total_frames: int,
    video_path: str | None,
    media_type: str,
):
    STATE.update({
        "video_path":   video_path,
        "media_type":   media_type,
        "first_frame":  first_frame,
        "frame_shape":  first_frame.shape,
        "fps":          fps,
        "total_frames": total_frames,
        "zone_points":  [],
        "tripwire":     None,
        "frame_data":   {},
        "all_alerts":   [],
        "heatmap_img":  None,
        "processing":   False,
        "progress":     0,
        "progress_text":"",
    })


def _media_payload(first_frame: np.ndarray, fps: int, total_frames: int, media_type: str) -> dict:
    return {
        "fps":          fps,
        "total_frames": total_frames,
        "first_frame":  to_b64(first_frame),
        "width":        first_frame.shape[1],
        "height":       first_frame.shape[0],
        "media_type":   media_type,
    }


def _init_from_video_bytes(raw: bytes, suffix: str = ".mp4"):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(raw)
    tfile.flush()
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame0 = cap.read()
    cap.release()

    if not ret:
        return None, "Cannot read video"

    _reset_media_state(
        first_frame=frame0,
        fps=fps,
        total_frames=max(total, 1),
        video_path=tfile.name,
        media_type="video",
    )
    return _media_payload(frame0, fps, max(total, 1), "video"), None


def _init_from_image_bytes(raw: bytes):
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Cannot read image"

    _reset_media_state(
        first_frame=img,
        fps=1,
        total_frames=1,
        video_path=None,
        media_type="image",
    )
    return _media_payload(img, 1, 1, "image"), None


# ── ROUTES ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("static/index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("media") or request.files.get("video")
    if f is None:
        return jsonify({"error": "No file uploaded"}), 400

    input_type = (request.form.get("input_type") or "video").lower()
    filename = f.filename or "upload.bin"
    suffix = os.path.splitext(filename)[1] or ".mp4"
    selected_model = _safe_model_key(request.form.get("model_type"))
    STATE["selected_model"] = selected_model
    STATE["conf_threshold"] = _safe_conf_threshold(
        request.form.get("conf_threshold"),
        STATE.get("conf_threshold", 0.25),
    )
    
    raw = f.read()
    if not raw:
        return jsonify({"error": "Uploaded file is empty"}), 400

    if input_type == "image" and not _is_allowed_image_suffix(filename):
        return jsonify({"error": "Image uploads must be .jpg or .png"}), 400
    if input_type == "video" and not _is_allowed_video_suffix(filename):
        return jsonify({"error": "Video uploads must be .mp4"}), 400


    payload = None
    err = "Unsupported media"

    if input_type == "image":
        payload, err = _init_from_image_bytes(raw)
    elif input_type == "video":
        payload, err = _init_from_video_bytes(raw, suffix)
    else:
        # auto-detect: try image first for image-looking filenames, then fallback to video.
        if _is_image_filename(filename):
            payload, err = _init_from_image_bytes(raw)
            if payload is None:
                payload, err = _init_from_video_bytes(raw, suffix)
        else:
            payload, err = _init_from_video_bytes(raw, suffix)
            if payload is None:
                payload, err = _init_from_image_bytes(raw)

    if payload is None:
        return jsonify({"error": err}), 400
    return jsonify(payload)


@app.route("/upload_link", methods=["POST"])
def upload_link():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    input_type = (data.get("input_type") or "auto").lower()
    selected_model = _safe_model_key(data.get("model_type"))
    STATE["selected_model"] = selected_model
    STATE["conf_threshold"] = _safe_conf_threshold(
        data.get("conf_threshold"),
        STATE.get("conf_threshold", 0.25),
    )

    if not url:
        return jsonify({"error": "URL is required"}), 400


    try:
        req = Request(url, headers={"User-Agent": "SentinelAI/1.0"})
        with urlopen(req, timeout=20) as resp:
            raw = resp.read()
            content_type = resp.headers.get("Content-Type", "")
    except Exception as ex:
        return jsonify({"error": f"Could not fetch URL: {ex}"}), 400

    if not raw:
        return jsonify({"error": "No data found at URL"}), 400

    parsed = urlparse(url)
    name = os.path.basename(parsed.path) or "remote_media"
    suffix = os.path.splitext(name)[1] or ".mp4"

    if input_type == "image" and not _is_allowed_image_suffix(name):
        return jsonify({"error": "Image links must point to .jpg or .png files"}), 400
    if input_type == "video" and not _is_allowed_video_suffix(name):
        return jsonify({"error": "Video links must point to .mp4 files"}), 400

    payload = None
    err = "Unsupported media"
    if input_type == "image":
        payload, err = _init_from_image_bytes(raw)
    elif input_type == "video":
        payload, err = _init_from_video_bytes(raw, suffix)
    else:
        looks_image = content_type.startswith("image/") or _is_image_filename(name)
        if looks_image:
            payload, err = _init_from_image_bytes(raw)
            if payload is None:
                payload, err = _init_from_video_bytes(raw, suffix)
        else:
            payload, err = _init_from_video_bytes(raw, suffix)
            if payload is None:
                payload, err = _init_from_image_bytes(raw)

    if payload is None:
        return jsonify({"error": err}), 400
    return jsonify(payload)


@app.route("/set_geometry", methods=["POST"])
def set_geometry():
    data = request.json
    STATE["zone_points"] = [list(p) for p in data.get("zone", [])]
    tw   = data.get("tripwire")
    STATE["tripwire"]    = [list(tw[0]), list(tw[1])] if tw else None

    preview = STATE["first_frame"].copy()
    if len(STATE["zone_points"]) >= 3:
        za  = np.array(STATE["zone_points"], dtype=np.int32)
        ovl = preview.copy()
        cv2.fillPoly(ovl, [za], (0, 0, 200))
        cv2.addWeighted(ovl, 0.25, preview, 0.75, 0, preview)
        cv2.polylines(preview, [za], True, (0, 0, 255), 2)
        cv2.putText(preview, "RESTRICTED ZONE",
                    (STATE["zone_points"][0][0]+6, STATE["zone_points"][0][1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    if STATE["tripwire"]:
        p1, p2 = tuple(STATE["tripwire"][0]), tuple(STATE["tripwire"][1])
        cv2.line(preview, p1, p2, (0,50,255), 2)
        cv2.putText(preview, "TRIPWIRE", (p1[0]+6, p1[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,50,255), 2)

    return jsonify({"preview": to_b64(preview)})


@app.route("/get_video_src")
def get_video_src():
    video_path = STATE.get("video_path")
    if STATE.get("media_type") != "video" or not video_path or not os.path.exists(video_path):
        return jsonify({"url": None})
    return jsonify({"url": "/video_source"})


@app.route("/video_source")
def video_source():
    video_path = STATE.get("video_path")
    if STATE.get("media_type") != "video" or not video_path or not os.path.exists(video_path):
        return jsonify({"error": "No video loaded"}), 400

    mime_type, _ = mimetypes.guess_type(video_path)
    return send_file(video_path, mimetype=mime_type or "application/octet-stream")


def _run_processing():
    engine = ThreatEngine(fps=STATE["fps"])
    if STATE["zone_points"]:
        engine.set_zone(STATE["zone_points"])
    if STATE["tripwire"]:
        engine.set_tripwire(STATE["tripwire"][0], STATE["tripwire"][1])

    frame_skip = max(1, int(STATE.get("frame_skip", 3)))
    conf_threshold = _safe_conf_threshold(STATE.get("conf_threshold", 0.25), 0.25)

    # Single-image path: process once and build review payloads compatible with video flow.
    if STATE.get("media_type") == "image":
        frame = STATE["first_frame"].copy()
        model = get_model()
        results = model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)

        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cid = int(box.cls)
                lbl = model.names[cid]
                conf = float(box.conf)
                if conf < conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tid = int(box.id) if box.id is not None else -1
                dets.append((tid, lbl, conf, x1, y1, x2, y2))

        alerts = engine.process_frame(0, dets)
        STATE["frame_data"] = {
            0: {
                "dets": dets,
                "alerts": [
                    {
                        "frame_idx":   a.frame_idx,
                        "timestamp":   round(a.timestamp, 2),
                        "track_id":    a.track_id,
                        "label":       a.label,
                        "event":       a.event,
                        "direction":   a.direction,
                        "threat_level":a.threat_level,
                        "position":    list(a.position),
                        "confidence":  round(a.confidence, 2),
                    }
                    for a in alerts
                ],
            }
        }

        hm = None
        if engine.intrusion_positions:
            hm = engine.generate_heatmap(STATE["frame_shape"], engine.intrusion_positions)

        STATE["all_alerts"] = [
            {
                "frame_idx":   a.frame_idx,
                "timestamp":   round(a.timestamp, 2),
                "track_id":    a.track_id,
                "label":       a.label,
                "event":       a.event,
                "direction":   a.direction,
                "threat_level":a.threat_level,
                "position":    list(a.position),
                "confidence":  round(a.confidence, 2),
            }
            for a in engine.alerts
        ]
        STATE["heatmap_img"] = hm
        STATE["total_frames"] = 1
        STATE["progress_text"] = "Frame 1 / 1"
        STATE["processing"] = False
        STATE["progress"] = 100
        return

    cap   = cv2.VideoCapture(STATE["video_path"])
    total = STATE["total_frames"] or 1
    fdata = {}
    idx   = 0
    last_dets = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_skip == 0:
            model = get_model()
            results = model.track(frame, tracker="bytetrack.yaml",
                                  persist=True, verbose=False)

            dets = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cid  = int(box.cls)
                    lbl  = model.names[cid]
                    conf = float(box.conf)
                    if conf < conf_threshold:
                        continue
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    tid  = int(box.id) if box.id is not None else -1
                    dets.append((tid, lbl, conf, x1, y1, x2, y2))
            last_dets = dets
            alerts = engine.process_frame(idx, dets)
        else:
            dets   = last_dets
            alerts = []

        fdata[idx] = {
            "dets": dets,
            "alerts": [
                {
                    "frame_idx":   a.frame_idx,
                    "timestamp":   round(a.timestamp, 2),
                    "track_id":    a.track_id,
                    "label":       a.label,
                    "event":       a.event,
                    "direction":   a.direction,
                    "threat_level":a.threat_level,
                    "position":    list(a.position),
                    "confidence":  round(a.confidence, 2),
                }
                for a in alerts
            ],
        }

        STATE["progress"]      = int(idx / total * 100)
        STATE["progress_text"] = f"Frame {idx} / {total}"
        idx += 1

    cap.release()

    hm = None
    if engine.intrusion_positions:
        hm = engine.generate_heatmap(STATE["frame_shape"], engine.intrusion_positions)

    STATE["frame_data"]   = fdata
    STATE["all_alerts"]   = [
        {
            "frame_idx":   a.frame_idx,
            "timestamp":   round(a.timestamp, 2),
            "track_id":    a.track_id,
            "label":       a.label,
            "event":       a.event,
            "direction":   a.direction,
            "threat_level":a.threat_level,
            "position":    list(a.position),
            "confidence":  round(a.confidence, 2),
        }
        for a in engine.alerts
    ]
    STATE["heatmap_img"]  = hm
    STATE["total_frames"] = idx
    STATE["processing"]   = False
    STATE["progress"]     = 100


@app.route("/process", methods=["POST"])
def process():
    if STATE["processing"]:
        return jsonify({"error": "Already processing"}), 400
    body = request.get_json(silent=True) or {}
    if "frame_skip" in body:
        try:
            STATE["frame_skip"] = max(1, int(body["frame_skip"]))
        except Exception:
            STATE["frame_skip"] = 3
    if "conf_threshold" in body:
        STATE["conf_threshold"] = _safe_conf_threshold(
            body.get("conf_threshold"),
            STATE.get("conf_threshold", 0.25),
        )
    if "model_type" in body:
        STATE["selected_model"] = _safe_model_key(body.get("model_type"))
    STATE["processing"] = True
    STATE["progress"]   = 0
    t = threading.Thread(target=_run_processing, daemon=True)
    t.start()
    return jsonify({"started": True})


@app.route("/progress")
def progress():
    return jsonify({
        "progress": STATE["progress"],
        "text":     STATE["progress_text"],
        "done":     not STATE["processing"] and STATE["progress"] == 100,
    })


@app.route("/frame/<int:idx>")
def get_frame(idx):
    if STATE["first_frame"] is None:
        return jsonify({"error": "No video loaded"}), 400

    raw      = read_raw_frame(idx)
    rendered = render_frame(raw, idx)

    fdata  = STATE["frame_data"].get(idx, {})
    alerts = fdata.get("alerts", [])
    dets   = fdata.get("dets",   [])
    counts = {}
    for (_, lbl, _, *_rest) in dets:
        counts[lbl] = counts.get(lbl, 0) + 1

    return jsonify({
        "frame":     to_b64(rendered),
        "alerts":    alerts,
        "counts":    counts,
        "timestamp": f"{idx / STATE['fps']:.1f}s",
    })


@app.route("/alerts")
def get_alerts():
    return jsonify(STATE["all_alerts"])


@app.route("/heatmap")
def get_heatmap():
    if STATE["heatmap_img"] is None:
        return jsonify({"heatmap": None, "stats": {}})

    base = STATE["first_frame"].copy()
    hm   = STATE["heatmap_img"]
    if hm.shape[:2] != base.shape[:2]:
        hm = cv2.resize(hm, (base.shape[1], base.shape[0]))
    blended = cv2.addWeighted(base, 0.55, hm, 0.45, 0)

    if len(STATE["zone_points"]) >= 3:
        cv2.polylines(blended, [np.array(STATE["zone_points"], np.int32)], True, (0,0,255), 2)
    if STATE["tripwire"]:
        cv2.line(blended, tuple(STATE["tripwire"][0]), tuple(STATE["tripwire"][1]), (0,50,255), 2)

    alerts = STATE["all_alerts"]
    cc, ec, lc = {}, {}, {}
    for a in alerts:
        cc[a["label"]]        = cc.get(a["label"], 0)        + 1
        ec[a["event"]]        = ec.get(a["event"], 0)        + 1
        lc[a["threat_level"]] = lc.get(a["threat_level"], 0) + 1

    return jsonify({
        "heatmap": to_b64(blended),
        "stats": {
            "by_class": cc, "by_event": ec, "by_level": lc,
            "entries":    sum(1 for a in alerts if a["event"]=="LINE_CROSSED" and a["direction"]=="ENTRY"),
            "exits":      sum(1 for a in alerts if a["event"]=="LINE_CROSSED" and a["direction"]=="EXIT"),
            "intrusions": sum(1 for a in alerts if a["event"]=="ZONE_INTRUSION"),
        },
    })


@app.route("/export_csv")
def export_csv():
    alerts = STATE["all_alerts"]
    if not alerts:
        return jsonify({"error": "No alerts"}), 400

    output = io_module.StringIO()
    fields = ["timestamp","frame_idx","track_id","label","event",
              "direction","threat_level","confidence","position"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for a in alerts:
        writer.writerow({**a, "position": f"({a['position'][0]},{a['position'][1]})"})

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=sentinel_alerts.csv"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)
