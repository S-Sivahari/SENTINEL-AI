"""
SentinelAI — Flask Backend
Run: python app.py
"""
import os, csv, base64, tempfile, threading, io as io_module

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, Response
from ultralytics import YOLO

from threat_engine import ThreatEngine

app   = Flask(__name__, static_folder="static")
model = YOLO("yolov8n.pt")

CLASS_COLORS = {
    "person":     (0,   255,   0),
    "car":        (255, 165,   0),
    "truck":      (0,   165, 255),
    "motorcycle": (255,   0, 255),
    "bus":        (0,   255, 255),
}

STATE = {
    "video_path":   None,
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
}


# ── HELPERS ────────────────────────────────────────────────────────────────────

def to_b64(frame_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


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
    cap = cv2.VideoCapture(STATE["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else np.zeros(STATE["frame_shape"], dtype=np.uint8)


# ── ROUTES ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("static/index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f     = request.files["video"]
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    f.save(tfile.name)

    cap = cv2.VideoCapture(tfile.name)
    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame0 = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Cannot read video"}), 400

    STATE.update({
        "video_path":   tfile.name,
        "first_frame":  frame0,
        "frame_shape":  frame0.shape,
        "fps":          fps,
        "total_frames": total,
        "zone_points":  [],
        "tripwire":     None,
        "frame_data":   {},
        "all_alerts":   [],
        "heatmap_img":  None,
        "processing":   False,
        "progress":     0,
        "progress_text":"",
    })

    return jsonify({
        "fps":          fps,
        "total_frames": total,
        "first_frame":  to_b64(frame0),
        "width":        frame0.shape[1],
        "height":       frame0.shape[0],
    })


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


def _run_processing():
    engine = ThreatEngine(fps=STATE["fps"])
    if STATE["zone_points"]:
        engine.set_zone(STATE["zone_points"])
    if STATE["tripwire"]:
        engine.set_tripwire(STATE["tripwire"][0], STATE["tripwire"][1])

    cap   = cv2.VideoCapture(STATE["video_path"])
    total = STATE["total_frames"] or 1
    fdata = {}
    idx   = 0
    last_dets = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % 3 == 0:
            results = model.track(frame, tracker="bytetrack.yaml",
                                  persist=True, verbose=False)
            dets = []
            for r in results:
                for box in r.boxes:
                    cid  = int(box.cls)
                    lbl  = model.names[cid]
                    conf = float(box.conf)
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
    if STATE["video_path"] is None:
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
