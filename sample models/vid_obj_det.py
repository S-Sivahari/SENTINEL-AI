import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from pathlib import Path
from datetime import datetime
import json
import os

st.set_page_config(layout="wide", page_title="Sentinel AI")

st.markdown("# Sentinel AI — Border Threat Detection")

model = YOLO("yolov8m.pt")

CLASS_COLORS = {
    "person":     (0, 255, 0),
    "car":        (255, 165, 0),
    "truck":      (0, 165, 255),
    "motorcycle": (255, 0, 255),
    "bus":        (0, 255, 255),
}

# Initialize session state
if "video_history" not in st.session_state:
    st.session_state.video_history = {}
if "current_video" not in st.session_state:
    st.session_state.current_video = None
if "video_data" not in st.session_state:
    st.session_state.video_data = {}
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "current_frame_idx" not in st.session_state:
    st.session_state.current_frame_idx = 0

# Sidebar - Video History
st.sidebar.header("Upload History")

if st.session_state.video_history:
    selected_video = st.sidebar.selectbox(
        "Select a video",
        options=list(st.session_state.video_history.keys()),
        key="video_select"
    )
    if selected_video:
        st.session_state.current_video = selected_video
else:
    st.sidebar.info("No videos uploaded yet")

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    # Video Upload Section
    st.markdown("## Upload & Process Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file:
        # Save video details
        video_name = uploaded_file.name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        
        # Process video to extract all frames and detections
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Store video info
        video_id = f"{video_name}_{timestamp}"
        st.session_state.video_history[video_id] = {
            "name": video_name,
            "timestamp": timestamp,
            "path": tfile.name,
            "fps": fps,
            "total_frames": total_frames
        }
        st.session_state.current_video = video_id
        
        # Process and store frame data
        frame_data = {}
        with st.spinner("Processing video..."):
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)
                counts = {}
                detections = []
                
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls)
                        label = model.names[cls_id]
                        conf = float(box.conf)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        detections.append({
                            "label": label,
                            "confidence": f"{conf:.2%}",
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })
                        counts[label] = counts.get(label, 0) + 1
                
                frame_data[frame_idx] = {
                    "counts": counts,
                    "detections": detections,
                    "frame": frame
                }
                frame_idx += 1
        
        st.session_state.video_data[video_id] = frame_data
        cap.release()
        st.success(f"Processed {total_frames} frames successfully!")

# Display current video
if st.session_state.current_video:
    video_id = st.session_state.current_video
    video_info = st.session_state.video_history[video_id]
    frame_data = st.session_state.video_data.get(video_id, {})
    
    st.markdown(f"## Video: {video_info['name']}")
    st.caption(f"Uploaded: {video_info['timestamp']} | Frames: {video_info['total_frames']}")
    
    if frame_data:
        # Video player controls
        st.markdown("### Video Controls")
        col_play, col_pause, col_screenshot = st.columns(3)
        
        with col_play:
            if st.button("Play", key="play_btn", use_container_width=True):
                st.session_state.is_playing = True
        
        with col_pause:
            if st.button("Pause", key="pause_btn", use_container_width=True):
                st.session_state.is_playing = False
        
        with col_screenshot:
            if st.button("Screenshot", key="screenshot_btn", use_container_width=True):
                current_frame = frame_data[st.session_state.current_frame_idx]["frame"]
                rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                from PIL import Image
                img = Image.fromarray(rgb_frame)
                img_path = f"screenshot_{video_id}_{timestamp_str}.png"
                img.save(img_path)
                st.success(f"Screenshot saved: {img_path}")
        
        # Frame slider
        st.session_state.current_frame_idx = st.slider(
            "Frame",
            min_value=0,
            max_value=video_info['total_frames'] - 1,
            value=st.session_state.current_frame_idx,
            key="frame_slider"
        )
        
        # Display frame with detections
        frame_idx = st.session_state.current_frame_idx
        frame = frame_data[frame_idx]["frame"].copy()
        detections = frame_data[frame_idx]["detections"]
        counts = frame_data[frame_idx]["counts"]
        
        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            color = CLASS_COLORS.get(label, (200, 200, 200))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        
        # Display frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_frame, caption=f"Frame {frame_idx + 1}/{video_info['total_frames']}")

# Sidebar - Display stats for current video
with col2:
    if st.session_state.current_video:
        video_id = st.session_state.current_video
        frame_data = st.session_state.video_data.get(video_id, {})
        
        if frame_data:
            frame_idx = st.session_state.current_frame_idx
            counts = frame_data[frame_idx]["counts"]
            
            st.markdown("### Detections")
            if counts:
                for label, count in sorted(counts.items()):
                    st.metric(label.capitalize(), count)
            else:
                st.info("No detections")
