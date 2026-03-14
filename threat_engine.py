import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Alert:
    frame_idx:    int
    timestamp:    float      # seconds
    track_id:     int
    label:        str
    event:        str        # LINE_CROSSED | ZONE_INTRUSION | LOITERING
    direction:    str        # ENTRY | EXIT | N/A
    threat_level: str        # LOW | MEDIUM | HIGH | CRITICAL
    position:     Tuple[int, int]
    confidence:   float


class ThreatEngine:
    """
    Handles all threat logic:
      - virtual tripwire line crossing (with direction)
      - polygon zone intrusion
      - dwell / loitering detection
      - combined threat scoring
      - heatmap generation
    """

    def __init__(self, fps: int = 25):
        self.fps             = fps
        self.tripwire_p1:    Optional[Tuple[int,int]] = None
        self.tripwire_p2:    Optional[Tuple[int,int]] = None
        self.zone:           Optional[np.ndarray]     = None
        self.track_history:  dict = {}
        self.alerts:         List[Alert] = []
        self.intrusion_positions: List[Tuple[int,int]] = []

    # ── SETUP ──────────────────────────────────────────────────────────────

    def set_tripwire(self, p1: tuple, p2: tuple):
        self.tripwire_p1 = (int(p1[0]), int(p1[1]))
        self.tripwire_p2 = (int(p2[0]), int(p2[1]))

    def set_zone(self, points: list):
        if len(points) >= 3:
            self.zone = np.array(points, dtype=np.int32)
        else:
            self.zone = None

    def reset(self):
        self.track_history        = {}
        self.alerts               = []
        self.intrusion_positions  = []

    # ── GEOMETRY ───────────────────────────────────────────────────────────

    def _side(self, p1, p2, pt) -> float:
        """
        Cross product sign:
          positive  →  pt is on the 'threat' side of line p1→p2
          negative  →  pt is on the 'safe' side
        """
        return float(
            (p2[0]-p1[0]) * (pt[1]-p1[1])
          - (p2[1]-p1[1]) * (pt[0]-p1[0])
        )

    def _in_zone(self, pt) -> bool:
        if self.zone is None:
            return False
        return cv2.pointPolygonTest(
            self.zone, (float(pt[0]), float(pt[1])), measureDist=False
        ) >= 0

    def _bbox_in_zone(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        Check if ANY part of the bounding box intersects with the zone.
        This includes:
        - Any corner inside the zone
        - Any edge of bbox crossing zone boundary
        - Zone completely inside bbox
        """
        if self.zone is None:
            return False
        
        # Check if any corner of bbox is inside zone
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        for corner in corners:
            if self._in_zone(corner):
                return True
        
        # Create a bounding box polygon
        bbox = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        
        # Check if any zone vertex is inside the bbox
        for vertex in self.zone:
            pt = tuple(vertex)
            if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
                return True
        
        # Check if bounding box edges intersect with zone edges
        zone_edges = [(self.zone[i], self.zone[(i+1) % len(self.zone)]) for i in range(len(self.zone))]
        bbox_edges = [
            ((x1, y1), (x2, y1)),  # top
            ((x2, y1), (x2, y2)),  # right
            ((x2, y2), (x1, y2)),  # bottom
            ((x1, y2), (x1, y1))   # left
        ]
        
        for be in bbox_edges:
            for ze in zone_edges:
                if self._segments_intersect(be[0], be[1], ze[0], ze[1]):
                    return True
        
        return False

    def _segments_intersect(self, p1, p2, p3, p4) -> bool:
        """Check if line segments (p1-p2) and (p3-p4) intersect"""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    # ── SCORING ────────────────────────────────────────────────────────────

    def _score(self, label, conf, crossed, in_zone, dwell_secs, group_size=1) -> str:
        s = 0
        if crossed:           s += 60
        if in_zone:           s += 30
        if dwell_secs >= 8:   s += 25
        if label == "person": s += 10
        if label in ("car","truck","bus","motorcycle"): s += 15
        if group_size >= 3:   s += 20
        s = int(s * conf)
        if s >= 90: return "CRITICAL"
        if s >= 60: return "HIGH"
        if s >= 30: return "MEDIUM"
        return "LOW"

    # ── MAIN PER-FRAME LOGIC ───────────────────────────────────────────────

    def process_frame(
        self,
        frame_idx: int,
        detections: List[Tuple]   # (track_id, label, conf, x1, y1, x2, y2)
    ) -> List[Alert]:
        """
        Call once per frame after YOLO + ByteTrack.
        Returns list of new Alert objects triggered this frame.
        """
        frame_alerts:  List[Alert] = []
        DWELL_FRAMES = self.fps * 8   # 8 seconds

        # count people inside zone for group detection
        zone_persons = sum(
            1 for (_, lbl, _, x1, y1, x2, y2) in detections
            if lbl == "person" and self._bbox_in_zone(x1, y1, x2, y2)
        )

        for (track_id, label, conf, x1, y1, x2, y2) in detections:
            foot = ((x1+x2)//2, int(y2))   # bottom-centre = where feet touch ground

            # initialise per-track memory
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    "prev_sign":       None,
                    "zone_entry_frame": None,
                    "frames_in_zone":  0,
                    "alerted_cross":   False,
                    "alerted_zone":    False,
                }

            h = self.track_history[track_id]

            # ── TRIPWIRE ──────────────────────────────────────────────────
            if self.tripwire_p1 is not None:
                curr = self._side(self.tripwire_p1, self.tripwire_p2, foot)
                prev = h["prev_sign"]

                if prev is not None:
                    # negative → positive  =  crossed INTO threat side (ENTRY)
                    if prev < 0 and curr > 0:
                        level = self._score(label, conf, True, self._in_zone(foot), 0, zone_persons)
                        a = Alert(frame_idx, frame_idx/self.fps, track_id, label,
                                  "LINE_CROSSED", "ENTRY", level, foot, conf)
                        frame_alerts.append(a)
                        self.alerts.append(a)
                        self.intrusion_positions.append(foot)
                        h["alerted_cross"] = True

                    # positive → negative  =  crossed OUT (EXIT)
                    elif prev > 0 and curr < 0:
                        a = Alert(frame_idx, frame_idx/self.fps, track_id, label,
                                  "LINE_CROSSED", "EXIT", "LOW", foot, conf)
                        frame_alerts.append(a)
                        self.alerts.append(a)
                        h["alerted_cross"] = False   # reset so re-entry fires again

                h["prev_sign"] = curr

            # ── ZONE INTRUSION ────────────────────────────────────────────
            in_zone = self._bbox_in_zone(x1, y1, x2, y2)

            if in_zone:
                h["frames_in_zone"] += 1

                # first frame inside zone → fire ZONE_INTRUSION alert once
                if h["zone_entry_frame"] is None:
                    h["zone_entry_frame"] = frame_idx
                    if not h["alerted_zone"]:
                        level = self._score(label, conf, False, True, 0, zone_persons)
                        a = Alert(frame_idx, frame_idx/self.fps, track_id, label,
                                  "ZONE_INTRUSION", "ENTRY", level, foot, conf)
                        frame_alerts.append(a)
                        self.alerts.append(a)
                        self.intrusion_positions.append(foot)
                        h["alerted_zone"] = True

                # dwell threshold reached → LOITERING
                if h["frames_in_zone"] == DWELL_FRAMES:
                    dwell = DWELL_FRAMES / self.fps
                    level = self._score(label, conf, False, True, dwell, zone_persons)
                    a = Alert(frame_idx, frame_idx/self.fps, track_id, label,
                              "LOITERING", "N/A", level, foot, conf)
                    frame_alerts.append(a)
                    self.alerts.append(a)
                    self.intrusion_positions.append(foot)

            else:
                # object left zone — reset dwell state
                h["frames_in_zone"]  = 0
                h["zone_entry_frame"] = None
                h["alerted_zone"]    = False

        return frame_alerts

    # ── HEATMAP ────────────────────────────────────────────────────────────

    def generate_heatmap(self, frame_shape, positions: list = None) -> np.ndarray:
        """
        Returns a BGR heatmap image (same size as frame_shape) showing
        density of all intrusion / crossing events.
        Red = high activity, blue = low.
        """
        h, w   = frame_shape[:2]
        canvas = np.zeros((h, w), dtype=np.float32)
        pts    = positions if positions is not None else self.intrusion_positions

        for (cx, cy) in pts:
            if 0 <= int(cx) < w and 0 <= int(cy) < h:
                cv2.circle(canvas, (int(cx), int(cy)), radius=50, color=1.0, thickness=-1)

        if canvas.max() > 0:
            canvas /= canvas.max()

        colored = cv2.applyColorMap((canvas * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return colored
