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
    event:        str        # LINE_CROSSED | TRIPWIRE_TOUCH | ZONE_INTRUSION | LOITERING
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
        self.zone_overlap_threshold: float = 0.01
        self.tripwire_contact_padding_px: int = 2
        self.side_epsilon_px: float = 2.0
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

    def _signed_distance_to_line(self, p1, p2, pt) -> float:
        """
        Signed perpendicular distance (in pixels) from point to directed line p1->p2.
          positive  -> point is on positive side
          negative  -> point is on negative side
          near zero -> point lies on/very near line
        """
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        denom = float(np.hypot(dx, dy))
        if denom == 0.0:
            return 0.0
        return float(
            (p2[0]-p1[0]) * (pt[1]-p1[1])
          - (p2[1]-p1[1]) * (pt[0]-p1[0])
        ) / denom

    def _line_side(self, p1, p2, pt) -> int:
        """Classify which side of line the point is on: -1, 0, +1."""
        d = self._signed_distance_to_line(p1, p2, pt)
        if d > self.side_epsilon_px:
            return 1
        if d < -self.side_epsilon_px:
            return -1
        return 0

    def _in_zone(self, pt) -> bool:
        if self.zone is None:
            return False
        return cv2.pointPolygonTest(
            self.zone, (float(pt[0]), float(pt[1])), measureDist=False
        ) >= 0

    def _bbox_zone_overlap_ratio(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        Returns intersection_area / bbox_area between bbox and zone polygon.
        Uses a local ROI mask so only meaningful overlap contributes.
        """
        if self.zone is None:
            return 0.0

        bx1, by1, bx2, by2 = int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))
        bw = bx2 - bx1
        bh = by2 - by1
        bbox_area = max(1, bw * bh)

        zx, zy, zw, zh = cv2.boundingRect(self.zone)
        ix1 = max(bx1, zx)
        iy1 = max(by1, zy)
        ix2 = min(bx2, zx + zw)
        iy2 = min(by2, zy + zh)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        roi_w = ix2 - ix1
        roi_h = iy2 - iy1

        zone_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        bbox_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

        shifted_zone = self.zone.copy()
        shifted_zone[:, 0] -= ix1
        shifted_zone[:, 1] -= iy1
        cv2.fillPoly(zone_mask, [shifted_zone.astype(np.int32)], (1,))

        cv2.rectangle(
            bbox_mask,
            (bx1 - ix1, by1 - iy1),
            (bx2 - ix1, by2 - iy1),
            (1,),
            thickness=-1,
        )

        inter = cv2.bitwise_and(zone_mask, bbox_mask)
        inter_area = int(np.count_nonzero(inter))
        return float(inter_area) / float(bbox_area)

    def _bbox_in_zone(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Object is considered in-zone when object bbox overlaps/touches zone bbox."""
        if self.zone is None:
            return False

        bx1, by1 = int(min(x1, x2)), int(min(y1, y2))
        bx2, by2 = int(max(x1, x2)), int(max(y1, y2))
        zx, zy, zw, zh = cv2.boundingRect(self.zone)
        zx2 = zx + max(0, zw)
        zy2 = zy + max(0, zh)

        return self._boxes_touch_or_overlap(bx1, by1, bx2, by2, zx, zy, zx2, zy2)

    @staticmethod
    def _boxes_touch_or_overlap(
        ax1: int, ay1: int, ax2: int, ay2: int,
        bx1: int, by1: int, bx2: int, by2: int,
    ) -> bool:
        """Inclusive overlap test: touching edges also counts as contact."""
        a_min_x, a_max_x = min(ax1, ax2), max(ax1, ax2)
        a_min_y, a_max_y = min(ay1, ay2), max(ay1, ay2)
        b_min_x, b_max_x = min(bx1, bx2), max(bx1, bx2)
        b_min_y, b_max_y = min(by1, by2), max(by1, by2)

        return not (
            a_max_x < b_min_x
            or b_max_x < a_min_x
            or a_max_y < b_min_y
            or b_max_y < a_min_y
        )

    @staticmethod
    def _point_in_bbox(px: float, py: float, x1: int, y1: int, x2: int, y2: int) -> bool:
        bx1, by1 = min(x1, x2), min(y1, y2)
        bx2, by2 = max(x1, x2), max(y1, y2)
        return bx1 <= px <= bx2 and by1 <= py <= by2

    @staticmethod
    def _orientation(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> int:
        """Returns 0 for collinear, 1 for clockwise, 2 for counterclockwise."""
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-6:
            return 0
        return 1 if val > 0 else 2

    @staticmethod
    def _on_segment(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
        return (
            min(a[0], c[0]) - 1e-6 <= b[0] <= max(a[0], c[0]) + 1e-6
            and min(a[1], c[1]) - 1e-6 <= b[1] <= max(a[1], c[1]) + 1e-6
        )

    def _segments_intersect(
        self,
        p1: Tuple[float, float],
        q1: Tuple[float, float],
        p2: Tuple[float, float],
        q2: Tuple[float, float],
    ) -> bool:
        o1 = self._orientation(p1, q1, p2)
        o2 = self._orientation(p1, q1, q2)
        o3 = self._orientation(p2, q2, p1)
        o4 = self._orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and self._on_segment(p1, p2, q1):
            return True
        if o2 == 0 and self._on_segment(p1, q2, q1):
            return True
        if o3 == 0 and self._on_segment(p2, p1, q2):
            return True
        if o4 == 0 and self._on_segment(p2, q1, q2):
            return True
        return False
    def _bbox_touches_tripwire(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """True when object bbox overlaps/touches tripwire bbox (with small padding)."""
        if self.tripwire_p1 is None or self.tripwire_p2 is None:
            return False

        bx1, by1 = int(min(x1, x2)), int(min(y1, y2))
        bx2, by2 = int(max(x1, x2)), int(max(y1, y2))

        tx1 = min(self.tripwire_p1[0], self.tripwire_p2[0]) - self.tripwire_contact_padding_px
        ty1 = min(self.tripwire_p1[1], self.tripwire_p2[1]) - self.tripwire_contact_padding_px
        tx2 = max(self.tripwire_p1[0], self.tripwire_p2[0]) + self.tripwire_contact_padding_px
        ty2 = max(self.tripwire_p1[1], self.tripwire_p2[1]) + self.tripwire_contact_padding_px

        if self._boxes_touch_or_overlap(bx1, by1, bx2, by2, tx1, ty1, tx2, ty2):
            return True

        p1 = (float(self.tripwire_p1[0]), float(self.tripwire_p1[1]))
        p2 = (float(self.tripwire_p2[0]), float(self.tripwire_p2[1]))

        # Fast-path: line endpoint already inside bbox.
        if self._point_in_bbox(p1[0], p1[1], bx1, by1, bx2, by2):
            return True
        if self._point_in_bbox(p2[0], p2[1], bx1, by1, bx2, by2):
            return True

        top_left = (float(bx1), float(by1))
        top_right = (float(bx2), float(by1))
        bottom_right = (float(bx2), float(by2))
        bottom_left = (float(bx1), float(by2))
        edges = [
            (top_left, top_right),
            (top_right, bottom_right),
            (bottom_right, bottom_left),
            (bottom_left, top_left),
        ]

        return any(self._segments_intersect(p1, p2, e1, e2) for (e1, e2) in edges)

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
                    "last_stable_side": None,
                    "zone_entry_frame": None,
                    "frames_in_zone":  0,
                    "alerted_cross":   False,
                    "alerted_touch":   False,
                    "alerted_zone":    False,
                }

            h = self.track_history[track_id]

            # ── TRIPWIRE ──────────────────────────────────────────────────
            if self.tripwire_p1 is not None:
                curr_side = self._line_side(self.tripwire_p1, self.tripwire_p2, foot)
                last_side = h["last_stable_side"]
                crossed_now = False

                # Ignore near-line jitter; only update/alert on stable side values.
                if curr_side != 0:
                    if last_side is None:
                        h["last_stable_side"] = curr_side
                    elif curr_side != last_side:
                        direction = "ENTRY" if (last_side < 0 and curr_side > 0) else "EXIT"
                        level = self._score(label, conf, True, self._bbox_in_zone(x1, y1, x2, y2), 0, zone_persons)
                        a = Alert(
                            frame_idx,
                            frame_idx / self.fps,
                            track_id,
                            label,
                            "LINE_CROSSED",
                            direction,
                            level,
                            foot,
                            conf,
                        )
                        frame_alerts.append(a)
                        self.alerts.append(a)
                        self.intrusion_positions.append(foot)
                        h["alerted_cross"] = (direction == "ENTRY")
                        h["last_stable_side"] = curr_side
                        crossed_now = True

                touches_now = self._bbox_touches_tripwire(x1, y1, x2, y2)
                if touches_now and not crossed_now and not h["alerted_touch"]:
                    level = self._score(label, conf, True, self._bbox_in_zone(x1, y1, x2, y2), 0, zone_persons)
                    a = Alert(
                        frame_idx,
                        frame_idx / self.fps,
                        track_id,
                        label,
                        "TRIPWIRE_TOUCH",
                        "N/A",
                        level,
                        foot,
                        conf,
                    )
                    frame_alerts.append(a)
                    self.alerts.append(a)
                    self.intrusion_positions.append(foot)
                    h["alerted_touch"] = True
                elif not touches_now:
                    h["alerted_touch"] = False

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

    def generate_heatmap(
        self,
        frame_shape,
        positions: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
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
                cv2.circle(canvas, (int(cx), int(cy)), radius=50, color=(1.0,), thickness=-1)

        if canvas.max() > 0:
            canvas /= canvas.max()

        colored = cv2.applyColorMap((canvas * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return colored
