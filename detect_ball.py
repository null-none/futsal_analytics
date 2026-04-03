#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import csv
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PAD = 10
STATS_BAR_H = 64

ZONE_COLORS_BGR = [
    (200,  80,  80),   # Zone 1
    ( 80, 200,  80),   # Zone 2
    ( 80,  80, 200),   # Zone 3
    ( 80, 200, 200),   # Zone 4
]


class BallDetector:
    def __init__(
        self,
        weights: str = "best.pt",
        source: str = "input.mp4",
        out_dir: str = "out",
        conf: float = 0.25,
        imgsz: int = 1280,
        ball_class: int = 1,
    ):
        self.weights = weights
        self.source = source
        self.out_dir = Path(out_dir)
        self.conf = conf
        self.imgsz = imgsz
        self.ball_class = ball_class

    # ------------------------------------------------------------------
    # Court corner selection
    # ------------------------------------------------------------------

    @staticmethod
    def select_court_corners(first_frame, timeout_sec=30, save_path=None):
        """
        Click order: top-left → bottom-left → bottom-right → top-right
        Keys: Esc/Space/Enter — skip | R — reset
        Saves to JSON if save_path given. Returns np.float32 (4,2) or None.
        """
        LABELS = ["1: top-left", "2: bottom-left", "3: bottom-right", "4: top-right"]
        COLORS = [(0, 255, 255), (0, 200, 255), (0, 140, 255), (0, 80, 255)]

        pts = []
        img = first_frame.copy()
        win = "Select 4 court corners | Esc/Space - skip | R - reset"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        def on_mouse(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
                pts.append((x, y))
                if len(pts) == 4 and save_path is not None:
                    Path(save_path).write_text(json.dumps(pts))
                    print(f"[OK] Saved court corners to {save_path}")

        cv2.setMouseCallback(win, on_mouse)
        deadline = time.time() + timeout_sec

        while True:
            remaining = max(0, deadline - time.time())
            disp = img.copy()
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 52), (0, 0, 0), -1)
            cv2.addWeighted(disp, 0.5, img, 0.5, 0, disp)

            if len(pts) < 4:
                cv2.putText(disp, f"Click {LABELS[len(pts)]}  ({remaining:.0f}s)",
                            (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                            (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(disp, "Done! Press any key to start...",
                            (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                            (0, 255, 120), 2, cv2.LINE_AA)

            for i, (px, py) in enumerate(pts):
                cv2.circle(disp, (px, py), 8, COLORS[i], -1, cv2.LINE_AA)
                cv2.circle(disp, (px, py), 8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(disp, str(i + 1), (px + 10, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if len(pts) >= 2:
                cv2.polylines(disp, [np.array(pts, dtype=np.int32)],
                              isClosed=(len(pts) == 4),
                              color=(200, 200, 0), thickness=2, lineType=cv2.LINE_AA)

            cv2.imshow(win, disp)
            key = cv2.waitKey(30) & 0xFF

            if key in (27, 32, 13):
                cv2.destroyWindow(win)
                return None
            if key in (ord('r'), ord('R')):
                pts.clear()
            if len(pts) == 4 and key != 255:
                break
            if remaining == 0:
                print("[!] Timeout — zone analysis requires court corners, skipping")
                cv2.destroyWindow(win)
                return None

        cv2.destroyWindow(win)
        return np.array(pts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Zone geometry
    # ------------------------------------------------------------------

    @staticmethod
    def compute_zone_polygons(corners):
        """Split court into 4 equal horizontal zones (left to right)."""
        tl, bl, br, tr = corners
        fracs = [0.0, 0.25, 0.50, 0.75, 1.0]
        zones = []
        for i in range(4):
            t0, t1 = fracs[i], fracs[i + 1]
            top_l = tl + t0 * (tr - tl)
            top_r = tl + t1 * (tr - tl)
            bot_l = bl + t0 * (br - bl)
            bot_r = bl + t1 * (br - bl)
            zones.append(np.array([top_l, bot_l, bot_r, top_r], dtype=np.int32))
        return zones

    @staticmethod
    def ball_zone_index(cx, cy, homography):
        """Returns zone index 0-3 (left to right) or None if outside court."""
        if homography is None:
            return None
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        norm = cv2.perspectiveTransform(pt, homography)[0][0]
        nx, ny = float(norm[0]), float(norm[1])
        if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
            return min(3, int(nx * 4))
        return None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_zones(frame, zone_polys, zone_counts):
        """Semi-transparent zone overlay with accumulated ball-presence percentages."""
        total = sum(zone_counts)

        overlay = frame.copy()
        for i, poly in enumerate(zone_polys):
            cv2.fillPoly(overlay, [poly], ZONE_COLORS_BGR[i])
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

        for poly in zone_polys:
            cv2.polylines(frame, [poly], isClosed=True,
                          color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        for i, poly in enumerate(zone_polys):
            cx_z = int(poly[:, 0].mean())
            cy_z = int(poly[:, 1].mean())
            pct  = zone_counts[i] / total * 100 if total > 0 else 0.0
            label = f"{pct:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(frame, label,
                        (cx_z - tw // 2, cy_z + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _draw_stats_bar(frame, zone_counts):
        """Bottom bar: Zone 1-4 with ball-presence count and percentage."""
        fh, fw = frame.shape[:2]
        bar_y  = fh - STATS_BAR_H

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (fw, fh), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        total  = sum(zone_counts)
        zone_w = fw // 4

        for i in range(4):
            x0   = i * zone_w
            x1   = x0 + zone_w if i < 3 else fw
            col  = ZONE_COLORS_BGR[i]
            cx_z = (x0 + x1) // 2
            pct  = zone_counts[i] / total * 100 if total > 0 else 0.0

            cv2.rectangle(frame, (x0, bar_y), (x1, bar_y + 5), col, -1)

            if i > 0:
                cv2.line(frame, (x0, bar_y + 5), (x0, fh), (70, 70, 70), 1)

            zone_label = f"Zone {i + 1}"
            (tw, _), _ = cv2.getTextSize(zone_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.putText(frame, zone_label,
                        (cx_z - tw // 2, bar_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

            pct_str = f"{pct:.1f}%"
            (tw3, _), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            cv2.putText(frame, pct_str,
                        (cx_z - tw3 // 2, bar_y + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def detect_ball(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(self.weights)

        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read first frame")

        # Load saved corners or select interactively
        corners_file = self.out_dir / "court_corners.json"
        if corners_file.exists():
            court_corners = np.array(json.loads(corners_file.read_text()), dtype=np.float32)
            print(f"[OK] Loaded court corners from {corners_file}")
        else:
            court_corners = self.select_court_corners(first_frame, save_path=corners_file)

        homography = None
        zone_polys = []
        if court_corners is not None:
            dst_norm = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
            homography, _ = cv2.findHomography(court_corners, dst_norm)
            zone_polys    = self.compute_zone_polygons(court_corners)
            print(f"[OK] Homography set — 4 zones active (ball class index = {self.ball_class})")
        else:
            print("[!] No corners selected — zone analysis disabled")

        out_path = self.out_dir / "ball_result.mp4"
        writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        csv_path = self.out_dir / "ball_detections.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["frame", "x1", "y1", "x2", "y2", "cx", "cy", "conf", "zone"])

        zone_counts       = [0, 0, 0, 0]
        ball_positions    = []
        frames_to_process = [first_frame]
        frame_id          = 0

        while True:
            if frames_to_process:
                frame = frames_to_process.pop(0)
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            results = model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                    verbose=False, classes=[self.ball_class])

            # Use highest-confidence detection for zone counting
            detections = []
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf_val = float(b.conf[0])
                    cx, cy   = (x1 + x2) / 2, (y1 + y2) / 2
                    detections.append((conf_val, x1, y1, x2, y2, cx, cy))
            detections.sort(key=lambda d: d[0], reverse=True)

            rows    = []
            counted = False
            for conf_val, x1, y1, x2, y2, cx, cy in detections:
                zone_idx = self.ball_zone_index(cx, cy, homography)
                if zone_idx is not None and not counted:
                    zone_counts[zone_idx] += 1
                    counted = True
                    ball_positions.append((int(cx), int(cy)))
                rows.append([frame_id, int(x1), int(y1), int(x2), int(y2),
                              int(cx), int(cy), round(conf_val, 3),
                              (zone_idx + 1) if zone_idx is not None else -1])

            if zone_polys:
                self._draw_zones(frame, zone_polys, zone_counts)
            self._draw_stats_bar(frame, zone_counts)

            if rows:
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(rows)

            writer.write(frame)
            cv2.imshow("YOLO Ball — Zone Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_id += 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        print(f"\n[OK] Video: {out_path}")
        print(f"[OK] CSV:   {csv_path}")
        print("\n=== Zone summary ===")
        total = sum(zone_counts)
        for i, cnt in enumerate(zone_counts):
            pct = cnt / total * 100 if total > 0 else 0.0
            print(f"  Zone {i + 1}: {cnt:6d} frames  ({pct:.1f}%)")

        # Generate pitch heatmap if pitch.png exists
        pitch_png_path = Path(__file__).parent / "pitch.png"
        if ball_positions and homography is not None and pitch_png_path.exists():
            self._save_pitch_heatmap(ball_positions, homography, pitch_png_path)

    def _save_pitch_heatmap(self, ball_positions, homography, pitch_png_path):
        try:
            from PIL import Image
            from heat_map.heatmap import Heatmapper
        except ImportError:
            print("[!] PIL or heat_map not installed — skipping pitch heatmap")
            return

        heatmap_path = self.out_dir / "pitch_heatmap.jpg"
        base_img = Image.open(str(pitch_png_path))
        pw, ph_img = base_img.size

        pts      = np.array([[p] for p in ball_positions], dtype=np.float32)
        norm_pts = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
        pitch_pts = [
            (int(nx * pw), int(ny * ph_img))
            for nx, ny in norm_pts
            if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0
        ]

        if pitch_pts:
            heatmapper = Heatmapper(point_diameter=40, point_strength=0.05, opacity=0.65)
            result = heatmapper.heatmap_on_img(pitch_pts, base_img)
            result.convert("RGB").save(str(heatmap_path), "JPEG", quality=92)
            print(f"[OK] Pitch heatmap: {heatmap_path}")
        else:
            print("[!] No ball positions on pitch after filtering — heatmap skipped")
