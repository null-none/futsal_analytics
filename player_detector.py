#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import csv
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

RADAR_W, RADAR_H = 240, 130
HEAT_W, HEAT_H = 220, 140
PAD = 10
_RADAR_TITLE_H = 18


class PlayerDetector:
    def __init__(
        self,
        weights: str = "best.pt",
        source: str = "input.mp4",
        out_dir: str = "out",
        conf: float = 0.25,
        imgsz: int = 1280,
        dot_radius: int = 5,
    ):
        self.weights = weights
        self.source = source
        self.out_dir = Path(out_dir)
        self.conf = conf
        self.imgsz = imgsz
        self.dot_radius = dot_radius

    # ------------------------------------------------------------------
    # Court corner selection
    # ------------------------------------------------------------------

    @staticmethod
    def select_court_corners(first_frame, timeout_sec=12):
        LABELS = ["1: top-left", "2: bottom-left", "3: bottom-right", "4: top-right"]
        COLORS = [(0, 255, 255), (0, 200, 255), (0, 140, 255), (0, 80, 255)]

        pts = []
        img = first_frame.copy()
        win = "Select 4 court corners | Esc/Space - skip | R - reset"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        def on_mouse(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
                pts.append((x, y))

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
                cv2.putText(disp, "Done! Press any key...",
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
                print("[!] Timeout — radar running without homography")
                cv2.destroyWindow(win)
                return None

        cv2.destroyWindow(win)
        return np.array(pts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_center_dot(frame, cx, cy, r=5):
        h_img, w_img = frame.shape[:2]
        cx_i = max(0, min(w_img - 1, int(cx)))
        cy_i = max(0, min(h_img - 1, int(cy)))
        cv2.circle(frame, (cx_i, cy_i), int(r), (0, 0, 255), -1, cv2.LINE_AA)

    @staticmethod
    def _draw_convex_hull(frame, points):
        if len(points) < 3:
            return
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [hull], (0, 0, 255))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [hull], isClosed=True,
                      color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_radar(frame, centers, frame_w, frame_h, homography):
        x0 = frame_w - RADAR_W - PAD
        y0 = PAD
        rx0 = x0 + 2
        ry0 = y0 + _RADAR_TITLE_H
        rw = RADAR_W - 4
        rh = RADAR_H - _RADAR_TITLE_H - 2
        cx_mid = rx0 + rw // 2
        cy_mid = ry0 + rh // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (rx0, ry0), (rx0 + rw, ry0 + rh), (120, 60, 10), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        lc = (220, 220, 220)
        cv2.rectangle(frame, (rx0, ry0), (rx0 + rw, ry0 + rh), lc, 1)
        cv2.line(frame, (cx_mid, ry0), (cx_mid, ry0 + rh), lc, 1)
        r_circle = max(5, int(rw * 0.075))
        cv2.circle(frame, (cx_mid, cy_mid), r_circle, lc, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx_mid, cy_mid), 2, lc, -1)
        r_goal = max(8, int(rw * 0.14))
        cv2.ellipse(frame, (rx0, cy_mid), (r_goal, r_goal), 0, -90, 90, lc, 1, cv2.LINE_AA)
        cv2.ellipse(frame, (rx0 + rw, cy_mid), (r_goal, r_goal), 0, 90, 270, lc, 1, cv2.LINE_AA)

        cv2.rectangle(frame, (x0, y0), (x0 + RADAR_W, y0 + RADAR_H), (180, 180, 180), 1)
        cv2.putText(frame, "RADAR", (x0 + 6, y0 + _RADAR_TITLE_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"n={len(centers)}", (x0 + RADAR_W - 46, y0 + _RADAR_TITLE_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

        if not centers:
            return

        src_pts = np.array(centers, dtype=np.float32).reshape(-1, 1, 2)
        if homography is not None:
            dst = cv2.perspectiveTransform(src_pts, homography)
            radar_pts = []
            for p in dst:
                nx, ny = float(p[0][0]), float(p[0][1])
                px = rx0 + int(np.clip(nx, 0, 1) * rw)
                py = ry0 + int(np.clip(ny, 0, 1) * rh)
                radar_pts.append((px, py))
        else:
            radar_pts = []
            for cx, cy in centers:
                px = rx0 + int(cx / frame_w * rw)
                py = ry0 + int(cy / frame_h * rh)
                px = max(rx0, min(rx0 + rw - 1, px))
                py = max(ry0, min(ry0 + rh - 1, py))
                radar_pts.append((px, py))

        for px, py in radar_pts:
            cv2.circle(frame, (px, py), 4, (0, 60, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 4, (255, 255, 255), 1, cv2.LINE_AA)

        if len(radar_pts) >= 3:
            pts = np.array(radar_pts, dtype=np.int32).reshape(-1, 1, 2)
            hull = cv2.convexHull(pts)
            cv2.polylines(frame, [hull], isClosed=True,
                          color=(0, 60, 255), thickness=1, lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_heatmap(frame, heat_acc):
        x0, y0 = PAD, PAD
        h_small = cv2.resize(heat_acc, (HEAT_W, HEAT_H))
        max_val = h_small.max()
        if max_val > 0:
            h_norm = (h_small / max_val * 255).astype(np.uint8)
        else:
            h_norm = np.zeros((HEAT_H, HEAT_W), dtype=np.uint8)
        h_color = cv2.applyColorMap(h_norm, cv2.COLORMAP_JET)
        roi = frame[y0:y0 + HEAT_H, x0:x0 + HEAT_W]
        cv2.addWeighted(h_color, 0.6, roi, 0.4, 0, roi)
        frame[y0:y0 + HEAT_H, x0:x0 + HEAT_W] = roi
        cv2.rectangle(frame, (x0, y0), (x0 + HEAT_W, y0 + HEAT_H), (180, 180, 180), 1)
        cv2.putText(frame, "HEATMAP", (x0 + 6, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def detect(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(self.weights)

        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read first frame")

        court_corners = self.select_court_corners(first_frame)
        homography = None
        if court_corners is not None:
            dst_norm = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
            homography, _ = cv2.findHomography(court_corners, dst_norm)
            print("[OK] Homography computed from 4 court corners")
        else:
            print("[!] No corners selected — radar using simple normalization")

        out_path = self.out_dir / "result.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        csv_path = self.out_dir / "detections.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["frame", "x1", "y1", "x2", "y2", "cx", "cy", "conf", "cls", "label"])

        heat_acc = np.zeros((h, w), dtype=np.float32)
        blob_r = max(w, h) // 12
        frames_to_process = [first_frame]

        frame_id = 0
        while True:
            if frames_to_process:
                frame = frames_to_process.pop(0)
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            results = model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                    verbose=False, classes=[0])
            rows = []
            centers = []

            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf_val = float(b.conf[0])
                    cls_id = int(b.cls[0])
                    label = r.names.get(cls_id, str(cls_id))
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    self._draw_center_dot(frame, cx, cy, r=self.dot_radius)
                    centers.append((cx, cy))

                    cx_i = max(0, min(w - 1, int(cx)))
                    cy_i = max(0, min(h - 1, int(cy)))
                    cv2.circle(heat_acc, (cx_i, cy_i), blob_r, 1.0, -1)

                    rows.append([frame_id, int(x1), int(y1), int(x2), int(y2),
                                  int(cx), int(cy), round(conf_val, 3), cls_id, label])

            heat_acc *= 0.97
            heat_acc = cv2.GaussianBlur(heat_acc, (0, 0), sigmaX=max(w, h) // 30)

            if len(centers) >= 3:
                self._draw_convex_hull(frame, centers)

            self._draw_heatmap(frame, heat_acc)
            self._draw_radar(frame, centers, w, h, homography)

            if rows:
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(rows)

            writer.write(frame)
            cv2.imshow("YOLO Players", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_id += 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"[OK] Video: {out_path}")
        print(f"[OK] CSV:   {csv_path}")
