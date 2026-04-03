#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import csv
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO

NAMED_COLORS = {
    "red":     (  0,   0, 220),
    "blue":    (220, 100,   0),
    "green":   (  0, 200,   0),
    "yellow":  (  0, 220, 220),
    "cyan":    (220, 220,   0),
    "magenta": (220,   0, 220),
    "orange":  (  0, 165, 255),
    "white":   (255, 255, 255),
    "purple":  (200,   0, 200),
    "lime":    (  0, 255,   0),
    "pink":    (147,  20, 255),
    "teal":    (128, 128,   0),
    "black":   ( 30,  30,  30),
}


class SpeedTracker:
    def __init__(
        self,
        weights: str = "best.pt",
        source: str = "input.mp4",
        out_dir: str = "out_speed",
        conf: float = 0.25,
        imgsz: int = 1280,
        court_w_m: float = 40.0,
        court_h_m: float = 20.0,
        court_w_px: float = None,
        smooth: int = 15,
        team1_name: str = "TEAM 1",
        team2_name: str = "TEAM 2",
        team1_color: str = "blue",
        team2_color: str = "red",
    ):
        self.weights = weights
        self.source = source
        self.out_dir = Path(out_dir)
        self.conf = conf
        self.imgsz = imgsz
        self.court_w_m = court_w_m
        self.court_h_m = court_h_m
        self.court_w_px = court_w_px
        self.smooth = smooth
        self.team_names = [team1_name, team2_name]
        self.team_colors = [self.parse_color(team1_color), self.parse_color(team2_color)]

    # ------------------------------------------------------------------
    # Color parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_color(s: str) -> tuple:
        """Accepts a color name, #RRGGBB, or R,G,B — returns BGR tuple."""
        s = s.strip().lower()
        if s in NAMED_COLORS:
            return NAMED_COLORS[s]
        if s.startswith("#") and len(s) == 7:
            r = int(s[1:3], 16)
            g = int(s[3:5], 16)
            b = int(s[5:7], 16)
            return (b, g, r)
        parts = s.split(",")
        if len(parts) == 3:
            r, g, b = int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())
            return (b, g, r)
        raise ValueError(
            f"Unknown color format: '{s}'. "
            f"Use a name ({'/'.join(NAMED_COLORS)}), #RRGGBB, or R,G,B"
        )

    # ------------------------------------------------------------------
    # Court corner selection
    # ------------------------------------------------------------------

    @staticmethod
    def select_court_corners(first_frame, timeout_sec=12, save_path=None):
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
                print("[!] Timeout — speed will use simple scale (no court filter)")
                cv2.destroyWindow(win)
                return None

        cv2.destroyWindow(win)
        return np.array(pts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_top_bar(frame, t1_name, t2_name, c1, c2):
        """Top bar: TEAM 1  VS  TEAM 2."""
        fw = frame.shape[1]
        bar_h = 50
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (fw, bar_h), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
        cv2.line(frame, (0, bar_h), (fw, bar_h), (60, 60, 60), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"  {t1_name}", (11, 33), font, 0.65, (0, 0, 0), 4)
        cv2.putText(frame, f"  {t1_name}", (10, 32), font, 0.65, c1,        2)

        vs = "VS"
        (vw, _), _ = cv2.getTextSize(vs, font, 0.55, 2)
        cv2.putText(frame, vs, ((fw - vw) // 2 + 1, 33), font, 0.55, (0,   0,   0),   3)
        cv2.putText(frame, vs, ((fw - vw) // 2,     32), font, 0.55, (160, 160, 160), 2)

        (t2w, _), _ = cv2.getTextSize(f"{t2_name}  ", font, 0.65, 2)
        cv2.putText(frame, f"{t2_name}  ", (fw - t2w - 11, 33), font, 0.65, (0, 0, 0), 4)
        cv2.putText(frame, f"{t2_name}  ", (fw - t2w - 10, 32), font, 0.65, c2,        2)

    @staticmethod
    def _draw_stat_panel(frame, x, y, pw, ph, title, total_km, sprint_km, color):
        """Semi-transparent team stats panel (distance + sprint distance)."""
        ov = frame.copy()
        cv2.rectangle(ov, (x, y), (x + pw, y + ph), (12, 12, 12), -1)
        cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

        cv2.rectangle(frame, (x, y), (x + pw, y + 4), color, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, _), _ = cv2.getTextSize(title, font, 0.52, 2)
        tx = x + (pw - tw) // 2
        cv2.putText(frame, title, (tx + 1, y + 24), font, 0.52, (0,   0,   0), 3)
        cv2.putText(frame, title, (tx,     y + 23), font, 0.52, color,          2)

        cv2.line(frame, (x + 8, y + 30), (x + pw - 8, y + 30), (55, 55, 55), 1)

        row1 = f"DIST   {total_km:.3f} km"
        (rw1, _), _ = cv2.getTextSize(row1, font, 0.48, 1)
        cv2.putText(frame, row1, (x + (pw - rw1) // 2 + 1, y + 52), font, 0.48, (0,   0,   0),   2)
        cv2.putText(frame, row1, (x + (pw - rw1) // 2,     y + 51), font, 0.48, (210, 210, 210), 1)

        row2 = f"SPR    {sprint_km:.3f} km"
        (rw2, _), _ = cv2.getTextSize(row2, font, 0.48, 1)
        cv2.putText(frame, row2, (x + (pw - rw2) // 2 + 1, y + 72), font, 0.48, (0,   0,   0),   2)
        cv2.putText(frame, row2, (x + (pw - rw2) // 2,     y + 71), font, 0.48, (210, 210, 210), 1)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def track(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(self.weights)

        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")

        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Load saved corners or select interactively
        corners_file  = self.out_dir / "court_corners.json"
        court_corners = None
        if corners_file.exists():
            court_corners = np.array(json.loads(corners_file.read_text()), dtype=np.float32)
            print(f"[OK] Loaded court corners from {corners_file}")
        else:
            ok, first_frame = cap.read()
            if not ok:
                raise RuntimeError("Cannot read first frame")
            court_corners = self.select_court_corners(first_frame, save_path=corners_file)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Build homography pixel → real-world metres
        homography    = None
        court_poly    = None
        fallback_scale = self.court_w_m / self.court_w_px if self.court_w_px else None

        if court_corners is not None:
            dst_real = np.array([
                [0,              0             ],
                [0,              self.court_h_m],
                [self.court_w_m, self.court_h_m],
                [self.court_w_m, 0             ],
            ], dtype=np.float32)
            homography, _ = cv2.findHomography(court_corners, dst_real)
            court_poly    = court_corners.astype(np.int32)
            print(f"[OK] Homography built — {self.court_w_m}m × {self.court_h_m}m")
        elif fallback_scale:
            print(f"[!] No corners — simple scale {fallback_scale:.5f} m/px")
        else:
            fallback_scale = 1.0
            print("[!] No corners and no court_w_px — speed uncalibrated (px/s)")

        out_path = self.out_dir / "result_speed.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h))

        csv_path = self.out_dir / "speeds.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["frame", "player_id", "team", "cx", "cy", "speed_ms", "max_speed_ms"])

        history       = defaultdict(lambda: deque(maxlen=self.smooth + 1))
        max_speeds    = defaultdict(float)
        total_dist    = defaultdict(float)
        sprint_dist   = defaultdict(float)
        sprint_active = {}
        all_sprint_durations = []
        last_pos      = {}
        player_class  = {}

        team_dist_m   = [0.0, 0.0]
        team_sprint_m = [0.0, 0.0]

        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.track(frame, imgsz=self.imgsz, conf=self.conf,
                                   verbose=False, persist=True)

            csv_rows = []
            for r in results:
                if r.boxes.id is None:
                    continue
                for b, pid in zip(r.boxes, r.boxes.id.int().tolist()):
                    cls = int(b.cls[0]) if b.cls is not None else 0
                    if cls > 1:
                        continue
                    player_class[pid] = cls

                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    # Filter: skip detections outside court boundary
                    if court_poly is not None:
                        if cv2.pointPolygonTest(court_poly, (cx, cy), False) < 0:
                            continue

                    # Pixel → real-world coords (metres)
                    if homography is not None:
                        pt    = np.array([[[cx, cy]]], dtype=np.float32)
                        out_pt = cv2.perspectiveTransform(pt, homography)[0][0]
                        pos_m = (float(out_pt[0]), float(out_pt[1]))
                    else:
                        s     = fallback_scale
                        pos_m = (cx * s, cy * s)

                    history[pid].append(pos_m)
                    hist = list(history[pid])
                    if len(hist) >= 2:
                        seg_speeds = []
                        for i in range(1, len(hist)):
                            dx = hist[i][0] - hist[i - 1][0]
                            dy = hist[i][1] - hist[i - 1][1]
                            seg_speeds.append(np.sqrt(dx ** 2 + dy ** 2) * fps)
                        speed = float(np.mean(seg_speeds))
                    else:
                        speed = 0.0

                    if speed > max_speeds[pid]:
                        max_speeds[pid] = speed

                    color = (0, 0, 255) if speed > 5.0 else self.team_colors[cls]

                    if pid in last_pos:
                        pm   = last_pos[pid]
                        step = np.sqrt((pos_m[0] - pm[0]) ** 2 + (pos_m[1] - pm[1]) ** 2)
                        total_dist[pid]    += step
                        team_dist_m[cls]   += step
                        if speed > 5.0:
                            sprint_dist[pid]   += step
                            team_sprint_m[cls] += step
                    last_pos[pid] = pos_m

                    if speed > 5.0:
                        if pid not in sprint_active:
                            sprint_active[pid] = frame_id
                    else:
                        if pid in sprint_active:
                            dur = (frame_id - sprint_active.pop(pid)) / fps
                            all_sprint_durations.append(dur)

                    # FIFA-style shadow ellipse under player feet
                    box_w = x2 - x1
                    rx = max(int(box_w * 0.45), 20)
                    ry = max(int(rx * 0.28), 7)
                    ov_f = frame.copy()
                    cv2.ellipse(ov_f, (int(cx), y2), (rx, ry), 0, 0, 180, color, -1)
                    cv2.addWeighted(ov_f, 0.35, frame, 0.65, 0, frame)
                    cv2.ellipse(frame, (int(cx), y2), (rx, ry), 0, 0, 180, color, 2)

                    # Speed label above bounding box
                    lbl = f"{speed:.1f} m/s"
                    (lw_, lh_), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
                    cv2.rectangle(frame, (x1, y1 - lh_ - 10), (x1 + lw_ + 4, y1), color, -1)
                    cv2.putText(frame, lbl, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)

                    csv_rows.append([frame_id, pid, cls, int(cx), int(cy),
                                      round(speed, 3), round(max_speeds[pid], 3)])

            # HUD
            self._draw_top_bar(frame,
                               self.team_names[0], self.team_names[1],
                               self.team_colors[0], self.team_colors[1])

            pw, ph, mg = 220, 88, 12
            py_panel = vid_h - ph - mg
            self._draw_stat_panel(frame, mg, py_panel, pw, ph,
                                  self.team_names[0],
                                  team_dist_m[0] / 1000.0, team_sprint_m[0] / 1000.0,
                                  self.team_colors[0])
            self._draw_stat_panel(frame, vid_w - pw - mg, py_panel, pw, ph,
                                  self.team_names[1],
                                  team_dist_m[1] / 1000.0, team_sprint_m[1] / 1000.0,
                                  self.team_colors[1])

            writer.write(frame)

            if csv_rows:
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(csv_rows)

            cv2.imshow("Speed Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_id += 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        # Close any open sprints at end of video
        for pid, start_frame in sprint_active.items():
            all_sprint_durations.append((frame_id - start_frame) / fps)

        return self._build_summary(
            out_path, csv_path,
            team_dist_m, team_sprint_m,
            all_sprint_durations,
        )

    def _build_summary(self, out_path, csv_path, team_dist_m, team_sprint_m, sprint_durations):
        t1_km  = team_dist_m[0] / 1000.0
        t1_spr = team_sprint_m[0] / 1000.0
        t2_km  = team_dist_m[1] / 1000.0
        t2_spr = team_sprint_m[1] / 1000.0

        max_spr = max(sprint_durations, default=0.0)
        spr_5s  = sum(1 for d in sprint_durations if d >= 5)
        spr_7s  = sum(1 for d in sprint_durations if d >= 7)
        spr_9s  = sum(1 for d in sprint_durations if d >= 9)

        lines = [
            "=== SUMMARY ===",
            "",
            f"{self.team_names[0]}:",
            f"  Distance:              {t1_km:.3f} km",
            f"  Sprint (>5 m/s):       {t1_spr:.3f} km",
            "",
            f"{self.team_names[1]}:",
            f"  Distance:              {t2_km:.3f} km",
            f"  Sprint (>5 m/s):       {t2_spr:.3f} km",
            "",
            "TOTAL:",
            f"  Total run:             {(t1_km + t2_km):.3f} km",
            f"  Max sprint duration:   {max_spr:.1f} s",
            f"  Sprints >= 5 s:        {spr_5s}",
            f"  Sprints >= 7 s:        {spr_7s}",
            f"  Sprints >= 9 s:        {spr_9s}",
        ]

        summary_path = self.out_dir / "summary.txt"
        summary_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"[OK] Video:   {out_path}")
        print(f"[OK] CSV:     {csv_path}")
        print()
        for line in lines:
            print(line)
        print(f"[OK] Summary: {summary_path}")

        return lines
