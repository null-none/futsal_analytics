#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FutsalAnalytics — single entry point for the entire pipeline.

Examples:
    # Model training
    python main.py train --model yolo26s.pt --data data.yaml --epochs 50

    # Player detection (heatmap, radar, convex hull)
    python main.py detect --weights best.pt --source input.mp4

    # Speed & sprint tracking (two teams)
    python main.py speed --weights best.pt --source input.mp4 --court_w_px 950

    # Ball zone analysis
    python main.py ball --weights best.pt --source input.mp4 --ball_class 1
"""

import argparse

from trainer import YOLOTrainer
from player_detector import PlayerDetector
from speed_tracker import SpeedTracker
from detect_ball import BallDetector


class FutsalAnalytics(YOLOTrainer, PlayerDetector, SpeedTracker, BallDetector):
    """
    Combines YOLOTrainer, PlayerDetector, SpeedTracker and BallDetector into one class.
    All parameters are passed through the constructor and stored as attributes.
    """

    def __init__(
        self,
        # common
        weights: str = "best.pt",
        source: str = "input.mp4",
        out_dir: str = "out",
        conf: float = 0.25,
        imgsz: int = 1280,
        # training
        model_path: str = "yolo26s.pt",
        data_yaml: str = "data.yaml",
        epochs: int = 50,
        batch: int = 8,
        workers: int = 4,
        # detection
        dot_radius: int = 5,
        # speed
        court_w_m: float = 40.0,
        court_h_m: float = 20.0,
        court_w_px: float = None,
        smooth: int = 15,
        team1_name: str = "TEAM 1",
        team2_name: str = "TEAM 2",
        team1_color: str = "blue",
        team2_color: str = "red",
        # ball
        ball_class: int = 1,
    ):
        YOLOTrainer.__init__(
            self,
            model_path=model_path,
            data_yaml=data_yaml,
            imgsz=imgsz,
            epochs=epochs,
            batch=batch,
            workers=workers,
        )
        PlayerDetector.__init__(
            self,
            weights=weights,
            source=source,
            out_dir=out_dir,
            conf=conf,
            imgsz=imgsz,
            dot_radius=dot_radius,
        )
        SpeedTracker.__init__(
            self,
            weights=weights,
            source=source,
            out_dir=out_dir,
            conf=conf,
            imgsz=imgsz,
            court_w_m=court_w_m,
            court_h_m=court_h_m,
            court_w_px=court_w_px,
            smooth=smooth,
            team1_name=team1_name,
            team2_name=team2_name,
            team1_color=team1_color,
            team2_color=team2_color,
        )
        BallDetector.__init__(
            self,
            weights=weights,
            source=source,
            out_dir=out_dir,
            conf=conf,
            imgsz=imgsz,
            ball_class=ball_class,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Futsal Analytics — YOLO-based player analysis")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- train ---
    t = sub.add_parser("train", help="Train YOLO model")
    t.add_argument("--model",   default="yolo26s.pt")
    t.add_argument("--data",    default="data.yaml")
    t.add_argument("--imgsz",   type=int,   default=1280)
    t.add_argument("--epochs",  type=int,   default=50)
    t.add_argument("--batch",   type=int,   default=8)
    t.add_argument("--workers", type=int,   default=4)

    # --- detect ---
    d = sub.add_parser("detect", help="Player detection (heatmap, radar, convex hull)")
    d.add_argument("--weights",    required=True)
    d.add_argument("--source",     required=True)
    d.add_argument("--out_dir",    default="out")
    d.add_argument("--conf",       type=float, default=0.25)
    d.add_argument("--imgsz",      type=int,   default=1280)
    d.add_argument("--dot_radius", type=int,   default=5)

    # --- speed ---
    s = sub.add_parser("speed", help="Speed and sprint tracking (two teams)")
    s.add_argument("--weights",      required=True)
    s.add_argument("--source",       required=True)
    s.add_argument("--out_dir",      default="out_speed")
    s.add_argument("--conf",         type=float, default=0.25)
    s.add_argument("--imgsz",        type=int,   default=1280)
    s.add_argument("--court_w_m",    type=float, default=40.0)
    s.add_argument("--court_h_m",    type=float, default=20.0)
    s.add_argument("--court_w_px",   type=float, default=None)
    s.add_argument("--smooth",       type=int,   default=15)
    s.add_argument("--team1_name",   default="TEAM 1")
    s.add_argument("--team2_name",   default="TEAM 2")
    s.add_argument("--team1_color",  default="blue")
    s.add_argument("--team2_color",  default="red")

    # --- ball ---
    b = sub.add_parser("ball", help="Ball detection with 4-zone court analysis")
    b.add_argument("--weights",    required=True)
    b.add_argument("--source",     required=True)
    b.add_argument("--out_dir",    default="out_ball")
    b.add_argument("--conf",       type=float, default=0.25)
    b.add_argument("--imgsz",      type=int,   default=1280)
    b.add_argument("--ball_class", type=int,   default=1,
                   help="YOLO class index for the ball (0-based)")

    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == "train":
        fa = FutsalAnalytics(
            model_path=args.model,
            data_yaml=args.data,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
        )
        fa.train()

    elif args.cmd == "detect":
        fa = FutsalAnalytics(
            weights=args.weights,
            source=args.source,
            out_dir=args.out_dir,
            conf=args.conf,
            imgsz=args.imgsz,
            dot_radius=args.dot_radius,
        )
        fa.detect()

    elif args.cmd == "speed":
        fa = FutsalAnalytics(
            weights=args.weights,
            source=args.source,
            out_dir=args.out_dir,
            conf=args.conf,
            imgsz=args.imgsz,
            court_w_m=args.court_w_m,
            court_h_m=args.court_h_m,
            court_w_px=args.court_w_px,
            smooth=args.smooth,
            team1_name=args.team1_name,
            team2_name=args.team2_name,
            team1_color=args.team1_color,
            team2_color=args.team2_color,
        )
        fa.track()

    elif args.cmd == "ball":
        fa = FutsalAnalytics(
            weights=args.weights,
            source=args.source,
            out_dir=args.out_dir,
            conf=args.conf,
            imgsz=args.imgsz,
            ball_class=args.ball_class,
        )
        fa.detect_ball()


if __name__ == "__main__":
    main()
