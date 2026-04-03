# Futsal Analytics

YOLO-based analytics for futsal: player detection, heatmap, radar, speed tracking, sprint stats, and ball zone analysis.

## Install

```bash
pip install -r requirements.txt
```

## Usage

### Train a model

```bash
python main.py train \
    --model yolo26s.pt \
    --data data.yaml \
    --epochs 50 \
    --batch 8
```

---

### Detect players — heatmap + radar + convex hull

```bash
python main.py detect \
    --weights best.pt \
    --source input.mp4 \
    --out_dir out \
    --conf 0.25 \
    --dot_radius 1
```

On first run you will be asked to click **4 court corners** for the radar (top-left → bottom-left → bottom-right → top-right).  
Corners are saved to `out/court_corners.json` and reloaded automatically on subsequent runs.  
Press `Esc` / `Space` to skip corner selection. Press `Q` to stop playback.

**Output:** `out/result.mp4`, `out/detections.csv`, `out/court_corners.json`

---

### Track speed & sprints — two teams

```bash
python main.py speed \
    --weights best.pt \
    --source input.mp4 \
    --out_dir out_speed \
    --court_w_m 40 \
    --court_h_m 20 \
    --court_w_px 950 \
    --team1_name "Cherno More" --team1_color green \
    --team2_name "Eter"        --team2_color purple
```

| Flag | Default | Description |
|------|---------|-------------|
| `--court_w_m` | `40.0` | Real court width in metres |
| `--court_h_m` | `20.0` | Real court height in metres |
| `--court_w_px` | `None` | Fallback pixel width if no corners selected |
| `--smooth` | `15` | Speed smoothing window (frames) |
| `--team1_color` | `blue` | Color name, `#RRGGBB`, or `R,G,B` |
| `--team2_color` | `red` | Color name, `#RRGGBB`, or `R,G,B` |

Available color names: `red`, `blue`, `green`, `yellow`, `cyan`, `magenta`, `orange`, `white`, `purple`, `lime`, `pink`, `teal`, `black`.

Court corners are loaded from `out_speed/court_corners.json` if it exists, otherwise selected interactively.  
Players outside the court boundary are automatically filtered out.  
Sprint threshold: **5 m/s**.

**Output:** `out_speed/result_speed.mp4`, `out_speed/speeds.csv`, `out_speed/summary.txt`

---

### Ball detection — 4-zone court analysis

```bash
python main.py ball \
    --weights best.pt \
    --source input.mp4 \
    --out_dir out_ball \
    --ball_class 1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--ball_class` | `1` | YOLO class index for the ball (0-based) |

The court is divided into **4 equal horizontal zones** (left → right).  
Each frame the ball's zone is recorded and the running percentage is shown as an overlay and bottom bar.

If `pitch.png` is present in the project directory and `heat_map` is installed, a **pitch heatmap** is saved at the end.

**Output:** `out_ball/ball_result.mp4`, `out_ball/ball_detections.csv`, `out_ball/court_corners.json`  
Optional: `out_ball/pitch_heatmap.jpg`

---

## Project structure

```
futsal_analytics/
├── main.py            # FutsalAnalytics class (inherits all three) + CLI
├── trainer.py         # YOLOTrainer
├── player_detector.py # PlayerDetector  — heatmap, radar, convex hull
├── speed_tracker.py   # SpeedTracker    — two teams, homography, sprints
├── detect_ball.py     # BallDetector    — 4-zone analysis, pitch heatmap
├── pitch.png          # (optional) top-down court image for heatmap
└── requirements.txt
```

## Use as a library

```python
from player_detector import PlayerDetector
from speed_tracker import SpeedTracker
from detect_ball import BallDetector

# Player detection
PlayerDetector(weights="best.pt", source="match.mp4", out_dir="out").detect()

# Speed tracking
SpeedTracker(
    weights="best.pt",
    source="match.mp4",
    out_dir="out_speed",
    court_w_m=40,
    court_h_m=20,
    team1_name="Home", team1_color="blue",
    team2_name="Away", team2_color="red",
).track()

# Ball zone analysis
BallDetector(
    weights="best.pt",
    source="match.mp4",
    out_dir="out_ball",
    ball_class=1,
).detect_ball()
```

Or use the combined class:

```python
from main import FutsalAnalytics

fa = FutsalAnalytics(
    weights="best.pt",
    source="match.mp4",
    court_w_m=40,
    team1_name="Home", team1_color="blue",
    team2_name="Away", team2_color="red",
    ball_class=1,
)

fa.detect()       # player heatmap + radar
fa.track()        # speed + sprint stats
fa.detect_ball()  # ball zone analysis
```

## Court corners

All three modules share the same corner-selection UI.  
Click **4 corners in order**: top-left → bottom-left → bottom-right → top-right.  
Corners are saved to `<out_dir>/court_corners.json` and reused on the next run — no need to re-select every time.  
Press `R` to reset, `Esc` / `Space` / `Enter` to skip.
