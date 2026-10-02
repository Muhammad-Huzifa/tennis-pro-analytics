# Tennis Video Analytics

A video-analysis prototype combining a supplied YOLO ball detector with MediaPipe pose tracking. It produces a side-by-side annotated video, ball trajectory records, and frame-based movement statistics.

## Setup

Use Python 3.11, clone the project, and create an environment:

```bash
git clone https://github.com/Muhammad-Huzifa/tennis-pro-analytics.git
cd tennis-pro-analytics
python -m venv .venv
```

Activate `.venv\Scripts\activate.bat` in Windows Command Prompt, `.\.venv\Scripts\Activate.ps1` in PowerShell, `source .venv/Scripts/activate` in Git Bash, or `source .venv/bin/activate` on Linux/macOS.

```bash
python -m pip install -r requirements.txt
python main.py --help
python main.py --source path/to/match.mp4 --model path/to/best.pt
```

The ball checkpoint is not included. Use a ball-specific detector and a readable video of at least 640×480. To download an explicitly chosen URL instead of using a local video, pass that URL to `--source`.

## Outputs and structure

| Path | Purpose |
| --- | --- |
| `main.py` | Configurable source, model, and output CLI |
| `src/analyzer.py` | Detection, pose tracking, processing, and exports |
| `src/models.py` | Session and trajectory data structures |
| `src/visualization.py` | Overlays, trajectory, and heatmap drawing |
| `outputs/` | Default annotated video location |
| `docs/ANALYSIS.md` | Units, assumptions, and implementation limits |

The default video output is `outputs/Tennis_Pro_Analysis.mp4`. Analytics JSON and trajectory CSV are written in the current working directory. Use `--output path/to/output.mp4` to choose the video path.

Motion statistics are based on pixels, not calibrated physical speed. Read [analysis conventions](docs/ANALYSIS.md) before interpreting results. CLI help, Python syntax, and four video resource lifecycle checks passed. The checks use fake model/video dependencies; full dependency installation and real checkpoint/video inference were not performed.

Run the resource checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

Muhammad Huzifa — [GitHub](https://github.com/Muhammad-Huzifa)
