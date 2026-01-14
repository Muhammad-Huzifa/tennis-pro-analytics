# 🎾 AI Tennis Pro Analyzer

A professional computer vision system for analyzing tennis matches.

## Features
- Real-time Ball Trajectory Analysis
- Player Movement Heatmap
- Shot Speed Classification
- Rally Statistics

## Structure
- `src/models.py`: Data structures
- `src/visualization.py`: Drawing logic
- `src/analyzer.py`: Main logic pipeline
- `main.py`: Entry point

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Place your trained YOLO model weights (`best.pt`) in the root.
3. Run: `python main.py`
