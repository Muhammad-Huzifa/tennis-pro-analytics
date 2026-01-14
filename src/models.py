from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class BallPosition:
    "Represents a single ball detection with metadata."
    frame_id: int
    x: int
    y: int
    speed: float = 0.0
    speed_category: str = "Unknown"
    timestamp: float = 0.0

@dataclass
class PlayerMetrics:
    "Tracks player movement and performance."
    total_distance: float = 0.0
    max_speed: float = 0.0
    avg_speed: float = 0.0
    positions: List[Tuple[int, int]] = field(default_factory=list)
    movement_samples: int = 0

@dataclass
class RallyStats:
    "Comprehensive rally and match statistics."
    total_frames: int = 0
    ball_detections: int = 0
    avg_ball_speed: float = 0.0
    max_ball_speed: float = 0.0
    speed_distribution: Dict[str, int] = field(default_factory=lambda: {
        "slow": 0, "medium": 0, "fast": 0, "bullet": 0
    })
    court_zones_hit: Dict[str, int] = field(default_factory=lambda: {
        "left": 0, "center": 0, "right": 0,
        "net": 0, "mid": 0, "baseline": 0
    })

@dataclass
class AnalysisSession:
    "Complete analysis session data."
    video_source: str = ""
    analysis_date: str = ""
    duration_seconds: float = 0.0
    fps: int = 0
    resolution: Tuple[int, int] = (0, 0)
    rally_stats: RallyStats = field(default_factory=RallyStats)
    player_metrics: PlayerMetrics = field(default_factory=PlayerMetrics)
    ball_trajectory: List[BallPosition] = field(default_factory=list)
