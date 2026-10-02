import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import yt_dlp
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import math
import json
import csv
from collections import deque
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Tuple

from src.models import AnalysisSession, BallPosition
from src.visualization import VisualizationEngine

class TennisProAnalyzer:
    "Professional Tennis Analysis System."
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.viz = VisualizationEngine()
        self.session = AnalysisSession()
        
        # Tracking History
        self.ball_history = deque(maxlen=50)
        self.player_history = deque(maxlen=100)
        self.speed_history = deque(maxlen=30)
        
        # Models
        self.yolo_model = None
        self.pose_detector = None
        
        # Frame counter
        self.frame_count = 0
        
    def initialize_models(self):
        "Load AI models."
        print("🧠 Initializing AI Models...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.yolo_model = YOLO(self.model_path)
        
        mp_pose = mp.solutions.pose
        self.pose_detector = mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1
        )
        self.mp_pose = mp_pose
        
        print("✅ Models loaded successfully!")
    
    def download_video(self, url: str, output: str = "input_pro.mp4") -> str:
        print(f"⬇️ Downloading video from: {url[:50]}...")
        
        if os.path.exists(output):
            os.remove(output)
        
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output,
            'quiet': True,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        print("✅ Video downloaded!")
        return output
    
    def calculate_speed(self, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
        return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    
    def get_court_zone(self, x: int, y: int, w: int, h: int) -> Tuple[str, str]:
        if x < w * 0.33:
            h_zone = "left"
        elif x < w * 0.66:
            h_zone = "center"
        else:
            h_zone = "right"
        
        if y < h * 0.33:
            v_zone = "net"
        elif y < h * 0.66:
            v_zone = "mid"
        else:
            v_zone = "baseline"
        
        return h_zone, v_zone
    
    def detect_player(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(img_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Use hip center as player position
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            cx = int((left_hip.x + right_hip.x) / 2 * frame.shape[1])
            cy = int((left_hip.y + right_hip.y) / 2 * frame.shape[0])
            
            return (cx, cy), results.pose_landmarks
        
        return None, None
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        results = self.yolo_model.predict(frame, conf=0.3, verbose=False)[0]
        
        best_detection = None
        best_conf = 0
        
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                best_detection = (cx, cy)
                best_conf = conf
        
        return best_detection
    
    def draw_ball_trajectory(self, canvas: np.ndarray):
        if len(self.ball_history) < 2:
            return
        
        points = list(self.ball_history)
        
        for i in range(1, len(points)):
            pt1 = (points[i-1].x, points[i-1].y)
            pt2 = (points[i].x, points[i].y)
            
            # Get color based on speed
            _, color = self.viz.get_speed_category(points[i].speed)
            
            # Fade effect based on age
            alpha = (i / len(points))
            thickness = max(2, int(4 * alpha))
            
            cv2.line(canvas, pt1, pt2, color, thickness)
        
        # Draw current ball with glow effect
        if points:
            current = points[-1]
            # Outer glow
            cv2.circle(canvas, (current.x, current.y), 15, 
                      self.viz.COLORS['white'], 2)
            # Inner ball
            cv2.circle(canvas, (current.x, current.y), 8, 
                      self.viz.COLORS['primary'], -1)
    
    def draw_player_skeleton(self, canvas: np.ndarray, landmarks):
        if landmarks is None:
            return
        
        mp.solutions.drawing_utils.draw_landmarks(
            canvas,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(
                color=self.viz.COLORS['secondary'], 
                thickness=3, 
                circle_radius=4
            ),
            mp.solutions.drawing_utils.DrawingSpec(
                color=self.viz.COLORS['primary'], 
                thickness=2
            )
        )
    
    def draw_analytics_overlay(self, frame: np.ndarray, 
                                current_speed: float,
                                speed_category: str):
        h, w = frame.shape[:2]
        
        # === TOP LEFT: Live Stats Panel ===
        self.viz.draw_dashboard_panel(frame, 10, 10, 250, 180, "📊 LIVE ANALYTICS")
        
        y_offset = 55
        stats = [
            ("Frame", f"#{self.frame_count}"),
            ("Ball Detections", self.session.rally_stats.ball_detections),
            ("Current Speed", f"{current_speed:.1f} px/f"),
            ("Speed Class", speed_category),
            ("Max Speed", f"{self.session.rally_stats.max_ball_speed:.1f} px/f"),
            ("Avg Speed", f"{self.session.rally_stats.avg_ball_speed:.1f} px/f"),
        ]
        
        for label, value in stats:
            color = self.viz.COLORS['light']
            if label == "Speed Class":
                _, color = self.viz.get_speed_category(current_speed)
            self.viz.draw_stat_row(frame, 20, y_offset, label, value, color)
            y_offset += 22
        
        # === TOP RIGHT: Speed Distribution ===
        self.viz.draw_dashboard_panel(frame, w-220, 10, 210, 140, "⚡ SPEED DISTRIBUTION")
        
        dist = self.session.rally_stats.speed_distribution
        total = sum(dist.values()) or 1
        
        y_offset = 50
        for category in ['slow', 'medium', 'fast', 'bullet']:
            count = dist[category]
            pct = (count / total) * 100
            
            # Draw bar
            bar_w = int((count / total) * 150) if total > 0 else 0
            cv2.rectangle(frame, (w-210, y_offset-12), 
                         (w-210+bar_w, y_offset+2),
                         self.viz.SPEED_COLORS[category], -1)
            
            # Label
            cv2.putText(frame, f"{category.upper()}: {pct:.0f}%", 
                       (w-210, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       self.viz.COLORS['white'], 1)
            
            y_offset += 25
        
        # === BOTTOM LEFT: Speed Gauge ===
        self.viz.draw_dashboard_panel(frame, 10, h-80, 250, 70, "🎯 SPEED METER")
        self.viz.draw_speed_gauge(frame, 20, h-45, current_speed)
        
        # === BOTTOM RIGHT: Mini Heatmap ===
        if len(self.ball_history) > 5:
            positions = [(bp.x, bp.y) for bp in self.ball_history]
            self.viz.draw_mini_heatmap(frame, positions, 
                                       w-170, h-130, 160, 120, w, h)
            cv2.putText(frame, "BALL HEATMAP", (w-165, h-135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       self.viz.COLORS['primary'], 1)
        
        # === TOP CENTER: Branding ===
        title = "🎾 AI TENNIS PRO ANALYZER"
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        tx = (w - text_size[0]) // 2
        
        # Background for title
        cv2.rectangle(frame, (tx-10, 5), (tx+text_size[0]+10, 35),
                     self.viz.COLORS['dark'], -1)
        cv2.rectangle(frame, (tx-10, 5), (tx+text_size[0]+10, 35),
                     self.viz.COLORS['primary'], 2)
        cv2.putText(frame, title, (tx, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   self.viz.COLORS['white'], 2)
    
    def create_analysis_canvas(self, frame: np.ndarray, 
                                landmarks) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw grid lines for reference
        for i in range(0, w, w//6):
            cv2.line(canvas, (i, 0), (i, h), (30, 30, 30), 1)
        for i in range(0, h, h//6):
            cv2.line(canvas, (0, i), (w, i), (30, 30, 30), 1)
        
        # Draw player skeleton
        self.draw_player_skeleton(canvas, landmarks)
        
        # Draw ball trajectory
        self.draw_ball_trajectory(canvas)
        
        # Draw player movement trail
        if len(self.player_history) > 1:
            for i in range(1, len(self.player_history)):
                pt1 = self.player_history[i-1]
                pt2 = self.player_history[i]
                alpha = i / len(self.player_history)
                color = tuple(int(c * alpha) for c in self.viz.COLORS['secondary'])
                cv2.line(canvas, pt1, pt2, color, 2)
        
        return canvas
    
    def export_analytics(self, output_prefix: str = "tennis_analysis"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON Export
        json_file = f"{output_prefix}_{timestamp}.json"
        with open(json_file, 'w') as f:
            # Convert dataclass to dict
            export_data = {
                'session_info': {
                    'video_source': self.session.video_source,
                    'analysis_date': self.session.analysis_date,
                    'duration_seconds': self.session.duration_seconds,
                    'fps': self.session.fps,
                    'resolution': self.session.resolution
                },
                'rally_stats': asdict(self.session.rally_stats),
                'player_metrics': {
                    'total_distance': self.session.player_metrics.total_distance,
                    'max_speed': self.session.player_metrics.max_speed,
                    'avg_speed': self.session.player_metrics.avg_speed,
                    'movement_samples': self.session.player_metrics.movement_samples
                }
            }
            json.dump(export_data, f, indent=2)
        
        # CSV Export for trajectory data
        csv_file = f"{output_prefix}_trajectory_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_id', 'x', 'y', 'speed', 'speed_category', 'timestamp'])
            for bp in self.session.ball_trajectory:
                writer.writerow([bp.frame_id, bp.x, bp.y, 
                               f"{bp.speed:.2f}", bp.speed_category, 
                               f"{bp.timestamp:.3f}"])
        
        print(f"📁 Analytics exported:")
        print(f"   → {json_file}")
        print(f"   → {csv_file}")
        
        return json_file, csv_file
    
    def process_video(self, video_url: str, output_file: str = "Tennis_Pro_Analysis.mp4"):
        output = Path(output_file).resolve()
        local_source = Path(video_url)
        if local_source.is_file() and local_source.resolve() == output:
            raise ValueError("Input and output video paths must differ.")
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            if local_source.is_file():
                return self._process_video(str(local_source.resolve()), str(output))
            with TemporaryDirectory(prefix="tennis-video-") as folder:
                downloaded = self.download_video(video_url, str(Path(folder) / "input.mp4"))
                return self._process_video(downloaded, str(output))
        finally:
            if self.pose_detector is not None:
                self.pose_detector.close()

    def _process_video(self, video_url: str, output_file: str):
        # Initialize
        self.initialize_models()
        input_video = video_url
        
        # Session metadata
        self.session.video_source = video_url
        self.session.analysis_date = datetime.now().isoformat()
        
        # Open video
        cap = cv2.VideoCapture(input_video)
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if not cap.isOpened() or w < 640 or h < 480 or not math.isfinite(fps) or fps <= 0:
            cap.release()
            raise ValueError("Use a readable video of at least 640x480 with a valid frame rate.")
        self.session.fps = fps
        self.session.resolution = (w, h)
        self.session.duration_seconds = total_frames / fps if fps > 0 else 0
        
        # Output writer (Side by Side: Original + Analysis)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (w * 2, h))
        
        if not out.isOpened():
            cap.release()
            out.release()
            raise OSError("Could not open the output video writer.")
        print(f"\n🎬 Processing Video...")
        print(f"   Resolution: {w}x{h} @ {fps}fps")
        print(f"   Total Frames: {total_frames}")
        print(f"   Duration: {self.session.duration_seconds:.1f}s\n")
        
        current_speed = 0.0
        speed_category = "---"
        all_speeds = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                self.session.rally_stats.total_frames = self.frame_count
                
                # Progress indicator
                if self.frame_count % 30 == 0 and total_frames > 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"   Processing: {progress:.1f}% ({self.frame_count}/{total_frames})", 
                          end='\r')
                
                # === DETECT PLAYER ===
                player_result = self.detect_player(frame)
                player_pos, landmarks = player_result if player_result else (None, None)
                
                if player_pos:
                    self.player_history.append(player_pos)
                    
                    # Update player metrics
                    if len(self.player_history) >= 2:
                        dist = self.calculate_speed(
                            self.player_history[-2], 
                            self.player_history[-1]
                        )
                        self.session.player_metrics.total_distance += dist
                        self.session.player_metrics.movement_samples += 1
                        
                        if dist > self.session.player_metrics.max_speed:
                            self.session.player_metrics.max_speed = dist
                
                # === DETECT BALL ===
                ball_pos = self.detect_ball(frame)
                
                if ball_pos:
                    self.session.rally_stats.ball_detections += 1
                    
                    # Calculate speed
                    if len(self.ball_history) > 0:
                        prev = self.ball_history[-1]
                        current_speed = self.calculate_speed(
                            (prev.x, prev.y), ball_pos
                        )
                    else:
                        current_speed = 0.0
                    
                    all_speeds.append(current_speed)
                    
                    # Update max speed
                    if current_speed > self.session.rally_stats.max_ball_speed:
                        self.session.rally_stats.max_ball_speed = current_speed
                    
                    # Update average speed
                    self.session.rally_stats.avg_ball_speed = np.mean(all_speeds)
                    
                    # Get speed category
                    speed_category, _ = self.viz.get_speed_category(current_speed)
                    
                    # Update speed distribution
                    self.session.rally_stats.speed_distribution[
                        speed_category.lower()
                    ] += 1
                    
                    # Update court zones
                    h_zone, v_zone = self.get_court_zone(ball_pos[0], ball_pos[1], w, h)
                    self.session.rally_stats.court_zones_hit[h_zone] += 1
                    self.session.rally_stats.court_zones_hit[v_zone] += 1
                    
                    # Create ball position record
                    ball_record = BallPosition(
                        frame_id=self.frame_count,
                        x=ball_pos[0],
                        y=ball_pos[1],
                        speed=current_speed,
                        speed_category=speed_category,
                        timestamp=self.frame_count / fps
                    )
                    
                    self.ball_history.append(ball_record)
                    self.session.ball_trajectory.append(ball_record)
                
                # === CREATE VISUALIZATIONS ===
                
                # Analysis Canvas
                analysis_canvas = self.create_analysis_canvas(frame, landmarks)
                
                # Clone frame for overlay
                overlay_frame = frame.copy()
                
                # Add analytics overlay to original frame
                self.draw_analytics_overlay(overlay_frame, current_speed, speed_category)
                
                # Add label to analysis canvas
                cv2.putText(analysis_canvas, "TRAJECTORY ANALYSIS", (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                           self.viz.COLORS['primary'], 2)
                
                # Combine frames side by side
                combined = np.hstack((overlay_frame, analysis_canvas))
                
                out.write(combined)
        finally:
            cap.release()
            out.release()

        # Calculate final player metrics
        if self.session.player_metrics.movement_samples > 0:
            self.session.player_metrics.avg_speed = (
                self.session.player_metrics.total_distance / 
                self.session.player_metrics.movement_samples
            )
        
        print(f"\n\n✅ Processing Complete!")
        print(f"   → Output Video: {output_file}")
        
        # Export analytics
        self.export_analytics()
        
        # Print summary
        self.print_summary()
        
        return output_file
    
    def print_summary(self):
        stats = self.session.rally_stats
        player = self.session.player_metrics
        
        print("\n" + "="*60)
        print("📊 ANALYSIS SUMMARY")
        print("="*60)
        print(f"\n🎬 Video Info:")
        print(f"   • Duration: {self.session.duration_seconds:.1f} seconds")
        print(f"   • Resolution: {self.session.resolution[0]}x{self.session.resolution[1]}")
        print(f"   • FPS: {self.session.fps}")
        print(f"   • Total Frames Analyzed: {stats.total_frames}")
        
        print(f"\n🎾 Ball Statistics:")
        print(f"   • Total Detections: {stats.ball_detections}")
        print(f"   • Average Speed: {stats.avg_ball_speed:.2f} px/frame")
        print(f"   • Maximum Speed: {stats.max_ball_speed:.2f} px/frame")
        
        print(f"\n⚡ Speed Distribution:")
        total = sum(stats.speed_distribution.values()) or 1
        for cat, count in stats.speed_distribution.items():
            pct = (count / total) * 100
            bar = "█" * int(pct / 5)
            print(f"   • {cat.upper():8s}: {bar} {pct:.1f}%")
        
        print(f"\n🏃 Player Movement:")
        print(f"   • Total Distance: {player.total_distance:.0f} pixels")
        print(f"   • Max Speed: {player.max_speed:.2f} px/frame")
        print(f"   • Avg Speed: {player.avg_speed:.2f} px/frame")
        
        print(f"\n📍 Court Zone Activity:")
        for zone, count in stats.court_zones_hit.items():
            if count > 0:
                print(f"   • {zone.upper()}: {count} hits")
        
        print("\n" + "="*60)
