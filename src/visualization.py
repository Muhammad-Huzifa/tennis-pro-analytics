import cv2
import numpy as np
from typing import Tuple

class VisualizationEngine:
    "Handles all professional-grade visualizations."
    
    # Color Palette (BGR Format)
    COLORS = {
        'primary': (255, 165, 0),      # Orange
        'secondary': (0, 255, 255),    # Cyan
        'accent': (147, 20, 255),      # Pink
        'success': (0, 255, 100),      # Green
        'warning': (0, 200, 255),      # Yellow
        'danger': (0, 0, 255),         # Red
        'dark': (30, 30, 30),          # Dark Gray
        'light': (220, 220, 220),      # Light Gray
        'white': (255, 255, 255),
        'black': (0, 0, 0),
    }
    
    # Speed Category Colors
    SPEED_COLORS = {
        'slow': (0, 255, 100),         # Green
        'medium': (0, 255, 255),       # Yellow
        'fast': (0, 165, 255),         # Orange
        'bullet': (0, 0, 255),         # Red
    }
    
    @staticmethod
    def get_speed_category(speed: float) -> Tuple[str, Tuple[int, int, int]]:
        "Classify speed into categories with corresponding colors."
        if speed < 8:
            return "SLOW", VisualizationEngine.SPEED_COLORS['slow']
        elif speed < 20:
            return "MEDIUM", VisualizationEngine.SPEED_COLORS['medium']
        elif speed < 35:
            return "FAST", VisualizationEngine.SPEED_COLORS['fast']
        else:
            return "BULLET", VisualizationEngine.SPEED_COLORS['bullet']
    
    @staticmethod
    def draw_gradient_line(img, pt1, pt2, color1, color2, thickness=3):
        steps = 10
        for i in range(steps):
            t = i / steps
            x = int(pt1[0] + t * (pt2[0] - pt1[0]))
            y = int(pt1[1] + t * (pt2[1] - pt1[1]))
            x_next = int(pt1[0] + (t + 1/steps) * (pt2[0] - pt1[0]))
            y_next = int(pt1[1] + (t + 1/steps) * (pt2[1] - pt1[1]))
            
            color = tuple(int(c1 + t * (c2 - c1)) for c1, c2 in zip(color1, color2))
            cv2.line(img, (x, y), (x_next, y_next), color, thickness)
    
    @staticmethod
    def draw_dashboard_panel(img, x, y, w, h, title, opacity=0.85):
        overlay = img.copy()
        
        # Main Panel Background
        cv2.rectangle(overlay, (x, y), (x+w, y+h), 
                     VisualizationEngine.COLORS['dark'], -1)
        
        # Border
        cv2.rectangle(overlay, (x, y), (x+w, y+h), 
                     VisualizationEngine.COLORS['primary'], 2)
        
        # Title Bar
        cv2.rectangle(overlay, (x, y), (x+w, y+30), 
                     VisualizationEngine.COLORS['primary'], -1)
        
        # Title Text
        cv2.putText(overlay, title, (x+10, y+22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   VisualizationEngine.COLORS['white'], 2)
        
        cv2.addWeighted(overlay, opacity, img, 1-opacity, 0, img)
        return img
    
    @staticmethod
    def draw_stat_row(img, x, y, label, value, color=None):
        if color is None:
            color = VisualizationEngine.COLORS['light']
        
        cv2.putText(img, f"{label}:", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   VisualizationEngine.COLORS['light'], 1)
        cv2.putText(img, str(value), (x+120, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    @staticmethod
    def draw_speed_gauge(img, x, y, speed, max_speed=50):
        gauge_w, gauge_h = 150, 20
        
        # Background
        cv2.rectangle(img, (x, y), (x+gauge_w, y+gauge_h), 
                     VisualizationEngine.COLORS['dark'], -1)
        cv2.rectangle(img, (x, y), (x+gauge_w, y+gauge_h), 
                     VisualizationEngine.COLORS['light'], 1)
        
        # Fill based on speed
        fill_w = int((min(speed, max_speed) / max_speed) * gauge_w)
        _, color = VisualizationEngine.get_speed_category(speed)
        cv2.rectangle(img, (x+2, y+2), (x+fill_w-2, y+gauge_h-2), color, -1)
        
        # Speed Text
        cv2.putText(img, f"{speed:.1f} px/f", (x+gauge_w+10, y+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    @staticmethod
    def draw_mini_heatmap(img, positions, x, y, w, h, frame_w, frame_h):
        # Create mini heatmap
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        for px, py in positions:
            # Scale position to mini map
            mx = int((px / frame_w) * w)
            my = int((py / frame_h) * h)
            mx = max(0, min(w-1, mx))
            my = max(0, min(h-1, my))
            
            # Add gaussian blob
            cv2.circle(heatmap, (mx, my), 5, 1.0, -1)
        
        # Blur for smooth heatmap
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # Normalize and colorize
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)
        
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Draw on main image
        img[y:y+h, x:x+w] = cv2.addWeighted(
            img[y:y+h, x:x+w], 0.3, heatmap_color, 0.7, 0
        )
        
        # Border
        cv2.rectangle(img, (x, y), (x+w, y+h), 
                     VisualizationEngine.COLORS['primary'], 2)
