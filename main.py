import os
from src.analyzer import TennisProAnalyzer

# ==============================================================================
# 🚀 EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    # Configuration
    VIDEO_URL = "https://www.youtube.com/shorts/Pb2c0Bl7skE"
    
    # ⚠️ Check if we are in a Kaggle environment or local
    # Adjust MODEL_PATH as needed for your environment
    if os.path.exists("/kaggle/working/runs/detect/train/weights/best.pt"):
        MODEL_PATH = "/kaggle/working/runs/detect/train/weights/best.pt"
    else:
        MODEL_PATH = "best.pt" # Local assumption
    
    OUTPUT_FILE = "Tennis_Pro_Analysis.mp4"
    
    print(f"Using Model Path: {MODEL_PATH}")
    
    # Create analyzer and run
    try:
        analyzer = TennisProAnalyzer(model_path=MODEL_PATH)
        output = analyzer.process_video(
            video_url=VIDEO_URL,
            output_file=OUTPUT_FILE
        )
        print(f"\n🎉 Success! Output saved to: {output}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Tip: Make sure you have the YOLO model weights file available.")
