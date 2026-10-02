import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Analyze a tennis video with a supplied ball-detection checkpoint.")
    parser.add_argument("--source", required=True, help="Local video path or a downloadable video URL")
    parser.add_argument("--model", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/Tennis_Pro_Analysis.mp4")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("The ball-detection checkpoint is missing. Supply --model path/to/best.pt.")
    if not args.source.startswith(("https://", "http://")) and not Path(args.source).is_file():
        parser.error("The local video does not exist.")
    from src.analyzer import TennisProAnalyzer
    analyzer = TennisProAnalyzer(model_path=str(args.model.resolve()))
    result = analyzer.process_video(args.source, str(args.output.resolve()))
    print("Output:", result)

if __name__ == "__main__":
    main()
