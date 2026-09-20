import argparse
import os
from modules.surveillance_analysis import analyze_video


def process_video(input_path: str, output_path: str = None):
    result = analyze_video(input_path, output_video_path=output_path, enable_gradcam=False, enable_report=False, enable_email=False)
    print("-" * 50)
    print(f"Analysis complete in {result.processing_time:.1f} seconds!")
    if output_path:
        print(f"Annotated video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Violence Detection on a raw video.")
    parser.add_argument("video_path", type=str, help="Path to the input .mp4 video")
    parser.add_argument("--output", type=str, default="output_annotated.mp4", help="Path to save the annotated video (optional)")
    
    args = parser.parse_args()
    process_video(args.video_path, args.output)
