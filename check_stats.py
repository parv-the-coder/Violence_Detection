import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FRAME_OUTPUT_DIR

def main():
    proc_dir = FRAME_OUTPUT_DIR
    
    if not os.path.exists(proc_dir):
        print('Directory does not exist yet.')
        return

    total_images = 0
    total_videos = 0
    category_stats = {}

    for category in sorted(os.listdir(proc_dir)):
        cat_path = os.path.join(proc_dir, category)
        if not os.path.isdir(cat_path):
            continue
        
        cat_images = 0
        cat_videos = 0
        
        for video in os.listdir(cat_path):
            video_path = os.path.join(cat_path, video)
            if not os.path.isdir(video_path):
                continue
                
            cat_videos += 1
            images = len([f for f in os.listdir(video_path) if f.endswith('.jpg')])
            cat_images += images
            
        category_stats[category] = {'videos': cat_videos, 'images': cat_images}
        total_videos += cat_videos
        total_images += cat_images

    print(f'Total Videos Processed: {total_videos}')
    print(f'Total Frames Extracted: {total_images}\n')
    print('Breakdown by Category:')
    for cat, stats in category_stats.items():
        print(f'  - {cat}: {stats["videos"]} videos, {stats["images"]} images')

if __name__ == "__main__":
    main()
