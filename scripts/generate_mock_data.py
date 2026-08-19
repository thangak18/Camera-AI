import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.config import CLIP_FEATURES_DIR, MAP_KEYFRAMES_DIR, VECTOR_SIZE, OBJECTS_DIR, METADATA_DIR

def create_mock_data():
    os.makedirs(CLIP_FEATURES_DIR, exist_ok=True)
    os.makedirs(MAP_KEYFRAMES_DIR, exist_ok=True)
    
    # Tạo mock cho 2 video
    for video_id in ["L01_V001", "L01_V002"]:
        # Tạo numpy array (giả lập 10 keyframe mỗi video)
        num_frames = 10
        vectors = np.random.rand(num_frames, VECTOR_SIZE).astype(np.float32)
        np.save(Path(CLIP_FEATURES_DIR) / f"{video_id}.npy", vectors)
        
        # Tạo map_keyframes CSV
        df = pd.DataFrame({
            "n": range(1, num_frames + 1),
            "pts_time": np.linspace(0, 10, num_frames),
            "fps": [25.0] * num_frames,
            "frame_idx": np.linspace(0, 250, num_frames).astype(int)
        })
        df.to_csv(Path(MAP_KEYFRAMES_DIR) / f"{video_id}.csv", index=False)
        
    print("Mock data created successfully!")

if __name__ == "__main__":
    create_mock_data()
