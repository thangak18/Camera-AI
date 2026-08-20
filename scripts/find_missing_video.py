import os
from pathlib import Path

def main():
    DATA_DIR = Path(__file__).parent.parent.parent / "data" / "v1"
    CLIP_DIR = DATA_DIR / "clip_features"
    KEYFRAME_DIR = DATA_DIR / "keyframes"

    if not CLIP_DIR.exists() or not KEYFRAME_DIR.exists():
        print("Lỗi: Không tìm thấy thư mục clip_features hoặc keyframes.")
        print(f"Đường dẫn đang tìm: {DATA_DIR}")
        return

    clip_files = {p.stem for p in CLIP_DIR.glob("*.npy")}
    keyframe_dirs = {p.name for p in KEYFRAME_DIR.iterdir() if p.is_dir()}

    missing = clip_files - keyframe_dirs
    print(f"Tổng số file .npy: {len(clip_files)}")
    print(f"Tổng số thư mục keyframes: {len(keyframe_dirs)}")
    print(f"Videos có .npy nhưng không có keyframe: {missing}")

if __name__ == "__main__":
    main()
