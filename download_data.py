import os
import pandas as pd
import requests
from tqdm import tqdm

EXCEL_PATH = r"C:\Users\ADMIN\OneDrive\Documents\Documents\AI\Thi cuộc thi\AI Challenge\Camera-AI\Dữ liệu cho vòng Sơ Tuyển AIC 2026.xlsx"
DATA_DIR = r"C:\Users\ADMIN\OneDrive\Documents\Documents\AI\Thi cuộc thi\AI Challenge\data"

def download_file(url, target_path):
    head_res = requests.head(url, timeout=30)
    head_res.raise_for_status()
    total_size = int(head_res.headers.get('content-length', 0))

    # 1. Bỏ qua nếu file zip đã tải xong hoàn chỉnh
    if os.path.exists(target_path) and os.path.getsize(target_path) == total_size and total_size > 0:
        print(f"[SKIP] {os.path.basename(target_path)} already downloaded completely ({total_size / (1024*1024):.2f} MB).")
        return

    temp_path = target_path + ".tmp"
    initial_pos = 0
    headers = {}

    # 2. Hỗ trợ tải tiếp (Resume) nếu file .tmp đang bị ngắt giữa chừng
    if os.path.exists(temp_path):
        initial_pos = os.path.getsize(temp_path)
        if initial_pos < total_size:
            headers['Range'] = f"bytes={initial_pos}-"
            print(f"[RESUMING] {os.path.basename(target_path)} from {initial_pos / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB...")
        elif initial_pos == total_size:
            os.replace(temp_path, target_path)
            print(f"[DONE] {os.path.basename(target_path)}")
            return
        else:
            initial_pos = 0

    if initial_pos == 0:
        print(f"[DOWNLOADING] {os.path.basename(target_path)} ({total_size / (1024*1024):.2f} MB)...")

    mode = 'ab' if initial_pos > 0 else 'wb'
    response = requests.get(url, headers=headers, stream=True, timeout=30)
    response.raise_for_status()

    with open(temp_path, mode) as f, tqdm(
        total=total_size, initial=initial_pos, unit='iB', unit_scale=True, desc=os.path.basename(target_path)
    ) as pbar:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    os.replace(temp_path, target_path)
    print(f"[DONE] {os.path.basename(target_path)}")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.read_excel(EXCEL_PATH)
    
    print(f"Found {len(df)} files to download.")
    for idx, row in df.iterrows():
        filename = str(row['Filenames']).strip()
        link = str(row['Download link']).strip()
        target_file = os.path.join(DATA_DIR, filename)
        try:
            download_file(link, target_file)
        except Exception as e:
            print(f"[ERROR] Failed to download {filename} from {link}: {e}")

if __name__ == "__main__":
    main()
