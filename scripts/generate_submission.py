import os
import sys
import json
import argparse
import requests
import csv
import zipfile
import shutil
from pathlib import Path

def to_real_frame_idx(video_id: str, frame_idx: int) -> int:
    """Giả lập hàm chuyển đổi frame. Ở Production, cần map chính xác."""
    return int(frame_idx)

def parse_query_file(file_path: str):
    """
    Cố gắng parse file theo dạng JSON trước. 
    Nếu không phải JSON thì lấy toàn bộ nội dung file làm text thuần túy.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError:
        return content

def process_and_write_results(q_type: str, query_data, api_url: str, output_csv: str):
    """Gọi API và xuất thẳng ra file CSV theo chuẩn BTC."""
    
    headers = {
        'Content-Type': 'application/json',
        'Bypass-Tunnel-Reminder': 'true'
    }
    
    results = []
    
    try:
        if q_type == "kis":
            # Nếu truyền vào là dạng text thuần (query string)
            query_str = query_data if isinstance(query_data, str) else query_data.get("query", "")
            res = requests.post(
                f"{api_url}/search/kis", 
                json={"query": query_str, "top_k": 100},
                headers=headers
            )
            res.raise_for_status()
            results = res.json()
            
        elif q_type == "qa":
            if isinstance(query_data, str):
                question = query_data
                description = ""
            else:
                question = query_data.get("question_vi") or query_data.get("question", "")
                description = query_data.get("description_vi") or query_data.get("description", "")
                
            res = requests.post(
                f"{api_url}/search/qa", 
                json={"description": description, "question": question, "top_k": 100},
                headers=headers
            )
            res.raise_for_status()
            results = res.json()
            
        elif q_type == "trake":
            # Trake thường là danh sách sự kiện
            event_queries = []
            if isinstance(query_data, str):
                # Chia mỗi dòng là 1 sự kiện nếu format là text thuần
                event_queries = [line.strip() for line in query_data.split('\n') if line.strip()]
            else:
                event_queries = query_data.get("event_queries", [])
                
            res = requests.post(
                f"{api_url}/search/trake", 
                json={"event_queries": event_queries, "top_k": 100},
                headers=headers
            )
            res.raise_for_status()
            results = res.json()
            
    except Exception as e:
        print(f"❌ Lỗi khi gọi API cho file {os.path.basename(output_csv)}: {e}")
        return False

    # Ghi ra file CSV chuẩn BTC
    # Dùng QUOTE_MINIMAL mặc định của Python, nhưng đối với câu trả lời dạng chữ, có thể ép QUOTE_NONNUMERIC nếu cần.
    # BTC chấp nhận chuẩn quote của CSV tiêu chuẩn.
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        # Nếu là QA, bọc ngoặc kép cho tất cả các câu trả lời để an toàn 100%
        if q_type == "qa":
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        else:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            
        for result in results[:100]:
            if q_type == "kis":
                real_frame = to_real_frame_idx(result["video_id"], result["frame_id"])
                writer.writerow([result["video_id"], real_frame])
            elif q_type == "qa":
                real_frame = to_real_frame_idx(result["video_id"], result["frame_id"])
                writer.writerow([result["video_id"], real_frame, result.get("answer", "")])
            elif q_type == "trake":
                real_frames = [to_real_frame_idx(result["video_id"], f) for f in result.get("frame_ids", [])]
                row = [result["video_id"]] + real_frames
                writer.writerow(row)
                
    return True

def generate_submission(query_dir: str, api_url: str):
    if not os.path.exists(query_dir):
        print(f"Lỗi: Không tìm thấy thư mục {query_dir}")
        return

    # Tạo thư mục submission tạm thời
    submission_dir = "submission"
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir)
    
    print(f"📂 Đang quét các file đề thi (.txt) trong '{query_dir}'...")
    
    success_count = 0
    
    for filename in os.listdir(query_dir):
        if not filename.endswith(".txt"):
            continue
            
        file_path = os.path.join(query_dir, filename)
        
        # Nhận diện loại query từ tên file (vd: query-1-kis.txt -> kis)
        basename = filename[:-4]  # bỏ .txt
        if basename.endswith("-kis"):
            q_type = "kis"
        elif basename.endswith("-qa"):
            q_type = "qa"
        elif basename.endswith("-trake"):
            q_type = "trake"
        else:
            print(f"⚠️ Bỏ qua file {filename} vì không nhận diện được loại truy vấn ở đuôi tên file.")
            continue
            
        print(f"⏳ Đang giải quyết: {filename} (Loại: {q_type.upper()})...")
        query_data = parse_query_file(file_path)
        
        output_csv = os.path.join(submission_dir, f"{basename}.csv")
        
        if process_and_write_results(q_type, query_data, api_url, output_csv):
            success_count += 1
            print(f"   ✅ Đã xuất {basename}.csv")

    if success_count == 0:
        print("❌ Không tạo được file CSV nào. Dừng quá trình nén.")
        shutil.rmtree(submission_dir)
        return

    # Nén thư mục submission thành submission.zip
    print(f"\n📦 Đang nén thư mục '{submission_dir}' thành 'submission.zip'...")
    zip_filename = "submission.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Chỉ nén các file csv bên trong, giữ lại cấu trúc thư mục submission/
                zipf.write(file_path, arcname=os.path.join("submission", file))
                
    # Dọn dẹp thư mục tạm
    shutil.rmtree(submission_dir)
    
    print(f"\n🎉 HOÀN TẤT! File nộp bài của bạn là: {os.path.abspath(zip_filename)}")
    print("Bạn có thể cầm file này nộp thẳng lên hệ thống của BTC.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool tạo file ZIP nộp bài thi AIC 2026")
    parser.add_argument("--query_dir", type=str, required=True, help="Thư mục chứa các file .txt của BTC")
    parser.add_argument("--url", type=str, default="https://neat-parts-fall.loca.lt", help="URL của API Backend (Kaggle)")
    args = parser.parse_args()
    
    generate_submission(args.query_dir, args.url)
