import os
import re
import sys
import json
import argparse
import requests
import csv
import zipfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# src/common/frame_utils.to_real_frame_idx() giả định frame_id là số thứ tự
# keyframe ("n") và tra map_keyframes/<video_id>.csv để đổi sang frame_idx thật.
# Nhưng qdrant_loader.py (src/ingestion/qdrant_loader.py:126) đã gán thẳng
# frame_id = frame_idx ngay lúc nạp Qdrant, nên frame_id trả về từ API ĐÃ LÀ
# frame thật - gọi hàm đó luôn tra "n" trật (vì input không phải n), fallback về
# int(frame_id) không đổi - tốn IO đọc CSV + spam warning mà không đổi được gì.
#
# BẤT BIẾN cả hệ phải giữ: mọi frame_id đi ra khỏi API đều là frame_idx THẬT.
# Nơi duy nhất từng phá bất biến này là qa.reorder_by_frame() - nó chèn thêm
# dòng cho frame mà VLM chỉ ra, trước đây ghi "n" zero-pad; đã sửa để tự tra
# map_keyframes trước khi trả về. Đừng khôi phục việc convert ở đây: convert
# hai lần cũng cho 0 điểm y như không convert.
def to_real_frame_idx(video_id: str, frame_id) -> int:
    return int(frame_id)

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
                # KHÔNG truyền n_vlm_calls/deadline nữa: config.py đã đổi mặc
                # định sang profile offline (QA_N_VLM_CALLS=0 = hỏi HẾT các nhóm,
                # QA_DEADLINE=300s). Bản cũ gửi cứng 40 nhóm - con số đó từng là
                # nâng cấp so với mặc định 5, nay lại là HẠ CẤP so với mặc định 0
                # và âm thầm cắt bớt nhóm được hỏi. Để config làm nguồn sự thật
                # duy nhất, đổi profile bằng biến môi trường ở phía server.
                json={"description": description, "question": question, "top_k": 100},
                headers=headers
            )
            res.raise_for_status()
            results = res.json()
            
        elif q_type == "trake":
            # Trake thường là danh sách sự kiện
            event_queries = []
            if isinstance(query_data, str):
                # MỖI DÒNG KHÔNG RỖNG LÀ MỘT EVENT - kể cả dòng mở đầu không có
                # nhãn "E<số>". Log chấm của BTC ở đợt 1 báo query-p1-16-trake
                # "Expected 4 frame IDs, got 3": file đề có 4 dòng (1 dòng mô tả
                # cảnh mở đầu + E1..E3), BTC đếm đủ 4. Lọc theo nhãn E<số> sẽ
                # thiếu đúng 1 cột -> cả file bị loại khỏi vòng chấm.
                # Nhãn có thể có ":" (đề thử nghiệm) hoặc không (đợt 1) - bỏ
                # tiền tố đi trước khi đưa vào search, giữ nguyên phần mô tả.
                for line in query_data.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    m = re.match(r'^E\d+\s*[:.)-]?\s+(.*)', line)
                    event_queries.append(m.group(1).strip() if m else line)
            else:
                event_queries = query_data.get("event_queries", [])
                
            res = requests.post(
                f"{api_url}/search/trake",
                json={"event_queries": event_queries, "top_k": 100},
                headers=headers
            )
            res.raise_for_status()
            results = res.json()

            # Solver TRAKE yêu cầu 1 video xuất hiện trong top-100 candidate của
            # TẤT CẢ sự kiện - với sự kiện ngắn/mơ hồ dễ không có video nào khớp
            # đủ, trả về rỗng. File .csv rỗng có nguy cơ làm hệ thống BTC parse
            # lỗi cả lượt nộp, nên fallback về KIS của riêng sự kiện đầu tiên,
            # lặp lại cùng 1 frame cho đủ N events - thà đoán còn hơn để trống.
            if not results and event_queries:
                print(f"   ⚠️ TRAKE rỗng cho {os.path.basename(output_csv)} - fallback KIS sự kiện đầu tiên.")
                kis_res = requests.post(
                    f"{api_url}/search/kis",
                    json={"query": event_queries[0], "top_k": 1},
                    headers=headers
                )
                kis_res.raise_for_status()
                kis_results = kis_res.json()
                if kis_results:
                    top = kis_results[0]
                    results = [{
                        "video_id": top["video_id"],
                        "frame_ids": [top["frame_id"]] * len(event_queries),
                        "score": top["score"],
                    }]

    except Exception as e:
        print(f"❌ Lỗi khi gọi API cho file {os.path.basename(output_csv)}: {e}")
        return False

    # Ghi ra file CSV chuẩn BTC
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        # QUOTE_MINIMAL cho mọi loại - đúng khớp ví dụ chuẩn trong thể lệ (chỉ
        # answer mới quote khi có dấu phẩy/ngoặc kép/xuống dòng; video_id và
        # frame_id (số) không nên bị quote thừa, tránh sai lệch nếu hệ thống
        # BTC không parse CSV chuẩn mà so khớp chuỗi thô).
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

        for result in results[:100]:
            if q_type == "kis":
                real_frame = to_real_frame_idx(result["video_id"], result["frame_id"])
                writer.writerow([result["video_id"], real_frame])
            elif q_type == "qa":
                real_frame = to_real_frame_idx(result["video_id"], result["frame_id"])
                # Thể lệ: answer tối đa 100 ký tự.
                answer = result.get("answer", "")[:100]
                writer.writerow([result["video_id"], real_frame, answer])
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
    parser.add_argument("--url", type=str, default="https://small-lines-sip.loca.lt/", help="URL của API Backend (Kaggle)")
    args = parser.parse_args()
    
    generate_submission(args.query_dir, args.url)
