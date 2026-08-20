import json
import os
import sys
import argparse
import requests
import csv
from pathlib import Path

# Thêm path để import evaluate (hàm tính điểm)
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from scripts.evaluate import evaluate

# Copy hàm to_real_frame_idx để script này có thể tự đứng vững mà không cần load Backend
def to_real_frame_idx(video_id: str, frame_idx: int) -> int:
    """Giả lập hàm chuyển đổi frame. Có thể tùy chỉnh thêm nếu cần."""
    return frame_idx

def export_api_results_to_csv(results: list, filepath: str, task_type: str = "kis"):
    """Xuất kết quả từ API Response (List[dict]) ra file CSV."""
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    with open(filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Max 100 rows allowed by BTC
        for result in results[:100]:
            if task_type == "kis":
                real_frame = to_real_frame_idx(result["video_id"], result["frame_id"])
                writer.writerow([result["video_id"], real_frame])
            elif task_type == "qa":
                real_frame = to_real_frame_idx(result["video_id"], result["frame_id"])
                writer.writerow([result["video_id"], real_frame, result.get("answer", "")])
            elif task_type == "trake":
                real_frames = [to_real_frame_idx(result["video_id"], f) for f in result.get("frame_ids", [])]
                row = [result["video_id"]] + real_frames
                writer.writerow(row)

def run_batch_evaluation(devset_path: str, ground_truth_path: str, output_csv: str, api_url: str):
    if not os.path.exists(devset_path):
        print(f"Lỗi: Không tìm thấy {devset_path}")
        return

    with open(devset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries = data.get("queries", data) if isinstance(data, dict) else data
    all_results = []
    
    print(f"Đang gọi Kaggle API ({api_url}) để chạy {len(queries)} queries...")
    
    headers = {
        'Content-Type': 'application/json',
        'Bypass-Tunnel-Reminder': 'true' # Bỏ qua màn hình cảnh báo của localtunnel
    }
    
    for idx, q in enumerate(queries):
        # Tự động nhận diện loại câu hỏi (BTC format vs My format)
        if "question_vi" in q or "question" in q:
            q_type = "qa"
            question = q.get("question_vi") or q.get("question", "")
            description = q.get("description_vi") or q.get("description", "")
        elif "event_queries" in q:
            q_type = "trake"
        else:
            q_type = "kis"
            query = q.get("query", "")
            
        print(f"[{idx+1}/{len(queries)}] Đang xử lý query type '{q_type}'...")
        
        try:
            if q_type == "kis":
                res = requests.post(
                    f"{api_url}/search/kis", 
                    json={"query": query, "top_k": 100},
                    headers=headers
                )
                res.raise_for_status()
                all_results.extend(res.json())
                
            elif q_type == "qa":
                res = requests.post(
                    f"{api_url}/search/qa", 
                    json={"description": description, "question": question, "top_k": 100},
                    headers=headers
                )
                res.raise_for_status()
                all_results.extend(res.json())
                
            elif q_type == "trake":
                res = requests.post(
                    f"{api_url}/search/trake", 
                    json={"event_queries": q["event_queries"], "top_k": 100},
                    headers=headers
                )
                res.raise_for_status()
                all_results.extend(res.json())
                
        except Exception as e:
            print(f"Lỗi khi gọi API cho query {idx+1}: {e}")
            
    export_api_results_to_csv(all_results, output_csv, task_type="kis")
    print(f"Đã xuất kết quả ra {output_csv}")

    # Chạy evaluation
    if os.path.exists(ground_truth_path):
        scores = evaluate(output_csv, ground_truth_path)
        print("\n--- ĐIỂM SỐ TRUNG BÌNH (DEVSET) ---")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")
    else:
        print(f"Không tìm thấy Ground Truth tại {ground_truth_path}, bỏ qua chấm điểm.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devset", type=str, default="data/v1/devset_v1.json")
    parser.add_argument("--gt", type=str, default="data/v1/ground_truth.csv")
    parser.add_argument("--out", type=str, default="submission_devset.csv")
    # Thay đổi URL này thành URL Localtunnel của bạn khi chạy
    parser.add_argument("--url", type=str, default="https://neat-parts-fall.loca.lt")
    args = parser.parse_args()
    
    run_batch_evaluation(args.devset, args.gt, args.out, args.url)
