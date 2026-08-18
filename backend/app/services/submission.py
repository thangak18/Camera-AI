import csv
from typing import List, Any
import os

def export_to_csv(results: List[Any], filepath: str, task_type: str = "kis"):
    """
    Exports the search results to a CSV file complying with BTC requirements.
    results: List of SearchResult, QAResponse, or TrakeResponse.
    task_type: "kis", "qa", or "trake"
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Max 100 rows allowed by BTC
        for result in results[:100]:
            if task_type == "kis":
                # KIS format: video_id, frame_id
                writer.writerow([result.video_id, result.frame_id])
            elif task_type == "qa":
                # QA format: video_id, frame_id, answer
                writer.writerow([result.video_id, result.frame_id, result.answer])
            elif task_type == "trake":
                # Trake format: video_id, frame_id_1, ..., frame_id_N
                row = [result.video_id] + result.frame_ids
                writer.writerow(row)

