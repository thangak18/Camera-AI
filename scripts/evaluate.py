import pandas as pd
import argparse

def calculate_r_score(predicted_frame: str, window_start: str, window_end: str) -> float:
    """
    R-Score is 1.0 if predicted frame falls within the answer window, else 0.
    In real scenarios, it might decrease if it's further away, but for simplicity we use 1/0.
    """
    try:
        p = int(predicted_frame)
        s = int(window_start)
        e = int(window_end)
        return 1.0 if s <= p <= e else 0.0
    except ValueError:
        return 0.0

def evaluate(submission_file: str, ground_truth_file: str) -> dict:
    """
    Evaluates a single query's submission against ground truth.
    ground_truth_file CSV format: video_id, window_start, window_end
    submission_file CSV format: video_id, frame_id
    """
    sub_df = pd.read_csv(submission_file, header=None, names=['video_id', 'frame_id'])
    gt_df = pd.read_csv(ground_truth_file, header=None, names=['video_id', 'window_start', 'window_end'])
    
    if gt_df.empty:
        return {"error": "Ground truth is empty"}
        
    gt_video_id = str(gt_df.iloc[0]['video_id'])
    gt_start = str(gt_df.iloc[0]['window_start'])
    gt_end = str(gt_df.iloc[0]['window_end'])
    
    scores = []
    for _, row in sub_df.iterrows():
        vid = str(row['video_id'])
        frame = str(row['frame_id'])
        if vid == gt_video_id:
            score = calculate_r_score(frame, gt_start, gt_end)
            scores.append(score)
        else:
            scores.append(0.0)
            
    # Calculate R@K
    def r_at_k(k: int):
        return max(scores[:k]) if len(scores) >= 1 else 0.0

    r1 = r_at_k(1)
    r5 = r_at_k(5)
    r20 = r_at_k(20)
    r50 = r_at_k(50)
    r100 = r_at_k(100)
    
    final_score = (r1 + r5 + r20 + r50 + r100) / 5.0
    
    return {
        "R@1": r1,
        "R@5": r5,
        "R@20": r20,
        "R@50": r50,
        "R@100": r100,
        "Final Score": final_score
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIC 2026 Evaluation Script")
    parser.add_argument("--submission", type=str, required=True, help="Path to submission CSV")
    parser.add_argument("--ground_truth", type=str, required=True, help="Path to ground truth CSV")
    
    args = parser.parse_args()
    results = evaluate(args.submission, args.ground_truth)
    
    print("Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
