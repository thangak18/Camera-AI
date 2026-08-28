"""Grid test Q&A trên mẫu ĐÃ CÓ ground truth thật (q17 — đèo Tà Pừa).

Đặt tại: scripts/diag_qa_grid.py

Khác với `diag_qa.py` (quét rộng, không biết đáp án đúng), script này chạy hẹp
trên đúng một mẫu mà nhóm đã xác minh bằng mắt, nên trả lời được câu hỏi cụ
thể: CẤU HÌNH NÀO ĐỌC RA "Tà Pừa"?

Bối cảnh mẫu:
    Video đúng      : L22_V008
    Frame GT        : 5540 5638 5745 5860 5966 6090  (n = 058..063)
    Đáp án          : Tà Pừa, chỉ hiện trên chyron ở n=059
    Đợt 1 gửi VLM   : n=061 (cs[0] của nhóm) — KHÔNG chứa chữ ⇒ "khong ro"

Ba chiến lược so nhau, cột "candidate" là cái quan trọng nhất vì nó không tốn
thêm I/O: các frame ấy vốn đã nằm trong Top-100.

    neighbors : n±k, đúng cách `answer_qa` đang làm
    candidate : chính các frame cùng video trong Top-100 (MỚI)
    both      : gộp cả hai, khử trùng

Chạy:
    python scripts/diag_qa_grid.py
    python scripts/diag_qa_grid.py --max-side 448 1024 --k 0 2 3 5
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import unicodedata

os.environ.setdefault("HF_HOME", "D:/AIC/hf_cache")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from PIL import Image

from src.common.frame_utils import keyframe_path
from src.vlm.client import call_gemini, parse_answer, probe_models
from src.vlm.qa import PROMPT_QA

logger = logging.getLogger(__name__)

VIDEO_ID = "L22_V008"
QUESTION = (
    "Đoạn clip mô tả hậu quả của hiện tượng sạt lở đất nghiêm trọng, đất đá "
    "tràn xuống gây tắc nghẽn hoàn toàn một đoạn đường đèo. Trong một cảnh quay "
    "cận, có thể thấy một cột mốc đường bộ với phần đỉnh màu đỏ chỉ còn lộ ra "
    "phần trên, phần lớn đã bị đất đá vùi lấp. Ở cảnh quay cuối, một người đi xe "
    "máy đang vật lộn để điều khiển phương tiện vượt qua khu vực bùn lầy, trên "
    "chiếc xe máy có treo một vật màu xanh lá. Tên của con đèo là gì?"
)
EXPECTED = "tà pứa"

# n của frame VLM thật sự nhận được ở Đợt 1 (cs[0] của nhóm L22_V008).
SEED_N = 61

# Các frame cùng video có trong Top-100 Đợt 1, theo đúng thứ tự xếp hạng.
# n = 58..63 là 6 frame ground truth; 69/74/78 là frame khác cùng video.
CANDIDATE_N = [61, 58, 60, 74, 59, 69, 65, 63, 64]


def norm(text: str) -> str:
    """Bỏ dấu, hạ chữ thường — để so đáp án không bị lệch vì kiểu gõ."""
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def load(video_id: str, ns: list[int]) -> tuple[list[Image.Image], list[int]]:
    """Nạp ảnh theo danh sách n, bỏ qua tấm thiếu. Trả (ảnh, n thực nạp được)."""
    imgs, got = [], []
    for n in ns:
        path = keyframe_path(video_id, n)
        try:
            imgs.append(Image.open(path).convert("RGB"))
            got.append(n)
        except Exception as exc:
            logger.debug("Bỏ qua %s: %s", path, exc)
    return imgs, got


def pick(strategy: str, k: int, limit: int) -> list[int]:
    """Chọn danh sách n theo chiến lược, giữ thứ tự ưu tiên, khử trùng."""
    neighbors = list(range(SEED_N - k, SEED_N + k + 1))
    if strategy == "neighbors":
        chosen = neighbors
    elif strategy == "candidate":
        chosen = CANDIDATE_N
    else:  # both
        chosen = CANDIDATE_N + neighbors

    seen, out = set(), []
    for n in chosen:
        if n > 0 and n not in seen:
            seen.add(n)
            out.append(n)
    return out[:limit]


def run_one(strategy: str, k: int, max_side: int, limit: int) -> dict:
    """Một ô của grid: chọn ảnh, hỏi Gemini, chấm đúng/sai."""
    ns = pick(strategy, k, limit)
    imgs, got = load(VIDEO_ID, ns)
    has_gt = 59 in got  # tấm duy nhất chứa chữ "TÀ PỨA"

    if not imgs:
        return {"answer": "(KHÔNG MỞ ĐƯỢC ẢNH)", "seconds": 0.0,
                "n_images": 0, "has_gt_frame": has_gt, "ok": False, "ns": got}

    t0 = time.perf_counter()
    raw = call_gemini(PROMPT_QA.format(question=QUESTION), images=imgs,
                      timeout_s=45.0, max_side=max_side)
    dt = time.perf_counter() - t0
    answer, _ = parse_answer(raw)

    return {"answer": answer, "seconds": dt, "n_images": len(imgs),
            "has_gt_frame": has_gt, "ok": norm(EXPECTED) in norm(answer), "ns": got}


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, nargs="+", default=[0, 2, 3, 5])
    ap.add_argument("--max-side", type=int, nargs="+", default=[448, 1024])
    ap.add_argument("--strategies", nargs="+",
                    default=["neighbors", "candidate", "both"])
    ap.add_argument("--limit", type=int, default=9,
                    help="Trần số ảnh mỗi lần gọi (mặc định 9)")
    args = ap.parse_args()

    print("Dò model Gemini còn sống...")
    alive = probe_models()
    if not alive:
        print("❌ Không model nào khả dụng — kiểm tra .env trước đã.")
        return
    print(f"OK: {alive}\n")

    print(f"Mẫu: {VIDEO_ID}, đáp án đúng = 'Tà Pừa' (chỉ hiện ở n=059)")
    print(f"Đợt 1 gửi VLM đúng 1 tấm n={SEED_N:03d} ⇒ khong ro\n")
    print(f"{'chiến lược':<11}{'k':>2}{'px':>6}{'ảnh':>5}{'có 059':>8}"
          f"{'giây':>7}  {'answer':<24}kết quả")
    print("-" * 78)

    results = []
    for strategy in args.strategies:
        ks = args.k if strategy != "candidate" else [0]  # candidate không dùng k
        for k in ks:
            for side in args.max_side:
                r = run_one(strategy, k, side, args.limit)
                r.update(strategy=strategy, k=k, max_side=side)
                results.append(r)
                mark = "✅ ĐÚNG" if r["ok"] else "❌"
                print(f"{strategy:<11}{k:>2}{side:>6}{r['n_images']:>5}"
                      f"{'có' if r['has_gt_frame'] else '—':>8}"
                      f"{r['seconds']:>7.1f}  {r['answer'][:22]:<24}{mark}")

    print("-" * 78)
    good = [r for r in results if r["ok"]]
    if not good:
        print("Không cấu hình nào ra đúng. Kiểm tra: n=059 có nằm trong danh sách\n"
              "ảnh gửi đi không (cột 'có 059')? Nếu CÓ mà vẫn sai thì lỗi ở prompt\n"
              "hoặc độ nét, không phải ở việc chọn frame.")
        return

    cheap = min(good, key=lambda r: (r["n_images"], r["max_side"]))
    print(f"Rẻ nhất mà vẫn đúng: {cheap['strategy']} k={cheap['k']} "
          f"{cheap['max_side']}px, {cheap['n_images']} ảnh, {cheap['seconds']:.1f}s")
    print("\n⚠️ Đây là MỘT mẫu. Nó đủ để bác bỏ k=0, chưa đủ để chốt tham số.\n"
          "   Chốt tham số sau khi có thêm mẫu ở T3.1.")


if __name__ == "__main__":
    main()