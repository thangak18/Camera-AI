"""Chẩn đoán vì sao Q&A Đợt 1 trả "khong ro" — chạy trên file đã nộp thật.

Đặt tại: scripts/diag_qa.py

Trả lời đúng một câu hỏi: Gemini trả "khong ro" vì KHÔNG MỞ ĐƯỢC ẢNH, hay vì
MỞ ĐƯỢC MÀ KHÔNG ĐỌC NỔI? Hai nguyên nhân này cần hai cách sửa hoàn toàn khác
nhau, nên phải phân biệt trước khi đụng vào `qa.py`.

Đo 3 cấu hình, mỗi lần chỉ đổi MỘT biến so với cấu hình trước:

    A  k=0, 448px   ← đúng cấu hình đã chạy Đợt 1 (baseline)
    B  k=0, 1024px  ← đổi độ nét ảnh
    C  k=3, 1024px  ← thêm 6 keyframe lân cận

Chạy:
    python scripts/diag_qa.py --sub submission/ --videos 5
    python scripts/diag_qa.py --sub submission/ --only p1-17 --videos 8

Chi phí quota: videos × 4 câu × 3 cấu hình call Gemini (mặc định 60 call).
Có 5 key xoay vòng nên an toàn, nhưng đừng để mặc định --videos quá lớn.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path

# PHẢI đặt trước mọi import transformers/huggingface.
os.environ.setdefault("HF_HOME", "D:/AIC/hf_cache")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.common.config import MAP_KEYFRAMES_DIR
from src.common.frame_utils import keyframe_path
from src.vlm import client as vlm_client
from src.vlm.client import DEFAULT_ANSWER, call_gemini, parse_answer, probe_models
from src.vlm.qa import PROMPT_QA, load_neighbors

logger = logging.getLogger(__name__)

# Câu hỏi thật của Đợt 1, chép nguyên văn từ đề (giữ tiếng Việt gốc).
QUESTIONS: dict[str, str] = {
    "p1-3": (
        "Hình ảnh một con cá được đặt lên cân, sau đó có cảnh một con cá khác "
        "cùng loại bị một người cầm đuôi. Con số hiển thị cuối cùng trên cân là "
        "bao nhiêu?"
    ),
    "p1-9": (
        "Đoạn phim ghi lại cảnh những chiếc xe ô tô lội nước, chiếc xe màu vàng, "
        "màu đỏ và màu đen lần lượt chuẩn bị đi qua cầu. Con số được ghi trên "
        "biển báo bên trái của cây cầu là bao nhiêu?"
    ),
    "p1-15": (
        "Bản đồ phân bố động đất tại một vùng trên thế giới, với một bảng chú "
        "giải phía bên trái gồm các kí hiệu nhiều màu sắc tương ứng với các vị "
        "trí tâm chấn có cấp độ động đất khác nhau. Không tính bảng chú giải, "
        "có bao nhiêu vị trí ghi nhận động đất cấp độ 4?"
    ),
    "p1-17": (
        "Đoạn clip mô tả hậu quả của hiện tượng sạt lở đất nghiêm trọng, đất đá "
        "tràn xuống gây tắc nghẽn hoàn toàn một đoạn đường đèo. Trong một cảnh "
        "quay cận, có thể thấy một cột mốc đường bộ với phần đỉnh màu đỏ chỉ còn "
        "lộ ra phần trên, phần lớn đã bị đất đá vùi lấp. Ở cảnh quay cuối, một "
        "người đi xe máy đang vật lộn để điều khiển phương tiện vượt qua khu vực "
        "bùn lầy, trên chiếc xe máy có treo một vật màu xanh lá. Tên của con đèo "
        "là gì?"
    ),
}

# (tên cấu hình, k_neighbors, max_side)
CONFIGS: list[tuple[str, int, int]] = [
    ("A k=0 448px", 0, 448),
    ("B k=0 1024px", 0, 1024),
    ("C k=3 1024px", 3, 1024),
]

_rev_cache: dict[str, dict[int, int]] = {}


def to_n(video_id: str, frame_idx: int | str) -> int | None:
    """Đổi ngược frame_idx (giá trị trong file nộp) về `n` (tên file ảnh).

    Chiều xuôi `n -> frame_idx` là duy nhất, chiều ngược thì KHÔNG (Nhân đo được
    614/177.321 ca trùng). Dùng được ở đây vì mục đích chỉ là mở đúng tấm ảnh để
    chẩn đoán, không phải để ghi ra file nộp. Trùng thì lấy `n` nhỏ nhất.

    Returns:
        Số thứ tự keyframe, hoặc None nếu không tra được.
    """
    if video_id not in _rev_cache:
        path = MAP_KEYFRAMES_DIR / f"{video_id}.csv"
        table: dict[int, int] = {}
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    table.setdefault(int(row["frame_idx"]), int(row["n"]))
        except Exception as exc:
            logger.warning("Không đọc được %s: %s", path, exc)
        _rev_cache[video_id] = table
    return _rev_cache[video_id].get(int(frame_idx))


def first_frame_per_video(csv_path: Path, limit: int) -> list[tuple[str, int]]:
    """Frame đầu tiên của `limit` video đầu, giữ nguyên thứ tự trong file nộp."""
    groups: OrderedDict[str, int] = OrderedDict()
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                groups.setdefault(row[0], int(row[1]))
    return list(groups.items())[:limit]


def ask(question_vi: str, video_id: str, n: int, k: int,
        max_side: int) -> tuple[str, float, int]:
    """Một lần hỏi Gemini. Trả (answer, giây, số ảnh gửi đi).

    `max_side` được áp bằng cách gán biến module của client — `_to_part()` đọc
    nó ở mỗi lần gọi. Đây là cách tạm cho bước chẩn đoán; T1.4 sẽ đưa nó thành
    tham số đàng hoàng.
    """
    vlm_client.MAX_IMAGE_SIDE = max_side
    imgs = load_neighbors(video_id, n, k)
    if not imgs:
        return "(KHÔNG MỞ ĐƯỢC ẢNH)", 0.0, 0

    t0 = time.perf_counter()
    raw = call_gemini(PROMPT_QA.format(question=question_vi), images=imgs,
                      timeout_s=30.0)
    dt = time.perf_counter() - t0
    answer, _ = parse_answer(raw)
    return answer, dt, len(imgs)


def run(sub_dir: Path, videos: int, only: str | None) -> list[dict]:
    """Chạy chẩn đoán, in bảng, trả về các dòng kết quả."""
    rows: list[dict] = []

    for qid, question in QUESTIONS.items():
        if only and only not in qid:
            continue
        csv_path = sub_dir / f"query-{qid}-qa.csv"
        if not csv_path.exists():
            print(f"⚠️  Không thấy {csv_path}, bỏ qua.")
            continue

        print(f"\n{'=' * 78}\n{qid}  —  {question[:70]}...\n{'=' * 78}")
        header = f"{'video':<12}{'n':>5}  " + "".join(f"{c[0]:<26}" for c in CONFIGS)
        print(header)

        for video_id, frame_idx in first_frame_per_video(csv_path, videos):
            n = to_n(video_id, frame_idx)
            if n is None:
                print(f"{video_id:<12}{'?':>5}  ← KHÔNG TRA ĐƯỢC n từ frame_idx={frame_idx}")
                rows.append({"query": qid, "video_id": video_id, "n": None,
                             "config": "-", "answer": "KHÔNG TRA ĐƯỢC n",
                             "seconds": 0.0, "n_images": 0})
                continue

            exists = keyframe_path(video_id, n).exists()
            cells = []
            for name, k, side in CONFIGS:
                answer, dt, n_imgs = ask(question, video_id, n, k, side)
                rows.append({"query": qid, "video_id": video_id, "n": n,
                             "config": name, "answer": answer,
                             "seconds": round(dt, 1), "n_images": n_imgs})
                cells.append(f"{answer[:20]:<20}{dt:>4.1f}s ")

            flag = "" if exists else "  ← THIẾU FILE ẢNH"
            print(f"{video_id:<12}{n:>5}  " + "".join(cells) + flag)

    return rows


def summarize(rows: list[dict]) -> None:
    """In kết luận: tỉ lệ 'khong ro' theo từng cấu hình."""
    print(f"\n{'=' * 78}\nTỔNG KẾT\n{'=' * 78}")

    no_image = sum(1 for r in rows if r["n_images"] == 0 and r["config"] != "-")
    no_n = sum(1 for r in rows if r["config"] == "-")
    print(f"Không tra được n         : {no_n}")
    print(f"Tra được n nhưng thiếu ảnh: {no_image // max(len(CONFIGS), 1)}")

    for name, _, _ in CONFIGS:
        sub = [r for r in rows if r["config"] == name and r["n_images"] > 0]
        if not sub:
            continue
        bad = sum(1 for r in sub if r["answer"].strip().lower() == DEFAULT_ANSWER)
        avg = sum(r["seconds"] for r in sub) / len(sub)
        print(f"{name:<16} {len(sub) - bad:>3}/{len(sub):<3} có answer thật"
              f"   ({bad} 'khong ro')   trung bình {avg:.1f}s")

    print(
        "\nCách đọc:\n"
        "  • Phần lớn là 'thiếu ảnh' / 'không tra được n'  → lỗi dữ liệu, sửa ở\n"
        "    payload Qdrant và đường dẫn keyframes (việc của Nhân).\n"
        "  • Ảnh mở được nhưng A toàn 'khong ro' còn B/C khá hơn  → lỗi độ nét\n"
        "    và cửa sổ thời gian, sửa ở T1.4 (MAX_IMAGE_SIDE + k_neighbors).\n"
        "  • Cả A, B, C đều 'khong ro'  → retrieval chưa đưa đúng video vào\n"
        "    Top-100, không phải lỗi VLM. Báo lại Gia Phước.\n"
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sub", type=Path, required=True,
                    help="Thư mục chứa các file query-p1-*-qa.csv đã nộp Đợt 1")
    ap.add_argument("--videos", type=int, default=5,
                    help="Số video đầu mỗi câu (mặc định 5)")
    ap.add_argument("--only", default=None,
                    help="Chỉ chạy một câu, vd: p1-17")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/v1/eval/diag_qa.csv"))
    args = ap.parse_args()

    print("Dò model Gemini còn sống...")
    alive = probe_models()
    if not alive:
        print("❌ Không model nào khả dụng — kiểm tra GEMINI_API_KEY trước đã.")
        return
    print(f"OK: {alive}")

    rows = run(args.sub, args.videos, args.only)
    summarize(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Đã ghi chi tiết: {args.out}")


if __name__ == "__main__":
    main()
