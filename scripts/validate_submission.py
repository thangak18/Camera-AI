"""Soi file submission.zip TRƯỚC khi nộp lên hệ thống BTC.

Mỗi gói truy vấn chỉ được nộp 3 lần và điểm lấy theo lần CUỐI, nên không được
phép "nộp thử xem sao". Script này chặn đúng những lỗi đã thực sự làm nhóm mất
điểm ở vòng thử nghiệm:

  * Điểm 0.00 tuyệt đối vì 96% frame_id nộp lên không phải keyframe chính thức
    của BTC (index cũ tự trích lưới đều ~37 frame thay vì dùng map_keyframes).
    -> check FRAME_ID_IN_MAP bắt đúng ca này.
  * Cả 3 file Q&A đều ghi "khong ro" 100/100 dòng (VLM không mở được ảnh).
    -> check QA_DEFAULT_ANSWER.
  * File TRAKE chỉ có 1 dòng, lặp lại cùng một frame 4 lần.
    -> check TRAKE_DEGENERATE / TRAKE_MONOTONIC.

Dùng:
    python scripts/validate_submission.py --zip submission.zip \
        --query_dir docs/v1/queries-0 --map_dir data/v1/map_keyframes

Thoát mã 0 = an toàn để nộp. Khác 0 = CÒN LỖI, đừng nộp.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import zipfile
from collections import Counter

VIDEO_RE = re.compile(r"^[A-Z]\d+_V\d+$")
MAX_ROWS = 100
MAX_ANSWER_LEN = 100
DEFAULT_ANSWER = "khong ro"

errors: list[str] = []
warnings: list[str] = []


def err(f: str, code: str, msg: str) -> None:
    errors.append(f"  [{code}] {f}: {msg}")


def warn(f: str, code: str, msg: str) -> None:
    warnings.append(f"  [{code}] {f}: {msg}")


def load_valid_frames(map_dir: str, video_id: str, cache: dict) -> set[str] | None:
    """Tập frame_idx hợp lệ của một video; None nếu video không có trong map."""
    if video_id not in cache:
        path = os.path.join(map_dir, f"{video_id}.csv")
        if not os.path.exists(path):
            cache[video_id] = None
        else:
            with open(path, newline="", encoding="utf-8") as fh:
                cache[video_id] = {r["frame_idx"].strip() for r in csv.DictReader(fh)}
    return cache[video_id]


def expected_events(query_dir: str, basename: str) -> int | None:
    """Số event yêu cầu = số DÒNG KHÔNG RỖNG trong file đề.

    Đếm theo dòng chứ không theo nhãn 'E<số>': log chấm của BTC ở đợt 1 báo
    query-p1-16-trake "Expected 4 frame IDs, got 3" trong khi file chỉ có nhãn
    E1..E3 — dòng mô tả cảnh mở đầu (không nhãn) cũng được tính là 1 event.
    Đếm theo nhãn còn hỏng ở ca đánh nhầm nhãn (query-p1-18 đợt thử nghiệm ghi
    E2 hai lần, không có E3).
    """
    if not query_dir:
        return None
    path = os.path.join(query_dir, f"{basename}.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def check_file(name: str, raw: bytes, qtype: str, basename: str,
               map_dir: str, query_dir: str, cache: dict) -> None:
    if raw.startswith(b"\xef\xbb\xbf"):
        err(name, "BOM", "file có BOM UTF-8 - BTC yêu cầu UTF-8 thuần")
        raw = raw[3:]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        err(name, "ENCODING", f"không decode được UTF-8 ({exc})")
        return

    if not text.strip():
        err(name, "EMPTY", "file rỗng - mất trắng truy vấn này")
        return

    rows = [r for r in csv.reader(io.StringIO(text)) if r]

    if len(rows) > MAX_ROWS:
        err(name, "TOO_MANY_ROWS", f"{len(rows)} dòng > {MAX_ROWS}")
    if qtype in ("kis", "qa") and len(rows) < MAX_ROWS:
        warn(name, "FEW_ROWS",
             f"chỉ {len(rows)}/{MAX_ROWS} dòng - bỏ phí {MAX_ROWS - len(rows)} lượt đoán")

    if rows and not VIDEO_RE.match(rows[0][0].strip()):
        err(name, "HEADER",
            f"dòng đầu '{rows[0][0][:30]}' không phải video_id - BTC yêu cầu không có header")

    if rows and len(rows[0]) == 1 and ";" in rows[0][0]:
        err(name, "DELIMITER", "phân cách bằng ';' thay vì ','")
        return

    n_events = expected_events(query_dir, basename) if qtype == "trake" else None
    expected_cols = {"kis": 2, "qa": 3}.get(qtype)
    if qtype == "trake":
        expected_cols = 1 + n_events if n_events else None

    qa_answers: list[str] = []
    videos: list[str] = []

    for i, row in enumerate(rows, 1):
        tag = f"dòng {i}"

        if expected_cols and len(row) != expected_cols:
            extra = f" (1 video + {n_events} event)" if qtype == "trake" else ""
            err(name, "COLUMNS", f"{tag} có {len(row)} cột, cần {expected_cols}{extra}")
            continue

        video_id = row[0].strip()
        videos.append(video_id)

        if video_id.endswith(".mp4"):
            err(name, "MP4_SUFFIX", f"{tag} video_id còn đuôi .mp4")
            continue
        if not VIDEO_RE.match(video_id):
            err(name, "VIDEO_FORMAT", f"{tag} video_id '{video_id}' sai định dạng")
            continue

        valid = load_valid_frames(map_dir, video_id, cache)
        if valid is None:
            err(name, "VIDEO_NOT_FOUND",
                f"{tag} video '{video_id}' không có trong map_keyframes")
            continue

        frame_cells = row[1:] if qtype == "trake" else row[1:2]
        frames: list[int] = []
        for cell in frame_cells:
            if cell != cell.strip():
                err(name, "FRAME_WHITESPACE", f"{tag} frame_id '{cell}' có khoảng trắng thừa")
            s = cell.strip()
            if not re.fullmatch(r"\d+", s):
                err(name, "FRAME_NOT_INT", f"{tag} frame_id '{s}' không phải số nguyên")
                continue
            frames.append(int(s))
            # CHECK QUAN TRỌNG NHẤT - chính nó bắt được lỗi đã làm nhóm về 0.00
            if s not in valid:
                err(name, "FRAME_ID_IN_MAP",
                    f"{tag} frame_id {s} KHÔNG phải keyframe chính thức của {video_id}")

        if qtype == "trake" and frames and len(frames) == len(frame_cells):
            if len(set(frames)) == 1:
                err(name, "TRAKE_DEGENERATE",
                    f"{tag} lặp lại cùng một frame {frames[0]} cho mọi event")
            elif frames != sorted(frames) or len(set(frames)) != len(frames):
                err(name, "TRAKE_MONOTONIC",
                    f"{tag} frame không tăng dần nghiêm ngặt: {frames}")

        if qtype == "qa":
            answer = row[2]
            qa_answers.append(answer.strip())
            if not answer.strip():
                err(name, "QA_EMPTY_ANSWER", f"{tag} answer rỗng - dòng này chắc chắn 0 điểm")
            if len(answer) > MAX_ANSWER_LEN:
                err(name, "QA_ANSWER_TOO_LONG",
                    f"{tag} answer {len(answer)} ký tự > {MAX_ANSWER_LEN}")

    if qtype == "qa" and qa_answers:
        n_def = sum(1 for a in qa_answers if a.lower() == DEFAULT_ANSWER)
        ratio = n_def / len(qa_answers)
        if ratio == 1.0:
            err(name, "QA_DEFAULT_ANSWER",
                f"100% answer là '{DEFAULT_ANSWER}' - VLM không chạy, "
                f"truy vấn này chắc chắn 0 điểm")
        elif ratio > 0.5:
            warn(name, "QA_DEFAULT_ANSWER",
                 f"{ratio * 100:.0f}% answer là '{DEFAULT_ANSWER}' - VLM hỏng một phần")

    if qtype == "kis" and videos:
        counts = Counter(videos)
        top_id, top_n = counts.most_common(1)[0]
        if top_n > len(videos) * 0.5:
            warn(name, "LOW_DIVERSITY",
                 f"{top_n}/{len(videos)} dòng dồn vào một video ({top_id}); "
                 f"chỉ {len(counts)} video khác nhau - sai video là mất trắng cả 100 lượt")


def main() -> int:
    ap = argparse.ArgumentParser(description="Kiểm tra submission.zip trước khi nộp AIC 2026")
    ap.add_argument("--zip", default="submission.zip")
    ap.add_argument("--query_dir", default="docs/v1/queries-0",
                    help="Thư mục .txt của BTC (đối chiếu đủ file và số event TRAKE)")
    ap.add_argument("--map_dir", default="data/v1/map_keyframes")
    args = ap.parse_args()

    print(f"Kiểm tra: {args.zip}\n")

    if not os.path.exists(args.zip):
        print(f"[X] Không tìm thấy {args.zip}")
        return 1
    if not args.zip.lower().endswith(".zip"):
        err(args.zip, "NOT_ZIP", "BTC chỉ chấp nhận định dạng .zip")
    if not os.path.isdir(args.map_dir):
        print(f"[X] Không tìm thấy map_keyframes: {args.map_dir}")
        return 1

    try:
        zf = zipfile.ZipFile(args.zip)
    except zipfile.BadZipFile:
        print(f"[X] {args.zip} không phải file zip hợp lệ")
        return 1

    names = [n for n in zf.namelist() if not n.endswith("/")]

    for n in names:
        if n.startswith("__MACOSX/"):
            warn(n, "JUNK", "rác macOS trong zip, nên xoá")
            continue
        parts = n.split("/")
        if len(parts) != 2 or parts[0] != "submission":
            err(n, "STRUCTURE",
                "phải nằm trực tiếp trong thư mục 'submission/' (không nén thẳng file CSV)")
        if not n.lower().endswith(".csv"):
            err(n, "NOT_CSV", "chỉ được chứa file .csv (không .xlsx/.xls)")

    csvs = {os.path.basename(n)[:-4]: n
            for n in names
            if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")}

    if args.query_dir and os.path.isdir(args.query_dir):
        queries = {f[:-4] for f in os.listdir(args.query_dir) if f.endswith(".txt")}
        for missing in sorted(queries - set(csvs)):
            err(f"{missing}.csv", "MISSING_FILE", "có file đề nhưng thiếu file kết quả")
        for extra in sorted(set(csvs) - queries):
            warn(f"{extra}.csv", "EXTRA_FILE", "không khớp file đề nào")
        print(f"   Đề: {len(queries)} file  |  Kết quả: {len(csvs)} file")
    else:
        print(f"   (bỏ qua đối chiếu đề - không có {args.query_dir})")
        print(f"   Kết quả: {len(csvs)} file")

    cache: dict = {}
    for basename in sorted(csvs):
        qtype = next((t for t in ("kis", "qa", "trake") if basename.endswith(f"-{t}")), None)
        if qtype is None:
            warn(f"{basename}.csv", "UNKNOWN_TYPE", "đuôi tên không phải -kis/-qa/-trake")
            continue
        check_file(f"{basename}.csv", zf.read(csvs[basename]), qtype,
                   basename, args.map_dir, args.query_dir, cache)

    print()
    if warnings:
        print(f"[!] {len(warnings)} cảnh báo:")
        for w in warnings[:40]:
            print(w)
        if len(warnings) > 40:
            print(f"  ... và {len(warnings) - 40} cảnh báo nữa")
        print()

    if errors:
        print(f"[X] {len(errors)} LỖI - KHÔNG ĐƯỢC NỘP:")
        for e in errors[:40]:
            print(e)
        if len(errors) > 40:
            print(f"  ... và {len(errors) - 40} lỗi nữa")
        print("\n   Tổng hợp theo mã lỗi:")
        counts = Counter(re.search(r"\[(\w+)\]", e).group(1) for e in errors)
        for code, cnt in counts.most_common():
            print(f"     {code:22s} {cnt}")
        return 1

    print("[OK] Không có lỗi. File hợp lệ theo thể lệ, an toàn để nộp.")
    if warnings:
        print("     (vẫn nên xem lại các cảnh báo ở trên để tối đa điểm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
