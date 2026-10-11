"""Sinh submission.zip từ thư mục đề thi của BTC.

CÁCH CHÍNH — chạy thẳng trên máy có dữ liệu (đề tải về đâu thì chạy ở đó).
Không truyền `--local`/`--url` thì mặc định là local:

    python scripts/preflight.py                       # kiểm điều kiện, chạy TRƯỚC
    python scripts/generate_submission.py --query_dir <thư-mục-đề>
    python scripts/validate_submission.py --zip submission.zip         --query_dir <thư-mục-đề> --map_dir data/v1/map_keyframes

Cách phụ — từ máy KHÁC máy giữ dữ liệu, qua API:

    python scripts/generate_submission.py --query_dir <thư-mục-đề> --url https://xxx.loca.lt

VÌ SAO `--local` LÀ CÁCH CHÍNH: sinh bài nộp là job OFFLINE hàng chục phút; API
sinh ra cho vòng thi TRỰC TIẾP (<3s) và cho frontend. Ép job dài qua HTTP +
localtunnel là nguồn của sự cố 28/08: một truy vấn KIS mất ~2 phút khi Gemini
503, tunnel ngắt giữa chừng (502 Bad Gateway), rồi chính đường hầm hỏng luôn
nên mọi file sau đều `connect timeout`. `--local` bỏ hẳn lớp vận chuyển đó ra
khỏi bài toán, và không phải commit/push/clone gì trước khi chạy.

Chạy local cần: `data/v1/{keyframes,map_keyframes,siglip_features}` đầy đủ,
mạng tới Qdrant Cloud, và `GEMINI_API_KEY` trong `.env`. `preflight.py` kiểm
đủ cả bốn. Elasticsearch là tuỳ chọn — không có thì searcher tự bỏ nhánh
keyword, chỉ dùng vector.

BẤT BIẾN: mọi `frame_id` đi ra khỏi pipeline đều là **frame_idx THẬT**, đã tra
`map_keyframes` từ trước. Xem `to_real_frame_idx()` bên dưới.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trake.query_parser import TrakeQuery, parse_trake_query

# Nap .env TRUOC khi import bat cu thu gi doc config: `client._load_keys()` va
# `translator` deu doc GEMINI_API_KEY luc goi, nhung `probe_models` chay som.
# Chay local thi key nam trong .env chu khong phai bien moi truong cua shell.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MAX_ROWS = 100
MAX_ANSWER_LEN = 100
DEFAULT_ANSWER = "khong ro"


# src/common/frame_utils.to_real_frame_idx() giả định frame_id là số thứ tự
# keyframe ("n") và tra map_keyframes/<video_id>.csv để đổi sang frame_idx thật.
# Nhưng qdrant_loader.py (src/ingestion/qdrant_loader.py:126) đã gán thẳng
# frame_id = frame_idx ngay lúc nạp Qdrant, nên frame_id trả về từ pipeline ĐÃ
# LÀ frame thật — gọi hàm đó luôn tra "n" trật, fallback về int(frame_id) không
# đổi, tốn IO đọc CSV + spam warning mà không đổi được gì.
#
# BẤT BIẾN cả hệ phải giữ: mọi frame_id đi ra khỏi pipeline đều là frame_idx
# THẬT. Nơi duy nhất từng phá bất biến này là qa.reorder_by_frame() — nó chèn
# thêm dòng cho frame mà VLM chỉ ra, trước đây ghi "n" zero-pad; đã sửa để tự
# tra map_keyframes. Đừng khôi phục việc convert ở đây: convert hai lần cũng
# cho 0 điểm y như không convert.
def to_real_frame_idx(video_id: str, frame_id) -> int:
    return int(frame_id)


# ==========================================================================
# Hai transport, cùng một giao diện
# ==========================================================================
class LocalBackend:
    """Gọi thẳng pipeline trong cùng tiến trình. Không mạng, không tunnel."""

    name = "local"

    def __init__(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
        from app.services.search import (perform_kis_search, perform_qa_search,
                                         perform_trake_search)
        self._kis = perform_kis_search
        self._qa = perform_qa_search
        self._trake = perform_trake_search

    @staticmethod
    def _rows(results) -> list:
        """Pydantic model -> dict, để hai backend trả cùng một kiểu."""
        return [r if isinstance(r, dict) else r.model_dump() for r in results]

    def kis(self, query, top_k):
        return self._rows(self._kis(query, top_k))

    def qa(self, description, question, top_k):
        return self._rows(self._qa(description, question, top_k))

    def trake(self, event_queries, top_k):
        return self._rows(self._trake(event_queries, top_k=top_k))


class HttpBackend:
    """Gọi qua API. Dùng khi chạy từ máy khác máy giữ dữ liệu."""

    name = "http"

    # Trần tường minh. Bản cũ không đặt timeout: request treo vô hạn, và khi
    # proxy cắt giữa chừng thì lỗi hiện ra dưới dạng 502 rất khó lần.
    TIMEOUT = (10, 600)   # (connect, read)

    def __init__(self, url: str):
        import requests
        self._requests = requests
        self.url = url.rstrip("/")
        self.headers = {"Content-Type": "application/json",
                        "Bypass-Tunnel-Reminder": "true"}

    def _post(self, path: str, body: dict):
        res = self._requests.post(f"{self.url}{path}", json=body,
                                  headers=self.headers, timeout=self.TIMEOUT)
        res.raise_for_status()
        return res.json()

    def kis(self, query, top_k):
        return self._post("/search/kis", {"query": query, "top_k": top_k})

    def qa(self, description, question, top_k):
        # KHÔNG truyền n_vlm_calls/deadline: config.py là nguồn sự thật duy
        # nhất (profile offline = hỏi HẾT các nhóm). Bản cũ gửi cứng 40 nhóm —
        # từng là nâng cấp so với mặc định 5, nay là HẠ CẤP so với mặc định 0.
        return self._post("/search/qa", {"description": description,
                                         "question": question, "top_k": top_k})

    def trake(self, event_queries, top_k):
        return self._post("/search/trake", {"event_queries": event_queries,
                                            "top_k": top_k})


# ==========================================================================
# Đọc đề
# ==========================================================================
def parse_query_file(file_path: str):
    """Thử JSON trước; không phải JSON thì lấy toàn bộ nội dung làm text thuần."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def split_trake(query_data) -> TrakeQuery:
    """Tách file TRAKE thành các phân cảnh. Quy tắc đầy đủ ở
    `src/trake/query_parser.py`.

    Luật cũ ("mỗi dòng không rỗng là một event") sai trên đề Đợt 2: cả
    query-p2-8 lẫn query-p2-21 mở đầu bằng một dòng dẫn nhập TỰ KHAI số phân
    cảnh ("4 loại trái cây", "4 cảnh này xảy ra liên tiếp nhau") rồi mới tới 4
    dòng có nhãn — đếm theo dòng ra 5 cột. Lệch số cột là mất trắng cả file:
    log chấm Đợt 1 đã loại query-p1-16-trake với đúng lý do đó ("Expected 4
    frame IDs, got 3"). Thêm nữa query-p2-21 dùng nhãn "Cảnh <số>:" mà regex cũ
    chỉ biết "E<số>", nên chuỗi "Cảnh 1:" đi thẳng vào câu truy vấn SigLIP.

    Ca đã được BTC xác nhận ở Đợt 1 vẫn giữ nguyên: queries-1/query-p1-16 có
    dòng mở đầu KHÔNG nhãn mô tả một khoảnh khắc thật, không có dấu hiệu lời
    dẫn nào, nên vẫn được tính là event.
    """
    if not isinstance(query_data, str):
        return TrakeQuery(events=list(query_data.get("event_queries", [])),
                          context=[])
    return parse_trake_query(query_data)


def parse_events(query_data) -> list:
    """Chỉ danh sách phân cảnh, đã bỏ lời dẫn. Giữ tên cũ cho chỗ gọi cũ."""
    return split_trake(query_data).events


# ==========================================================================
# Giải một truy vấn
# ==========================================================================
def solve(q_type: str, query_data, backend):
    """Trả `(rows, notes)`. `notes` ghi lại mọi lần phải hạ cấp."""
    notes = []

    if q_type == "kis":
        query = query_data if isinstance(query_data, str) else query_data.get("query", "")
        return backend.kis(query, MAX_ROWS), notes

    if q_type == "qa":
        if isinstance(query_data, str):
            question, description = query_data, ""
        else:
            question = query_data.get("question_vi") or query_data.get("question", "")
            description = query_data.get("description_vi") or query_data.get("description", "")
        return backend.qa(description, question, MAX_ROWS), notes

    if q_type == "trake":
        parsed = split_trake(query_data)
        events = parsed.events
        # In ra cách tách để soi bằng mắt TRƯỚC khi tốn hàng chục phút chạy.
        # Sai số phân cảnh = sai số cột = mất trắng file, mà nhìn CSV xong thì
        # đã muộn: thể lệ chỉ cho nộp 3 lần/gói.
        print(f"    {len(events)} phan canh"
              + (f" (bo {len(parsed.context)} dong dan nhap)"
                 if parsed.context else ""))
        for i, event in enumerate(events, 1):
            print(f"      E{i}: {event[:88]}")
        # In luôn NỘI DUNG dòng bị bỏ, không chỉ số lượng. Dòng dẫn Đợt 3 hay
        # chứa manh mối phân biệt video ("con lân màu ĐỎ") mà không sự kiện nào
        # nhắc lại; thấy nó ở đây thì còn kịp gộp tay vào E1 trước khi chạy.
        for line in parsed.context:
            print(f"      [loi dan, khong tim] {line[:70]}")
        rows = backend.trake(events, MAX_ROWS)
        if not rows and events:
            # Solver TRAKE đòi 1 video có mặt trong top-100 của TẤT CẢ sự kiện;
            # sự kiện ngắn/mơ hồ dễ không có video nào khớp đủ nên trả rỗng.
            # File .csv rỗng có nguy cơ làm hệ thống BTC parse lỗi cả lượt nộp,
            # nên đoán bằng KIS của sự kiện đầu — thà đoán còn hơn để trống.
            notes.append("TRAKE rỗng, fallback KIS sự kiện đầu")
            kis_rows = backend.kis(events[0], 1)
            if kis_rows:
                top = kis_rows[0]
                rows = [{"video_id": top["video_id"],
                         "frame_ids": [top["frame_id"]] * len(events),
                         "score": top["score"]}]
        return rows, notes

    raise ValueError(f"loại truy vấn lạ: {q_type}")


def write_csv(path: str, q_type: str, rows) -> None:
    """Ghi CSV chuẩn BTC.

    QUOTE_MINIMAL cho mọi loại — khớp ví dụ trong thể lệ: chỉ answer mới quote
    khi có dấu phẩy/ngoặc kép/xuống dòng; video_id và frame_id (số) không nên
    bị quote thừa, tránh sai lệch nếu hệ thống BTC so khớp chuỗi thô.
    """
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for r in rows[:MAX_ROWS]:
            vid = r["video_id"]
            if q_type == "kis":
                w.writerow([vid, to_real_frame_idx(vid, r["frame_id"])])
            elif q_type == "qa":
                w.writerow([vid, to_real_frame_idx(vid, r["frame_id"]),
                            r.get("answer", "")[:MAX_ANSWER_LEN]])
            else:
                w.writerow([vid] + [to_real_frame_idx(vid, fid)
                                    for fid in r.get("frame_ids", [])])


# ==========================================================================
# Soi chất lượng — KHÔNG để bài nộp "trông thì ổn" lọt qua
# ==========================================================================
def audit(q_type: str, rows, notes) -> list:
    """Liệt kê dấu hiệu suy giảm. Rỗng = truy vấn này lành lặn."""
    bad = list(notes)
    if not rows:
        bad.append("KHONG CO DONG NAO")
        return bad

    if q_type in ("kis", "qa") and len(rows) < MAX_ROWS:
        bad.append(f"chi {len(rows)}/{MAX_ROWS} dong - tu bo phi luot doan")

    if q_type == "qa":
        answers = {r.get("answer", "") for r in rows}
        if answers <= {DEFAULT_ANSWER, ""}:
            bad.append("100% dong la 'khong ro' - VLM khong tra loi duoc")
        elif len(answers) == 1:
            bad.append(f"chi mot answer duy nhat: {answers.pop()!r}")

    if q_type == "trake" and rows:
        fids = rows[0].get("frame_ids", [])
        if len(fids) > 1 and len(set(fids)) == 1:
            bad.append("dong dau lap cung mot frame - degenerate")
    return bad


def summarize(rows, q_type: str) -> dict:
    return {"rows": len(rows),
            "videos": len({r["video_id"] for r in rows}) if rows else 0,
            "answers": (len({r.get("answer", "") for r in rows})
                        if q_type == "qa" and rows else None)}


# ==========================================================================
# Chạy cả bộ đề
# ==========================================================================
def generate_submission(query_dir: str, backend, retries: int,
                        out_zip: str, report_path: str) -> int:
    """Trả về EXIT CODE: 0 = sạch, khác 0 = có truy vấn hỏng hoặc suy giảm."""
    if not os.path.isdir(query_dir):
        print(f"[X] Khong tim thay thu muc {query_dir}")
        return 2

    files = sorted(f for f in os.listdir(query_dir) if f.endswith(".txt"))
    if not files:
        print(f"[X] Khong co file .txt nao trong {query_dir}")
        return 2

    submission_dir = "submission"
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir)

    try:
        from src.vlm import breaker_state, rate_state
    except Exception:
        def breaker_state():
            return {}

        def rate_state():
            return {}

    print(f"{len(files)} file de trong '{query_dir}' | backend = {backend.name}\n")
    report, failed, degraded = [], [], []

    for filename in files:
        basename = filename[:-4]
        q_type = next((t for t in ("kis", "qa", "trake")
                       if basename.endswith(f"-{t}")), None)
        if q_type is None:
            print(f"[!] bo qua {filename}: khong nhan ra loai truy van o duoi ten")
            continue

        query_data = parse_query_file(os.path.join(query_dir, filename))
        rows, notes, err, t0 = [], [], None, time.time()

        for attempt in range(1, retries + 2):
            t0 = time.time()
            try:
                rows, notes = solve(q_type, query_data, backend)
                err = None
                break
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                if attempt <= retries:
                    wait = 2.0 * attempt
                    print(f"    [!] {basename} lan {attempt} loi ({err[:80]}) "
                          f"- thu lai sau {wait:.0f}s")
                    time.sleep(wait)

        dt = time.time() - t0
        rec = {"file": basename, "type": q_type, "seconds": round(dt, 1),
               "error": err, "notes": notes, "breaker": breaker_state(),
               "rate": rate_state()}

        if err is not None:
            failed.append(basename)
            rec.update(rows=0, videos=0, answers=None, problems=[f"HONG: {err}"])
            print(f"[X] {basename:<24} HONG sau {retries + 1} lan thu - {err[:90]}")
        else:
            write_csv(os.path.join(submission_dir, f"{basename}.csv"), q_type, rows)
            problems = audit(q_type, rows, notes)
            rec.update(problems=problems, **summarize(rows, q_type))
            if problems:
                degraded.append(basename)
            mark = "[!]" if problems else "[v]"
            extra = f"   <- {'; '.join(problems)}" if problems else ""
            print(f"{mark} {basename:<24} {len(rows):>3} dong | "
                  f"{rec['videos']:>3} video | {dt:5.1f}s{extra}")
        report.append(rec)

    written = [r for r in report if r["error"] is None]
    if not written:
        print("\n[X] Khong tao duoc file CSV nao - khong nen gi ca.")
        shutil.rmtree(submission_dir)
        _save_report(report_path, report)
        return 1

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, names in os.walk(submission_dir):
            for name in names:
                zf.write(os.path.join(root, name),
                         arcname=os.path.join("submission", name))
    shutil.rmtree(submission_dir)
    _save_report(report_path, report)

    print(f"\n=> {out_zip}  ({len(written)}/{len(report)} truy van)")
    print(f"=> bao cao chi tiet: {report_path}")

    if failed or degraded:
        # Cố ý ồn ào. Thể lệ chỉ cho nộp 3 lần/gói và lấy điểm lần CUỐI, nên
        # một file thiếu hoặc một file toàn 'khong ro' lọt qua trong im lặng
        # đắt hơn nhiều so với việc phải đọc thêm mấy dòng cảnh báo.
        print("\n" + "=" * 62)
        if failed:
            print(f"[X] THIEU {len(failed)} file: {', '.join(failed)}")
        if degraded:
            print(f"[!] SUY GIAM {len(degraded)} file: {', '.join(degraded)}")
        print("KHONG nop file nay truoc khi hieu ro tung dong o tren.")
        print("=" * 62)
        return 1

    print("\nSach. Chay tiep validate_submission.py truoc khi nop.")
    return 0


def _save_report(path: str, report) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query_dir", required=True,
                    help="Thu muc chua file .txt cua BTC")
    # KHONG required: khong truyen gi = --local. Dem thi bot duoc mot thu phai
    # nho la bot duoc mot cho sai. Muon di duong HTTP thi phai noi ro bang --url.
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--local", action="store_true",
                     help="Goi thang pipeline trong tien trinh nay (MAC DINH)")
    src.add_argument("--url",
                     help="Chay qua API o may khac, vd https://xxx.loca.lt")
    ap.add_argument("--retries", type=int, default=2,
                    help="So lan thu lai moi truy van khi loi (mac dinh 2)")
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--report", default="outputs/submission_report.json")
    args = ap.parse_args()

    backend = HttpBackend(args.url) if args.url else LocalBackend()
    return generate_submission(args.query_dir, backend, args.retries,
                               args.out, args.report)


if __name__ == "__main__":
    sys.exit(main())
