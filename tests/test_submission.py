"""Test runner sinh bai nop - offline, khong can pipeline that.

    python tests/test_submission.py

Chot lai dung nhung gi tung lot qua trong im lang:

  * Mot truy van loi -> ban cu in mot dong do roi BO HAN file CSV do. Bai nop
    thieu file ma exit code van 0, nen khong co gi chan lai.
  * Q&A tra 'khong ro' ca 100 dong (ca 4/4 file Dot 1) van duoc dong goi nhu
    binh thuong.
  * TRAKE degenerate: mot dong, lap cung mot frame N lan.

The le chi cho nop 3 lan moi goi va lay diem lan CUOI, nen "nop thu xem sao"
la khong duoc phep - moi dau hieu suy giam phai noi ra TRUOC khi nop.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import generate_submission as gs

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [v] {name}")
    else:
        _failed += 1
        print(f"  [X] {name}" + (f"  - {detail}" if detail else ""))


def rows_kis(n: int, videos: int = 20) -> list:
    return [{"video_id": f"L21_V{i % videos:03d}", "frame_id": str(1000 + i),
             "score": 1.0 - i / 1000} for i in range(n)]


def rows_qa(n: int, answer: str | None = None) -> list:
    """answer=None -> moi video mot dap an (ca lanh manh).

    Q&A lanh manh KHONG THE cho ra dung mot answer tren nhieu video khac nhau:
    `answer_qa` hoi tung nhom video roi moi lan truyen dap an pho bien nhat cho
    nhom khong hoi duoc. Ca "100 dong mot answer" chinh la dau van tay cua loi
    Dot 1 (chi 5 nhom duoc hoi, 95% dong an dap an cua video KHAC), nen audit
    phai bat - fixture moi la thu phai doi.
    """
    out = []
    for i, r in enumerate(rows_kis(n)):
        out.append(dict(r, answer=answer if answer is not None
                        else f"dap an {i % 7}"))
    return out


class FakeBackend:
    name = "fake"

    def __init__(self, kis=None, qa=None, trake=None, raise_on=None):
        self._kis, self._qa, self._trake = kis, qa, trake
        self.raise_on = raise_on or set()

    def kis(self, query, top_k):
        if "kis" in self.raise_on:
            raise ConnectionError("502 Bad Gateway")
        return (self._kis if self._kis is not None else rows_kis(100))[:top_k]

    def qa(self, description, question, top_k):
        if "qa" in self.raise_on:
            raise ConnectionError("502 Bad Gateway")
        return (self._qa if self._qa is not None else rows_qa(100))[:top_k]

    def trake(self, event_queries, top_k):
        if "trake" in self.raise_on:
            raise ConnectionError("502 Bad Gateway")
        if self._trake is not None:
            return self._trake
        return [{"video_id": "L21_V001",
                 "frame_ids": [str(1000 + i) for i in range(len(event_queries))],
                 "score": 0.9}]


def make_queries(d: Path) -> None:
    (d / "q-1-kis.txt").write_text("nguoi dan ong ao trang", encoding="utf-8")
    (d / "q-2-qa.txt").write_text("Ten con deo la gi?", encoding="utf-8")
    (d / "q-3-trake.txt").write_text(
        "canh mo dau\nE1 nguoi chay toi\nE2 nguoi ve dich", encoding="utf-8")
    (d / "ghi-chu.txt").write_text("khong phai de", encoding="utf-8")


def run(backend, retries: int = 0):
    """Chay runner trong thu muc tam, tra `(exit_code, ten_file_trong_zip, report)`."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        qdir = tmp / "queries"
        qdir.mkdir()
        make_queries(qdir)

        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            code = gs.generate_submission(str(qdir), backend, retries,
                                          "submission.zip", "report.json")
            names = []
            if os.path.exists("submission.zip"):
                with zipfile.ZipFile("submission.zip") as zf:
                    names = sorted(zf.namelist())
            report = json.loads(Path("report.json").read_text(encoding="utf-8"))
            return code, names, report
        finally:
            os.chdir(cwd)


def test_duong_sach() -> None:
    print("\n[1] moi thu binh thuong -> exit 0")
    code, names, report = run(FakeBackend())
    check("exit code 0", code == 0, str(code))
    check("zip co du 3 file csv", len(names) == 3, str(names))
    check("bo qua file khong dung duoi ten", "submission/ghi-chu.csv" not in names)
    check("khong truy van nao bi danh dau suy giam",
          all(not r["problems"] for r in report), str(report))


def test_qa_toan_khong_ro() -> None:
    print("\n[2] Q&A tra 'khong ro' 100/100 -> phai bao dong")
    code, names, report = run(FakeBackend(qa=rows_qa(100, "khong ro")))
    qa = next(r for r in report if r["type"] == "qa")
    check("exit code khac 0", code == 1, str(code))
    check("van ghi du file (khong vut bo)", len(names) == 3, str(names))
    check("bat duoc 100% 'khong ro'",
          any("khong ro" in p for p in qa["problems"]), str(qa["problems"]))


def test_trake_degenerate() -> None:
    print("\n[3] TRAKE rong -> fallback KIS, va phai bao la degenerate")
    code, names, report = run(FakeBackend(trake=[], kis=rows_kis(1)))
    tr = next(r for r in report if r["type"] == "trake")
    check("exit code khac 0", code == 1, str(code))
    check("co ghi nhan da fallback",
          any("fallback" in p for p in tr["problems"]), str(tr["problems"]))
    check("co ghi nhan degenerate",
          any("degenerate" in p for p in tr["problems"]), str(tr["problems"]))


def test_thieu_dong() -> None:
    print("\n[4] tra ve it hon 100 dong -> phai bao, khong lam ngo")
    code, _, report = run(FakeBackend(kis=rows_kis(37)))
    kis = next(r for r in report if r["type"] == "kis")
    check("exit code khac 0", code == 1, str(code))
    check("bat duoc thieu dong",
          any("37/100" in p for p in kis["problems"]), str(kis["problems"]))


def test_mot_truy_van_hong() -> None:
    print("\n[5] mot truy van hong -> KHONG duoc am tham bo file")
    code, names, report = run(FakeBackend(raise_on={"qa"}), retries=0)
    qa = next(r for r in report if r["type"] == "qa")
    check("exit code khac 0", code == 1, str(code))
    check("cac file khac van duoc ghi", len(names) == 2, str(names))
    check("truy van hong duoc ghi ro trong bao cao",
          qa["error"] and "ConnectionError" in qa["error"], str(qa["error"]))
    check("bao cao noi ro la HONG",
          any("HONG" in p for p in qa["problems"]), str(qa["problems"]))


def test_parse_events() -> None:
    print("\n[6] TRAKE: moi dong khong rong la mot su kien")
    ev = gs.parse_events("canh mo dau\nE1 nguoi chay\nE2: ve dich\n\n")
    check("dem du 3 su kien (ke ca dong khong co nhan E)",
          len(ev) == 3, str(ev))
    check("da bo tien to nhan", ev[1] == "nguoi chay" and ev[2] == "ve dich",
          str(ev))
    check("giu nguyen dong mo dau", ev[0] == "canh mo dau", str(ev))


def main() -> int:
    test_duong_sach()
    test_qa_toan_khong_ro()
    test_trake_degenerate()
    test_thieu_dong()
    test_mot_truy_van_hong()
    test_parse_events()
    print(f"\n{'=' * 50}\n{_passed} pass . {_failed} fail")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
