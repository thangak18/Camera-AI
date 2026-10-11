"""Tien kiem truoc khi sinh bai nop tren MAY LOCAL.

    python scripts/preflight.py

Chay cai nay TRUOC khi de ve, khong phai luc dang com. Moi muc duoi day tung
hong that mot lan, va deu hong AM THAM - pipeline van chay, van xuat du file,
diem van 0:

  * symlink/thu muc du lieu rong  -> reranker "coverage 0%", VLM 100% khong ro
  * Qdrant tro nham collection    -> 96% frame_id khong phai keyframe cua BTC
  * payload thieu `n`             -> Stage B va VLM cung chet im lang
  * khong co GEMINI_API_KEY       -> khong dich duoc query, Final Score 0.000

Exit 0 = du dieu kien chay. Khac 0 = con thieu, doc dong [X].
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_fail = 0
_warn = 0


def ok(name: str, detail: str = "") -> None:
    print(f"  [v] {name}" + (f"  {detail}" if detail else ""))


def bad(name: str, detail: str = "") -> None:
    global _fail
    _fail += 1
    print(f"  [X] {name}" + (f"  {detail}" if detail else ""))


def warn(name: str, detail: str = "") -> None:
    global _warn
    _warn += 1
    print(f"  [!] {name}" + (f"  {detail}" if detail else ""))


def check_data() -> None:
    print("\n[1] Du lieu tren dia")
    from src.common.config import (KEYFRAMES_DIR, MAP_KEYFRAMES_DIR,
                                   SIGLIP_FEATURES_DIR)

    for label, d, pattern in (
        ("keyframes", KEYFRAMES_DIR, "*/*.jpg"),
        ("map_keyframes", MAP_KEYFRAMES_DIR, "*.csv"),
        ("siglip_features", SIGLIP_FEATURES_DIR, "*.npy"),
    ):
        if not d.exists():
            bad(f"{label}: khong ton tai", str(d))
            continue
        # Dem thu muc con truoc (nhanh), chi glob sau neu can - keyframes co
        # ~177k file, glob het rat cham.
        n_dir = sum(1 for _ in d.iterdir())
        first = next(d.glob(pattern), None)
        if first is None:
            bad(f"{label}: co thu muc nhung KHONG doc duoc file nao",
                f"{d} (pattern {pattern})")
        else:
            ok(f"{label}: {n_dir} muc", f"vd {first.name}")


def check_qdrant() -> None:
    print("\n[2] Qdrant")
    from src.common.config import (QDRANT_API_KEY, QDRANT_COLLECTION_NAME,
                                   QDRANT_TIMEOUT, QDRANT_URL)
    try:
        from qdrant_client import QdrantClient
        c = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY,
                         timeout=QDRANT_TIMEOUT)
        names = [x.name for x in c.get_collections().collections]
    except Exception as exc:
        bad("khong ket noi duoc Qdrant", f"{type(exc).__name__}: {exc}")
        return

    ok("ket noi duoc", QDRANT_URL.split("//")[-1][:40])
    if QDRANT_COLLECTION_NAME not in names:
        bad(f"thieu collection '{QDRANT_COLLECTION_NAME}'", f"co: {names}")
        return

    info = c.get_collection(QDRANT_COLLECTION_NAME)
    n = info.points_count or 0
    (ok if n > 100_000 else warn)(
        f"collection '{QDRANT_COLLECTION_NAME}': {n:,} diem",
        "" if n > 100_000 else "it hon du kien (~177k keyframe Batch 1)")

    # Payload phai co `n`: thieu no thi Stage B va VLM khong mo duoc anh, ca
    # hai deu fail mem nen khong ai biet cho toi luc xem diem.
    pts, _ = c.scroll(QDRANT_COLLECTION_NAME, limit=50, with_payload=True)
    if not pts:
        bad("scroll khong ra diem nao")
        return
    missing = [p for p in pts if (p.payload or {}).get("n") is None]
    if missing:
        bad(f"payload thieu 'n' o {len(missing)}/{len(pts)} diem mau",
            "Stage B va VLM se chet im lang")
    else:
        ok("payload co 'n'", f"mau {len(pts)} diem")

    sample = pts[0].payload or {}
    for key in ("video_id", "frame_id", "pts_time"):
        if sample.get(key) is None:
            warn(f"payload thieu '{key}'", str(sample)[:80])


def check_gemini() -> None:
    print("\n[3] Gemini")
    from src.common.config import GEMINI_KEY_ENV_NAMES
    have = [k for k in GEMINI_KEY_ENV_NAMES if os.getenv(k)]
    if not have:
        bad("khong co key nao", f"dat mot trong: {GEMINI_KEY_ENV_NAMES[:3]}")
        return
    ok(f"{len(have)} key trong env", ", ".join(have))
    if len(have) == 1:
        warn("chi 1 key", "profile offline hoi HET nhom video, nen co them key")

    from src.vlm.client import probe_models
    t0 = time.time()
    alive = probe_models()
    dt = time.time() - t0
    if alive:
        ok(f"{len(alive)} model song ({dt:.1f}s)", alive[0])
    else:
        bad(f"khong model nao tra loi ({dt:.1f}s)",
            "Gemini dang sap hoac key sai - Dang 2 se ra 'khong ro'")


def check_torch() -> None:
    print("\n[4] SigLIP / torch")
    try:
        from src.common.config import resolve_device
        dev = resolve_device()
    except Exception as exc:
        bad("torch khong dung duoc", f"{type(exc).__name__}: {exc}")
        return
    (ok if dev == "cuda" else warn)(
        f"device = {dev}",
        "" if dev == "cuda" else "CPU cham hon nhung van chay duoc")
    try:
        from src.reranking import get_reranker
        t0 = time.time()
        get_reranker().encode_text("warmup")
        ok(f"encode_text chay duoc ({time.time() - t0:.1f}s)")
    except Exception as exc:
        bad("khong nap duoc SigLIP", f"{type(exc).__name__}: {exc}")


def check_budget() -> None:
    print("\n[5] Cau hinh ngan sach")
    from src.common.config import (PROFILE, QA_DEADLINE, QA_N_VLM_CALLS,
                                   RERANK_STAGE_B, STAGE_B_DEADLINE,
                                   TRANSLATE_DEADLINE, VLM_CALL_TIMEOUT)
    # Sinh bai nop PHAI o profile offline. Chay nham profile live thi Q&A chi
    # duoc 8s va hoi 5 nhom video - file van ra du 100 dong, van trong nhu binh
    # thuong, chi la ~95% dong an dap an cua video khac. Dung loi Dot 1.
    (ok if PROFILE == "offline" else warn)(
        f"AIC_PROFILE = {PROFILE}",
        "" if PROFILE == "offline" else "sinh bai nop nen dung 'offline'")
    ok(f"call={VLM_CALL_TIMEOUT}s  translate={TRANSLATE_DEADLINE}s  "
       f"stageB={STAGE_B_DEADLINE}s  qa={QA_DEADLINE}s")
    ok(f"Stage B {'BAT' if RERANK_STAGE_B else 'TAT'}  |  "
       f"QA_N_VLM_CALLS={QA_N_VLM_CALLS} ({'hoi HET nhom' if QA_N_VLM_CALLS == 0 else 'gioi han'})")
    if VLM_CALL_TIMEOUT > QA_DEADLINE:
        bad("VLM_CALL_TIMEOUT > QA_DEADLINE",
            "model dau treo het deadline, 100 dong an fallback - loi Dot 1")


def main() -> int:
    print("=" * 64)
    print("TIEN KIEM - sinh bai nop tren may local")
    print("=" * 64)
    check_data()
    check_qdrant()
    check_torch()
    check_gemini()
    check_budget()

    print("\n" + "=" * 64)
    if _fail:
        print(f"{_fail} muc HONG, {_warn} canh bao - sua truoc khi de ve.")
    else:
        print(f"Du dieu kien chay. {_warn} canh bao.")
        print("\n  python scripts/generate_submission.py --query_dir <thu-muc-de> --local")
    print("=" * 64)
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
