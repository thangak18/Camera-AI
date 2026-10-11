"""Test lop chiu loi cua wrapper Gemini - chay offline, khong goi mang.

    python tests/test_client.py

Bo test nay chot lai dung nhung gi da hong that ngay 28/08, khi Google tra 503
tren MOI model:

  * Mot lenh call_gemini di het danh sach model, moi model mot timeout 25s
    -> 75s cho MOT lenh. Va no lap lai o CA BA cho goi doc lap (translator,
    Stage B, Q&A), nen mot truy van KIS mat ~2 phut roi tunnel dut giua chung.
  * 503 bi xu ly nhu loi la: nhay sang model khac thay vi lui lai thu lai
    cung model - dung cach de tich du (so model x timeout) roi van tra rong.
  * probe_models() gach ten model vinh vien du 503 chi la tam thoi.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import sys
import time

from src.common.config import (GEMINI_BREAKER_FAILS, GEMINI_MAX_INFLIGHT,
                               GEMINI_MODELS, GEMINI_RETRY_ON_503,
                               MODEL_PROBE_TTL)
from src.vlm import client

_passed = 0
_failed = 0

OVERLOAD = "503 UNAVAILABLE. This model is currently experiencing high demand"
DENIED = ("403 PERMISSION_DENIED. {'error': {'code': 403, 'message': "
          "'Your project has been denied access. Please contact support.'")


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [v] {name}")
    else:
        _failed += 1
        print(f"  [X] {name}" + (f"  - {detail}" if detail else ""))


def reset() -> None:
    """Ve trang thai sach. Breaker la bien module dung chung ca tien trinh."""
    client._breaker_fails = 0
    client._breaker_until = 0.0
    client._breaker_backoff = 0.0
    client._breaker_probing = False
    client._inflight = 0
    client._active_models = None
    client._active_models_at = 0.0


class Recorder:
    """Thay `_one_call`. Ghi lai model nao bi goi, gia lap ket qua mong muon.

    `behaviour(model, so_lan_da_goi)` tra ve chuoi ket qua, mot Exception de
    nem, hoac "__hang__" de gia lap treo den het ngan sach.
    """

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls: list[str] = []

    def __call__(self, model, contents, timeout_s):
        self.calls.append(model)
        out = self.behaviour(model, len(self.calls))
        if isinstance(out, BaseException):
            raise out
        if out == "__hang__":
            time.sleep(timeout_s)
            raise cf.TimeoutError()
        return out


class patched:
    """Thay `_one_call` trong pham vi `with`, luon tra lai ban that."""

    def __init__(self, behaviour):
        self.rec = Recorder(behaviour)

    def __enter__(self) -> Recorder:
        self._real = client._one_call
        client._one_call = self.rec
        return self.rec

    def __exit__(self, *exc) -> None:
        client._one_call = self._real


def test_deadline() -> None:
    print("\n[1] deadline chan viec nhan timeout theo so model")
    reset()
    with patched(lambda m, n: "__hang__") as rec:
        t0 = time.monotonic()
        out = client.call_gemini("hoi gi do", timeout_s=1.0)
        dt = time.monotonic() - t0
    check("tra ve chuoi rong", out == "", repr(out))
    # Ban cu: (so model) x 1s. Nay timeout_s la tran cho CA lenh goi.
    check(f"khong vuot tran ca lenh goi ({dt:.2f}s <= 1.5s)", dt <= 1.5,
          f"{dt:.2f}s, da goi {len(rec.calls)} model")
    check("het ngan sach thi dung, khong thu model tiep",
          len(rec.calls) == 1, str(rec.calls))


def test_503_thu_lai_cung_model() -> None:
    print("\n[2] 503 lui lai thu lai CUNG model truoc khi doi model")
    reset()
    # Gioi han 2 model de test khong ton thoi gian cho backoff cua ca 5.
    client._active_models = list(GEMINI_MODELS[:2])
    client._active_models_at = time.monotonic()
    with patched(lambda m, n: Exception(OVERLOAD)) as rec:
        client.call_gemini("hoi gi do", timeout_s=30.0,
                           deadline=time.monotonic() + 30.0)
    n_dau = rec.calls.count(GEMINI_MODELS[0])
    check(f"model dau duoc thu lai du {GEMINI_RETRY_ON_503} lan",
          n_dau == GEMINI_RETRY_ON_503, f"{n_dau} lan")
    check("het luot moi fallback sang model khac",
          len(set(rec.calls)) == 2, str(set(rec.calls)))
    reset()


def test_403_doi_key() -> None:
    print("\n[12] 403 doi KEY, khong nhay model")
    reset()
    # Ngay 04/09: key so 1 bi cam o cap project ("denied access"). 403 roi vao
    # nhanh Exception chung -> break -> thu model ke tiep bang DUNG cai key da
    # chet, du 5 model la het deadline ma chua tung cham toi key con song.
    client._keys = ["key-mot", "key-hai"]
    client._key_idx = 0
    client._dead_keys.clear()
    client._cooldown_until.clear()
    client._active_models = list(GEMINI_MODELS[:2])
    client._active_models_at = time.monotonic()

    def behaviour(model, n):
        return Exception(DENIED) if client._key_idx == 0 else "xong"

    with patched(behaviour) as rec:
        out = client.call_gemini("hoi gi do", timeout_s=30.0,
                                 deadline=time.monotonic() + 30.0)

    check("doi key roi tra ve ket qua that", out == "xong", repr(out))
    check("key bi cam bi danh dau chet", 0 in client._dead_keys,
          str(client._dead_keys))
    check("403 khong lam nhay sang model khac",
          len(set(rec.calls)) == 1, str(rec.calls))

    client._keys = []
    client._key_idx = 0
    client._dead_keys.clear()
    client._cooldown_until.clear()
    reset()


def test_403_moi_key_deu_cam() -> None:
    print("\n[13] moi key deu 403 -> dung han, khong quet het danh sach model")
    reset()
    client._keys = ["key-mot", "key-hai"]
    client._key_idx = 0
    client._dead_keys.clear()
    client._cooldown_until.clear()
    client._active_models = list(GEMINI_MODELS[:5])
    client._active_models_at = time.monotonic()

    with patched(lambda m, n: Exception(DENIED)) as rec:
        out = client.call_gemini("hoi gi do", timeout_s=30.0,
                                 deadline=time.monotonic() + 30.0)

    check("tra ve chuoi rong", out == "", repr(out))
    check("dung sau khi thu het key, khong quet 5 model",
          len(rec.calls) == 2, str(rec.calls))

    client._keys = []
    client._key_idx = 0
    client._dead_keys.clear()
    client._cooldown_until.clear()
    reset()


def test_probe_403_doi_key() -> None:
    print("\n[14] probe gap loi cap key thi doi key, khong gach het model")
    reset()
    client._keys = ["key-mot", "key-hai"]
    client._key_idx = 0
    client._dead_keys.clear()
    client._cooldown_until.clear()

    def behaviour(model, n):
        return Exception(DENIED) if client._key_idx == 0 else "pong"

    with patched(behaviour):
        alive = client.probe_models(list(GEMINI_MODELS[:2]))

    check("do ra du model sau khi doi key",
          alive == list(GEMINI_MODELS[:2]), str(alive))
    check("key bi cam bi danh dau chet", 0 in client._dead_keys,
          str(client._dead_keys))

    client._keys = []
    client._key_idx = 0
    client._dead_keys.clear()
    client._cooldown_until.clear()
    reset()


def test_probe_403_moi_key_deu_chet() -> None:
    print("\n[15] probe: moi key deu hong -> dung, khong ket vong lap")
    reset()
    client._keys = ["key-mot", "key-hai"]
    client._key_idx = 0
    client._dead_keys.clear()
    client._cooldown_until.clear()

    with patched(lambda m, n: Exception(DENIED)) as rec:
        alive = client.probe_models(list(GEMINI_MODELS[:2]))

    check("khong do ra model nao", alive == [], str(alive))
    # Model dau an het 2 key roi probe dung han: khong dot PROBE_TIMEOUT cho
    # cac model con lai. Neu vong `while` khong dung, test nay treo han.
    check("dung ngay khi het key, khong quet not model con lai",
          len(rec.calls) == 2, str(len(rec.calls)))

    client._keys = []
    client._key_idx = 0
    client._dead_keys.clear()
    client._cooldown_until.clear()
    reset()


def test_breaker_mo() -> None:
    print("\n[3] breaker mo sau N lan hong, roi tra rong tuc thi")
    reset()
    with patched(lambda m, n: Exception(OVERLOAD)) as rec:
        for _ in range(GEMINI_BREAKER_FAILS):
            client.call_gemini("x", timeout_s=0.5)
        st = client.breaker_state()
        check("breaker da MO", st["open"], str(st))

        n_truoc = len(rec.calls)
        t0 = time.monotonic()
        out = client.call_gemini("x", timeout_s=0.5)
        dt = time.monotonic() - t0
        check("lenh sau tra rong ngay lap tuc", out == "" and dt < 0.05,
              f"{out!r} sau {dt:.3f}s")
        check("khong ban them request nao", len(rec.calls) == n_truoc,
              f"{len(rec.calls) - n_truoc} request thua")
    reset()


def test_breaker_dong_lai() -> None:
    print("\n[4] breaker he mo mot lenh tham do, thanh cong thi dong lai")
    reset()
    khoe = {"ok": False}
    behaviour = lambda m, n: "ngon" if khoe["ok"] else Exception(OVERLOAD)
    with patched(behaviour):
        for _ in range(GEMINI_BREAKER_FAILS):
            client.call_gemini("x", timeout_s=0.5)
        check("dang mo", client.breaker_state()["open"])

        client._breaker_until = time.monotonic() - 0.01   # gia lap het han
        khoe["ok"] = True
        out = client.call_gemini("x", timeout_s=0.5)
        check("lenh tham do di duoc va co ket qua", out == "ngon", repr(out))
        check("breaker da DONG", not client.breaker_state()["open"],
              str(client.breaker_state()))
    reset()


def test_probe_rong() -> None:
    print("\n[5] probe khong dò duoc model nao != gach ten vinh vien")
    reset()
    with patched(lambda m, n: Exception(OVERLOAD)):
        alive = client.probe_models()
    check("probe tra ve rong", alive == [], str(alive))
    check("_active_models KHONG bi ghi de bang []",
          client._active_models is None, str(client._active_models))
    check("van con danh sach de thu khi co request that",
          client._models_to_try() == GEMINI_MODELS)


def test_probe_het_han() -> None:
    print("\n[6] ket qua probe het han thi thu lai ca danh sach")
    reset()
    client._active_models = [GEMINI_MODELS[0]]
    client._active_models_at = time.monotonic()
    check("con han: chi dung model da dò duoc",
          client._models_to_try() == [GEMINI_MODELS[0]])
    client._active_models_at = time.monotonic() - MODEL_PROBE_TTL - 1
    check("het han: quay lai ca danh sach",
          client._models_to_try() == GEMINI_MODELS, str(client._models_to_try()))
    reset()


def test_inflight() -> None:
    print("\n[7] cham tran inflight thi hong ngay, khong xep hang sau xac")
    reset()
    client._inflight = GEMINI_MAX_INFLIGHT
    raised = False
    try:
        client._acquire_inflight()
    except client.GeminiSaturated:
        raised = True
    check("_acquire_inflight nem GeminiSaturated khi day", raised)

    t0 = time.monotonic()
    out = client.call_gemini("x", timeout_s=5.0)
    dt = time.monotonic() - t0
    check("call_gemini tra rong ngay thay vi cho", out == "" and dt < 0.5,
          f"{out!r} sau {dt:.3f}s")
    reset()


def test_loi_code_khong_ket_co_tham_do() -> None:
    print("\n[9] loi code khong lam ket co tham do cua breaker")
    reset()
    client._breaker_until = time.monotonic() - 0.01   # dang o trang thai he mo
    client._breaker_backoff = 30.0
    with patched(lambda m, n: TypeError("bug trong code cua minh")):
        no = False
        try:
            client.call_gemini("x", timeout_s=0.5)
        except TypeError:
            no = True
    check("loi code van no ra ngoai, khong bi nuot", no)
    check("co tham do da duoc tra lai", client._breaker_probing is False,
          str(client._breaker_probing))
    # Neu co con ket True thi moi lan thu tiep theo deu bi tu choi im lang.
    client._breaker_until = time.monotonic() - 0.01
    with patched(lambda m, n: "ngon"):
        out = client.call_gemini("x", timeout_s=0.5)
    check("lan tham do sau van di duoc", out == "ngon", repr(out))
    reset()


def test_nhip_tu_dieu_chinh() -> None:
    """429 phai NOI nhip, thanh cong phai THU lai — khong doan hang muc cung.

    Do 28/08 voi 3 key that: nhip co dinh 0.7s (86 req/phut) lam ca ba key can
    quota trong ~20 giay, cau Q&A 42 nhom video mat trang 100 dong. Han muc
    that cua tung key thi khong biet truoc, nen phai tu do.
    """
    print("\n[11] nhip tu dieu chinh theo 429")
    reset()
    from src.common.config import GEMINI_MIN_INTERVAL_S, GEMINI_MAX_INTERVAL_S
    client._interval = GEMINI_MIN_INTERVAL_S
    goc = client._interval

    with patched(lambda m, n: Exception("429 RESOURCE_EXHAUSTED")):
        client.call_gemini("x", timeout_s=0.5)
    check("gap 429 -> nhip noi ra", client._interval > goc,
          f"{goc} -> {client._interval}")

    truoc = client._interval
    with patched(lambda m, n: "ngon"):
        client.call_gemini("x", timeout_s=5.0)
    check("thanh cong -> nhip thu lai", client._interval < truoc,
          f"{truoc} -> {client._interval}")

    client._interval = GEMINI_MAX_INTERVAL_S
    with patched(lambda m, n: Exception("429 RESOURCE_EXHAUSTED")):
        client.call_gemini("x", timeout_s=0.5)
    check("khong noi qua tran", client._interval <= GEMINI_MAX_INTERVAL_S,
          str(client._interval))

    client._interval = GEMINI_MIN_INTERVAL_S
    for _ in range(5):
        client._speed_up()
    check("khong thu xuong duoi san",
          client._interval >= GEMINI_MIN_INTERVAL_S, str(client._interval))
    check("rate_state bao dung", client.rate_state()["interval_s"] > 0,
          str(client.rate_state()))
    client._interval = goc
    reset()


def test_max_side_doc_luc_goi() -> None:
    print("\n[8] max_side doc bien module LUC GOI (de diag_qa quet duoc)")
    reset()
    seen = {}
    goc_part, goc_side = client._to_part, client.MAX_IMAGE_SIDE

    def fake_to_part(img, max_side):
        seen["side"] = max_side
        return "part"

    client._to_part = fake_to_part
    try:
        with patched(lambda m, n: "ok"):
            client.MAX_IMAGE_SIDE = 1234       # dung cach diag_qa.py gan de
            client.call_gemini("x", images=[object()], timeout_s=5.0)
            check("gan de bien module co tac dung", seen.get("side") == 1234,
                  str(seen))
            client.call_gemini("x", images=[object()], timeout_s=5.0,
                               max_side=321)
            check("truyen tuong minh van thang", seen.get("side") == 321,
                  str(seen))
    finally:
        client._to_part, client.MAX_IMAGE_SIDE = goc_part, goc_side
        reset()


def main() -> int:
    logging.basicConfig(level=logging.CRITICAL)

    test_deadline()
    test_503_thu_lai_cung_model()
    test_403_doi_key()
    test_403_moi_key_deu_cam()
    test_probe_403_doi_key()
    test_probe_403_moi_key_deu_chet()
    test_breaker_mo()
    test_breaker_dong_lai()
    test_probe_rong()
    test_probe_het_han()
    test_inflight()
    test_loi_code_khong_ket_co_tham_do()
    test_nhip_tu_dieu_chinh()
    test_max_side_doc_luc_goi()

    print(f"\n{'=' * 50}\n{_passed} pass . {_failed} fail")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
