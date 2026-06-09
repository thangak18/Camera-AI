from __future__ import annotations

from app.schemas.dto import SearchResult


def build_answer(query: str, results: list[SearchResult]) -> str:
    if not results:
        return (
            "Chua tim thay doan video phu hop. Hay thu mo ta cu the hon ve mau sac, "
            "doi tuong, vi tri hoac khoang thoi gian."
        )

    best = results[0]
    return (
        f"Tim thay {len(results)} ket qua lien quan cho truy van: '{query}'. "
        f"Ket qua manh nhat o camera {best.camera_name or best.camera_id}, "
        f"khoang {best.start_second:.0f}-{best.end_second:.0f}s, vi {best.reason.lower()}"
    )


def build_summary(query: str, results: list[SearchResult]) -> str:
    if not results:
        return "Chua co bang chung du de tong hop bao cao cho truy van nay."

    lines = [f"Bao cao ngan cho: {query}", ""]
    for index, result in enumerate(results, start=1):
        camera = result.camera_name or result.camera_id
        lines.append(
            f"{index}. {camera} | {result.start_second:.0f}-{result.end_second:.0f}s | "
            f"score {result.score:.2f} | {result.caption}"
        )
    return "\n".join(lines)

