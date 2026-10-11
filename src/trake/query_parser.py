"""Tách file đề TRAKE thành danh sách phân cảnh (event).

VÌ SAO CÓ FILE NÀY — luật cũ "mỗi dòng không rỗng là một event" sai trên đề Đợt 2:

    query-p2-8-trake.txt          query-p2-21-trake.txt
    ---------------------------   ---------------------------------------
    Video về một khu vườn ...     4 cảnh này xảy ra liên tiếp nhau.
    Đây là chuỗi liên tiếp các    Cảnh 1: Hai người phụ nữ dán niêm phong
    cảnh quay về 4 LOẠI trái cây. Cảnh 2: Các thùng mì tôm ...
    E1: Cảnh đầu tiên có sầu      Cảnh 3: Một người đàn ông nhấc thùng ...
    ... E2, E3, E4                Cảnh 4: Cảnh quay cận cảnh ...

Cả hai đề TỰ KHAI là 4 phân cảnh, nhưng đếm theo dòng ra 5 → CSV thừa một cột.
Thể lệ chấm theo đúng N cột; log Đợt 1 đã từng loại nguyên file vì lệch số cột
("query-p1-16-trake: Expected 4 frame IDs, got 3"). Thêm nữa `Cảnh <số>:` không
nằm trong regex bóc nhãn cũ (chỉ biết `E<số>`) nên chuỗi "Cảnh 1:" đi thẳng vào
câu truy vấn SigLIP — nhiễu thuần tuý.

ĐỢT 3 LÀM LỘ THÊM MỘT LỖ. Hai file dưới đây KHÔNG khai số phân cảnh, dòng dẫn
KHÔNG kết thúc bằng ":" và KHÔNG chứa cụm dẫn nhập nào — luật Đợt 2 đếm ra 5:

    queries-3/query-p2-21-trake.txt     queries-3/query-p2-34-trake.txt
    --------------------------------    -------------------------------
    Một món ăn làm từ tôm và các        Con lân màu đỏ đang biểu diễn
    loại gia vị.                        trên các cột trụ.
    E1: ... E2 ... E3 ... E4            (dòng trống)
                                        E1: ... E2 ... E3 ... E4

Bằng chứng dương rằng dòng đầu là LỜI DẪN chứ không phải phân cảnh: BTC đã tự
đánh số **liền mạch E1..E4**, và dòng đó chỉ nêu CHỦ ĐỀ cả video ("một món ăn
làm từ tôm", "con lân màu đỏ") — không có mốc thời điểm nào để đi tìm một frame.

QUY TẮC, theo thứ tự ưu tiên. Chỉ BỎ một dòng không nhãn khi có bằng chứng
dương rằng nó là lời dẫn; mọi trường hợp mơ hồ vẫn giữ nguyên hành vi cũ, để
không phá ca đã được BTC xác nhận ở Đợt 1 (queries-1/query-p1-16: dòng mở đầu
KHÔNG nhãn "Đoạn video BẮT ĐẦU BẰNG ảnh cận đầu một con lân trắng..." được BTC
tính là event thật, vì nó mô tả một khoảnh khắc cụ thể).

  1. Đề tự khai số phân cảnh ("4 cảnh", "4 loại trái cây") và con số đó khớp
     đúng số dòng CÓ NHÃN  -> các dòng không nhãn là lời dẫn, bỏ.
     Khớp với tổng số dòng -> giữ tất cả.
  2. Dòng không nhãn kết thúc bằng ":" hoặc chứa cụm dẫn nhập ("tìm các sự
     kiện sau", "gồm các khoảnh khắc", "xảy ra liên tiếp nhau", ...) -> lời dẫn.
  3. Nhãn đánh số LIỀN MẠCH 1..N (BTC đã tự đếm đủ) -> dòng không nhãn còn lại
     chỉ được tính là event khi nó CÓ MỐC THỜI ĐIỂM ("khoảnh khắc", "bắt đầu
     bằng", "đầu tiên", ...). Đây là chỗ tách Đợt 3 khỏi queries-1/p1-16:
     "Đoạn video bắt đầu bằng ảnh cận..." có mốc -> event; "Con lân màu đỏ
     đang biểu diễn trên các cột trụ." không có mốc -> lời dẫn.
  4. Nhãn thủng/trùng số (query-p1-18 ghi E1, E2, E2, E4) -> KHÔNG tin cách
     đánh số nữa, dòng không nhãn quay về là event y như luật cũ.

So khớp cụm từ ở đây BỎ DẤU trước khi so ("khoảnh khắc" == "khoanh khac"): đề
thật luôn có dấu, nhưng bỏ dấu thì một file gõ thiếu dấu cũng không làm lệch số
cột — mà lệch số cột là mất trắng cả file.

Nhãn nhận diện được: `E1`, `E1:`, `E1 ` (Đợt 1 không có dấu hai chấm),
`Cảnh 1:`, `Scene 1:`, `Sự kiện 1:`, `Khoảnh khắc 1:`, và số trần `1.` / `1)`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Nhãn chữ: dấu ngăn cách TUỲ CHỌN — đề Đợt 1 viết "E1 Khoảnh khắc..." không
# có dấu hai chấm, đề thử nghiệm viết "E1: ...".
_LABELLED_RE = re.compile(
    r"^(?:E|C[ảa]nh|Scene|Shot|S[ựu]\s*ki[ệe]n|Event|Kho[ảa]nh\s*kh[ắa]c)"
    r"\s*(?P<num>\d{1,2})\s*[:.)\]\-–—]?\s+(?P<body>\S.*)$",
    re.IGNORECASE,
)

# Số trần: dấu ngăn cách BẮT BUỘC, nếu không một câu mở đầu bằng số ("4 cảnh
# này xảy ra liên tiếp nhau.") sẽ bị hiểu nhầm thành nhãn.
_NUMBERED_RE = re.compile(
    r"^(?P<num>\d{1,2})\s*[:.)\]\-–—]\s+(?P<body>\S.*)$")

# "4 cảnh", "4 loại trái cây", "3 sự kiện", "4 khoảnh khắc"...
_DECLARED_COUNT_RE = re.compile(
    r"(\d{1,2})\s*(?:c[ảa]nh|lo[ạa]i|s[ựu]\s*ki[ệe]n|kho[ảa]nh\s*kh[ắa]c"
    r"|[đd]o[ạa]n|scenes?|shots?|events?|moments?)",
    re.IGNORECASE,
)


def _fold(text: str) -> str:
    """Bỏ dấu + hạ chữ thường, để so khớp cụm từ không phụ thuộc dấu tiếng Việt."""
    # NFD tách dấu thanh/dấu mũ thành ký tự tổ hợp rồi lọc đi, nhưng KHÔNG tách
    # được "đ" -> "d" (đ không phải d + dấu tổ hợp), nên phải thay tay.
    stripped = "".join(c for c in unicodedata.normalize("NFD", text.lower())
                       if not unicodedata.combining(c))
    return stripped.replace("đ", "d")


# Cụm chỉ có ở lời dẫn, không bao giờ ở một câu mô tả khoảnh khắc.
_HEADER_PHRASES = tuple(_fold(p) for p in (
    "tìm các", "gồm các", "bao gồm", "như sau", "sau đây",
    "các sự kiện", "các khoảnh khắc", "các cảnh", "các phân cảnh",
    "chuỗi liên tiếp", "liên tiếp nhau", "xảy ra liên tiếp",
    "theo thứ tự", "thứ tự thời gian", "lần lượt",
    "following", "in order", "the following",
))

# Mốc neo vào MỘT thời điểm cụ thể. Dòng không nhãn có mốc này thì đang mô tả
# một frame đi tìm được (=> event); không có thì nó chỉ nêu chủ đề cả video.
#
# CỐ Ý KHÔNG có "lúc" trần: bỏ dấu xong "lục" (màu lục) cũng thành "luc". Các
# mốc còn lại đã phủ hết ca thật. "khoảng khắc" là lỗi chính tả có thật trong
# đề (queries-0/query-p1-4 E4).
_MOMENT_MARKERS = tuple(_fold(p) for p in (
    "khoảnh khắc", "khoảng khắc", "thời điểm", "thời khắc",
    "bắt đầu", "mở đầu", "kết thúc",
    "đầu tiên", "cuối cùng", "ngay khi", "ngay lúc", "ngay sau",
    "cảnh đầu", "frame",
    "the moment", "first moment", "first scene", "last scene",
    "begins with", "starts with", "ends with",
))


@dataclass(frozen=True)
class TrakeQuery:
    """Kết quả tách một file đề TRAKE."""

    events: list[str]
    """Mô tả từng phân cảnh, đúng thứ tự thời gian, đã bóc tiền tố nhãn."""

    context: list[str]
    """Các dòng bị xếp là lời dẫn. Giữ lại để in ra cho người vận hành soi."""


def _strip_label(line: str) -> tuple[int, str] | None:
    """Trả `(số nhãn, mô tả)` nếu `line` có nhãn phân cảnh, ngược lại None."""
    for pattern in (_LABELLED_RE, _NUMBERED_RE):
        m = pattern.match(line)
        if m:
            return int(m.group("num")), m.group("body").strip()
    return None


def _looks_like_header(line: str) -> bool:
    """Dòng không nhãn này có DẤU HIỆU RÕ của lời dẫn không?"""
    if line.rstrip().endswith(":"):
        return True
    folded = _fold(line)
    return any(phrase in folded for phrase in _HEADER_PHRASES)


def _has_moment_marker(line: str) -> bool:
    """Dòng này có neo vào một thời điểm cụ thể không?"""
    folded = _fold(line)
    return any(marker in folded for marker in _MOMENT_MARKERS)


def _declared_count(lines: list[str]) -> int | None:
    """Số phân cảnh mà đề tự khai, lấy con số ĐẦU TIÊN gặp."""
    for line in lines:
        m = _DECLARED_COUNT_RE.search(line)
        if m:
            return int(m.group(1))
    return None


def _numbering_is_complete(numbers: list[int]) -> bool:
    """Nhãn có chạy đủ 1..N không? Thủng hoặc trùng số là BTC đánh nhầm."""
    return sorted(numbers) == list(range(1, len(numbers) + 1))


def parse_trake_query(text: str) -> TrakeQuery:
    """Tách `text` thành các phân cảnh. Xem docstring module cho quy tắc."""
    rows: list[tuple[str, tuple[int, str] | None]] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if line:
            rows.append((line, _strip_label(line)))

    if not rows:
        return TrakeQuery(events=[], context=[])

    labelled = [label for _, label in rows if label is not None]
    unlabelled = [line for line, label in rows if label is None]

    # Không có nhãn nào: mọi dòng là event (luật cũ). Chỉ gạt dòng kết thúc
    # bằng ":" ở đầu file — một câu như vậy không bao giờ là mô tả khoảnh khắc.
    if not labelled:
        if len(rows) > 1 and rows[0][0].rstrip().endswith(":"):
            return TrakeQuery(events=[line for line, _ in rows[1:]],
                              context=[rows[0][0]])
        return TrakeQuery(events=[line for line, _ in rows], context=[])

    if not unlabelled:
        return TrakeQuery(events=[body for _, body in labelled], context=[])

    # 1. Đề tự khai số phân cảnh -> tin con số đó trước mọi suy đoán khác.
    declared = _declared_count(unlabelled)
    if declared == len(labelled):
        return TrakeQuery(events=[body for _, body in labelled],
                          context=unlabelled)
    if declared == len(rows):
        return TrakeQuery(events=[label[1] if label is not None else line
                                  for line, label in rows], context=[])

    # 3. Nhãn chạy đủ 1..N -> BTC đã tự đếm; dòng không nhãn phải TỰ CHỨNG MINH
    #    là một khoảnh khắc thì mới được thêm cột.
    # 4. Nhãn thủng/trùng -> không tin cách đánh số, giữ nguyên luật cũ.
    trust_numbering = _numbering_is_complete([num for num, _ in labelled])

    events: list[str] = []
    context: list[str] = []
    for line, label in rows:
        if label is not None:
            events.append(label[1])
        # 2. Dấu hiệu lời dẫn xét TRƯỚC: query-p1-18 mở đầu bằng "...gồm các
        #    khoảnh khắc sơ chế:" — vừa có mốc vừa là lời dẫn, và nó là lời dẫn.
        elif _looks_like_header(line):
            context.append(line)
        elif trust_numbering and not _has_moment_marker(line):
            context.append(line)
        else:
            events.append(line)
    return TrakeQuery(events=events, context=context)


def count_events(text: str) -> int:
    """Số cột frame mà file CSV TRAKE tương ứng phải có."""
    return len(parse_trake_query(text).events)
