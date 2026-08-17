from app.services.embedding import HashingEmbedding, normalize_text, tokenize


def test_normalize_vietnamese_accents() -> None:
    assert normalize_text("Tìm xe máy màu đỏ ở Cổng chính!") == "tim xe may mau do o cong chinh"


def test_embedding_is_deterministic() -> None:
    embedder = HashingEmbedding(dim=32)
    first = embedder.embed("nguoi ao den di vao cong")
    second = embedder.embed("nguoi ao den di vao cong")
    assert first == second
    assert len(first) == 32


def test_tokenizer_expands_domain_terms() -> None:
    tokens = tokenize("Tìm người áo đen đi vào cổng chính")
    assert "black_shirt" in tokens
    assert "enter" in tokens
    assert "main_gate" in tokens

