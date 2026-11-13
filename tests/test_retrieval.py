from filinglens.models import StoredChunk
from filinglens.retrieval import BM25Retriever, tokenize


def test_tokenize_adds_chinese_bigrams():
    tokens = tokenize("收入增长 AI")

    assert "收入" in tokens
    assert "增长" in tokens
    assert "ai" in tokens


def test_bm25_prefers_relevant_chunk():
    chunks = [
        StoredChunk(
            id=1,
            report_id=1,
            chunk_index=0,
            page_number=1,
            text="公司收入增长主要来自海外市场扩张。",
        ),
        StoredChunk(
            id=2,
            report_id=1,
            chunk_index=1,
            page_number=2,
            text="公司建立了完善的信息安全制度。",
        ),
    ]

    results = BM25Retriever().search("收入增长来自哪里", chunks, top_k=2)

    assert results
    assert results[0].chunk_id == 1
    assert results[0].evidence_id == "E1"

