from filinglens.models import StoredChunk
from filinglens.retrieval import BM25Retriever, tokenize


def test_tokenize_adds_chinese_bigrams():
    tokens = tokenize("\u6536\u5165\u589e\u957f AI")

    assert "\u6536\u5165" in tokens
    assert "\u589e\u957f" in tokens
    assert "ai" in tokens


def test_bm25_prefers_relevant_chunk():
    chunks = [
        StoredChunk(
            id=1,
            report_id=1,
            chunk_index=0,
            page_number=1,
            text="Revenue growth was driven mainly by overseas expansion.",
        ),
        StoredChunk(
            id=2,
            report_id=1,
            chunk_index=1,
            page_number=2,
            text="The company maintains a comprehensive information security policy.",
        ),
    ]

    results = BM25Retriever().search("What drove revenue growth?", chunks, top_k=2)

    assert results
    assert results[0].chunk_id == 1
    assert results[0].evidence_id == "E1"

