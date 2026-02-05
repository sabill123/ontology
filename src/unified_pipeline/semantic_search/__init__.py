"""
Semantic Search Module (v17.0)

벡터 기반 의미 검색:
- 임베딩 생성
- 벡터 저장소 (FAISS)
- 시맨틱 검색
- Cross-table 엔티티 연결

사용 예시:
    searcher = SemanticSearcher(context)

    # 인덱싱
    await searcher.index_table("customers", ["name", "description"])

    # 검색
    results = await searcher.search("reliable supplier in Seoul")
"""

from .semantic_searcher import (
    SemanticSearcher,
    SearchResult,
    SearchConfig,
)

__all__ = [
    "SemanticSearcher",
    "SearchResult",
    "SearchConfig",
]
