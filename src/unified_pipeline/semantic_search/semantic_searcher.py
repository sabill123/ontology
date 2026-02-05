"""
Semantic Searcher (v17.0)

벡터 기반 의미 검색 엔진:
- 텍스트 임베딩 생성
- 벡터 인덱스 관리 (FAISS)
- 유사도 검색
- Cross-table 엔티티 연결

사용 예시:
    searcher = SemanticSearcher(context)

    # 테이블 인덱싱
    await searcher.index_table("customers", ["name", "description"])

    # 검색
    results = await searcher.search("reliable supplier in Seoul", limit=10)

    # 유사 엔티티 찾기
    similar = await searcher.find_similar("customer_123", top_k=5)
"""

import logging
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

if TYPE_CHECKING:
    from ..shared_context import SharedContext

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """검색 설정"""
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    index_type: str = "flat"  # "flat", "ivf", "hnsw"
    nlist: int = 100  # IVF clusters
    nprobe: int = 10  # IVF search probes
    ef_search: int = 64  # HNSW search effort


@dataclass
class SearchResult:
    """검색 결과"""
    entity_id: str
    table_name: str
    score: float
    data: Dict[str, Any]
    matched_text: str = ""

    # 메타데이터
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "table_name": self.table_name,
            "score": self.score,
            "data": self.data,
            "matched_text": self.matched_text,
            "rank": self.rank,
        }


@dataclass
class IndexedEntity:
    """인덱싱된 엔티티"""
    entity_id: str
    table_name: str
    text: str
    embedding_id: int
    data: Dict[str, Any]


class SemanticSearcher:
    """
    시맨틱 검색 엔진 (v17.0 - LLM 통합)

    기능:
    - 텍스트 임베딩 생성
    - 벡터 인덱스 관리
    - 유사도 검색
    - Cross-table 엔티티 발견
    - LLM 기반 쿼리 확장
    - LLM 기반 결과 요약
    """

    def __init__(
        self,
        context: Optional["SharedContext"] = None,
        config: Optional[SearchConfig] = None,
        llm_client=None,
    ):
        """
        Args:
            context: SharedContext
            config: 검색 설정
            llm_client: LLM 클라이언트 (쿼리 확장/결과 요약에 사용)
        """
        self.context = context
        self.config = config or SearchConfig()
        self.llm_client = llm_client

        # 임베딩 모델 (v22.0: lazy-load to avoid SIGBUS on init)
        self.model = None
        self._model_loaded = False

        # 벡터 인덱스
        self.index = None
        self._init_index()

        # 엔티티 매핑
        self.entities: Dict[int, IndexedEntity] = {}  # embedding_id -> entity
        self.entity_to_id: Dict[str, int] = {}  # entity_id -> embedding_id

        # 인덱싱 상태
        self.indexed_tables: Dict[str, List[str]] = {}  # table -> columns

        # 통계
        self.stats = {
            "total_indexed": 0,
            "total_searches": 0,
            "llm_query_expansions": 0,
            "llm_summarizations": 0,
        }

        logger.info(f"SemanticSearcher initialized (LLM: {'enabled' if llm_client else 'disabled'})")

    def _init_index(self) -> None:
        """FAISS 인덱스 초기화"""
        if not FAISS_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("FAISS or NumPy not available, using fallback")
            return

        dim = self.config.embedding_dim

        if self.config.index_type == "flat":
            self.index = faiss.IndexFlatIP(dim)  # Inner Product (cosine similarity)
        elif self.config.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, dim, self.config.nlist, faiss.METRIC_INNER_PRODUCT
            )
        elif self.config.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)

        logger.info(f"Initialized FAISS index: {self.config.index_type}")

    async def index_table(
        self,
        table_name: str,
        text_columns: List[str],
        id_column: Optional[str] = None,
    ) -> int:
        """
        테이블 인덱싱

        Args:
            table_name: 테이블 이름
            text_columns: 인덱싱할 텍스트 컬럼들
            id_column: ID 컬럼 (기본: 'id' 또는 첫 번째 컬럼)

        Returns:
            인덱싱된 엔티티 수
        """
        if not self.context or table_name not in self.context.tables:
            logger.error(f"Table not found: {table_name}")
            return 0

        table_info = self.context.tables[table_name]
        sample_data = table_info.get("sample_data", [])

        if not sample_data:
            logger.warning(f"No data to index for table: {table_name}")
            return 0

        # ID 컬럼 결정
        if not id_column:
            columns = table_info.get("columns", {})
            id_column = "id" if "id" in columns else list(columns.keys())[0] if columns else None

        indexed_count = 0
        texts = []
        entities_to_add = []

        for row in sample_data:
            # 텍스트 추출
            text_parts = []
            for col in text_columns:
                value = row.get(col)
                if value:
                    text_parts.append(str(value))

            if not text_parts:
                continue

            text = " ".join(text_parts)
            entity_id = str(row.get(id_column, f"{table_name}_{indexed_count}"))

            texts.append(text)
            entities_to_add.append({
                "entity_id": entity_id,
                "table_name": table_name,
                "text": text,
                "data": row,
            })
            indexed_count += 1

        # 임베딩 생성 및 인덱싱
        if texts:
            embeddings = self._encode(texts)
            self._add_to_index(embeddings, entities_to_add)

        # 상태 업데이트
        self.indexed_tables[table_name] = text_columns
        self.stats["total_indexed"] += indexed_count

        logger.info(f"Indexed {indexed_count} entities from {table_name}")

        return indexed_count

    async def search(
        self,
        query: str,
        limit: int = 10,
        tables: Optional[List[str]] = None,
        min_score: float = 0.0,
        expand_query: bool = False,
    ) -> List[SearchResult]:
        """
        시맨틱 검색

        Args:
            query: 검색 쿼리
            limit: 최대 결과 수
            tables: 검색할 테이블 (None이면 전체)
            min_score: 최소 유사도 점수
            expand_query: LLM으로 쿼리 확장 여부

        Returns:
            검색 결과 목록
        """
        self.stats["total_searches"] += 1

        # LLM으로 쿼리 확장 (옵션)
        search_query = query
        if expand_query and self.llm_client:
            expanded = await self._expand_query_with_llm(query)
            if expanded:
                search_query = expanded
                self.stats["llm_query_expansions"] += 1

        # 쿼리 임베딩
        query_embedding = self._encode([search_query])[0]

        # 검색
        scores, ids = self._search_index(query_embedding, limit * 2)

        results = []
        for score, idx in zip(scores, ids):
            if idx < 0:
                continue

            entity = self.entities.get(idx)
            if not entity:
                continue

            # 테이블 필터
            if tables and entity.table_name not in tables:
                continue

            # 점수 필터
            if score < min_score:
                continue

            results.append(SearchResult(
                entity_id=entity.entity_id,
                table_name=entity.table_name,
                score=float(score),
                data=entity.data,
                matched_text=entity.text,
                rank=len(results) + 1,
            ))

            if len(results) >= limit:
                break

        # Evidence Chain 기록
        self._record_search_to_evidence_chain(query, results)

        return results

    async def find_similar(
        self,
        entity_id: str,
        top_k: int = 5,
        exclude_same_table: bool = False,
    ) -> List[SearchResult]:
        """
        유사 엔티티 찾기

        Args:
            entity_id: 기준 엔티티 ID
            top_k: 반환할 결과 수
            exclude_same_table: 같은 테이블 제외 여부

        Returns:
            유사 엔티티 목록
        """
        if entity_id not in self.entity_to_id:
            logger.warning(f"Entity not found: {entity_id}")
            return []

        embedding_id = self.entity_to_id[entity_id]
        source_entity = self.entities[embedding_id]

        # 엔티티 임베딩으로 검색
        embedding = self._get_embedding(embedding_id)
        if embedding is None:
            return []

        scores, ids = self._search_index(embedding, top_k + 10)

        results = []
        for score, idx in zip(scores, ids):
            if idx < 0 or idx == embedding_id:
                continue

            entity = self.entities.get(idx)
            if not entity:
                continue

            # 같은 테이블 제외
            if exclude_same_table and entity.table_name == source_entity.table_name:
                continue

            results.append(SearchResult(
                entity_id=entity.entity_id,
                table_name=entity.table_name,
                score=float(score),
                data=entity.data,
                matched_text=entity.text,
                rank=len(results) + 1,
            ))

            if len(results) >= top_k:
                break

        return results

    async def find_cross_table_matches(
        self,
        source_table: str,
        target_table: str,
        threshold: float = 0.8,
    ) -> List[Tuple[SearchResult, SearchResult]]:
        """
        Cross-table 엔티티 매칭

        Args:
            source_table: 소스 테이블
            target_table: 타겟 테이블
            threshold: 매칭 임계값

        Returns:
            매칭된 엔티티 쌍 목록
        """
        matches = []

        # 소스 테이블의 모든 엔티티
        source_entities = [
            (eid, e) for eid, e in self.entities.items()
            if e.table_name == source_table
        ]

        for embedding_id, source_entity in source_entities:
            # 타겟 테이블에서 유사 엔티티 검색
            similar = await self.find_similar(
                source_entity.entity_id,
                top_k=3,
                exclude_same_table=True,
            )

            for result in similar:
                if result.table_name == target_table and result.score >= threshold:
                    source_result = SearchResult(
                        entity_id=source_entity.entity_id,
                        table_name=source_entity.table_name,
                        score=1.0,
                        data=source_entity.data,
                        matched_text=source_entity.text,
                    )
                    matches.append((source_result, result))

        return matches

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """텍스트 임베딩"""
        # v22.0: lazy-load model on first use
        if not self._model_loaded and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.config.model_name)
                logger.info(f"Loaded embedding model: {self.config.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
            self._model_loaded = True

        if self.model and SENTENCE_TRANSFORMERS_AVAILABLE:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()

        # 폴백: 간단한 해시 기반 "임베딩"
        embeddings = []
        for text in texts:
            hash_val = hashlib.md5(text.encode()).hexdigest()
            # 해시를 고정 차원 벡터로 변환
            embedding = [
                (int(hash_val[i:i+2], 16) / 255.0 - 0.5) * 2
                for i in range(0, min(len(hash_val), self.config.embedding_dim * 2), 2)
            ]
            # 차원 맞추기
            while len(embedding) < self.config.embedding_dim:
                embedding.append(0.0)
            embedding = embedding[:self.config.embedding_dim]
            embeddings.append(embedding)

        return embeddings

    def _add_to_index(
        self,
        embeddings: List[List[float]],
        entities_info: List[Dict[str, Any]],
    ) -> None:
        """인덱스에 추가"""
        if not FAISS_AVAILABLE or not NUMPY_AVAILABLE or not self.index:
            # 폴백: 메모리 저장
            for i, info in enumerate(entities_info):
                embedding_id = len(self.entities)
                entity = IndexedEntity(
                    entity_id=info["entity_id"],
                    table_name=info["table_name"],
                    text=info["text"],
                    embedding_id=embedding_id,
                    data=info["data"],
                )
                self.entities[embedding_id] = entity
                self.entity_to_id[info["entity_id"]] = embedding_id
            return

        # FAISS 인덱스에 추가
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)

        start_id = len(self.entities)

        for i, info in enumerate(entities_info):
            embedding_id = start_id + i
            entity = IndexedEntity(
                entity_id=info["entity_id"],
                table_name=info["table_name"],
                text=info["text"],
                embedding_id=embedding_id,
                data=info["data"],
            )
            self.entities[embedding_id] = entity
            self.entity_to_id[info["entity_id"]] = embedding_id

        self.index.add(embeddings_array)

    def _search_index(
        self,
        query_embedding: List[float],
        k: int,
    ) -> Tuple[List[float], List[int]]:
        """인덱스 검색"""
        if not FAISS_AVAILABLE or not NUMPY_AVAILABLE or not self.index:
            # 폴백: 브루트포스 검색
            return self._fallback_search(query_embedding, k)

        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)

        scores, ids = self.index.search(query_array, k)

        return scores[0].tolist(), ids[0].tolist()

    def _fallback_search(
        self,
        query_embedding: List[float],
        k: int,
    ) -> Tuple[List[float], List[int]]:
        """폴백 브루트포스 검색"""
        # 간단한 코사인 유사도 계산
        scores_ids = []

        for embedding_id, entity in self.entities.items():
            # 저장된 임베딩이 없으면 재생성
            entity_embedding = self._encode([entity.text])[0]
            score = self._cosine_similarity(query_embedding, entity_embedding)
            scores_ids.append((score, embedding_id))

        # 정렬
        scores_ids.sort(key=lambda x: x[0], reverse=True)
        scores_ids = scores_ids[:k]

        scores = [s for s, _ in scores_ids]
        ids = [i for _, i in scores_ids]

        return scores, ids

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """코사인 유사도 계산"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _get_embedding(self, embedding_id: int) -> Optional[List[float]]:
        """저장된 임베딩 조회"""
        if embedding_id not in self.entities:
            return None

        entity = self.entities[embedding_id]
        return self._encode([entity.text])[0]

    def _record_search_to_evidence_chain(
        self,
        query: str,
        results: List[SearchResult],
    ) -> None:
        """Evidence Chain에 검색 기록"""
        if not self.context or not self.context.evidence_chain:
            return

        try:
            from ..autonomous.evidence_chain import EvidenceType

            evidence_type = getattr(
                EvidenceType,
                "SEMANTIC_SEARCH",
                EvidenceType.QUALITY_ASSESSMENT
            )

            self.context.evidence_chain.add_block(
                phase="semantic_search",
                agent="SemanticSearcher",
                evidence_type=evidence_type,
                finding=f"Searched for: {query[:100]}",
                reasoning=f"Found {len(results)} results",
                conclusion=(
                    f"Top result: {results[0].entity_id} (score: {results[0].score:.2f})"
                    if results else "No results found"
                ),
                metrics={
                    "query": query,
                    "result_count": len(results),
                    "top_scores": [r.score for r in results],
                },
                confidence=results[0].score if results else 0.0,
            )
        except Exception as e:
            logger.warning(f"Failed to record to evidence chain: {e}")

    def get_search_summary(self) -> Dict[str, Any]:
        """검색 요약"""
        return {
            "stats": self.stats,
            "indexed_tables": self.indexed_tables,
            "total_entities": len(self.entities),
            "model_name": self.config.model_name,
            "index_type": self.config.index_type,
            "faiss_available": FAISS_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "llm_enabled": self.llm_client is not None,
        }

    # =========================================================================
    # LLM 기반 기능 (v17.0)
    # =========================================================================

    async def _expand_query_with_llm(self, query: str) -> Optional[str]:
        """
        LLM을 사용하여 검색 쿼리 확장

        동의어, 관련 개념, 도메인 특화 용어를 추가합니다.
        """
        if not self.llm_client:
            return None

        try:
            from ..model_config import get_v17_service_model
            from ...common.utils.llm import chat_completion

            model = get_v17_service_model("semantic_searcher")

            # 도메인 컨텍스트 수집
            domain_context = ""
            if self.context:
                tables = list(self.context.tables.keys())
                domain_context = f"데이터베이스 테이블: {', '.join(tables)}"

            prompt = f"""검색 쿼리를 확장해주세요. 동의어, 관련 개념, 도메인 특화 용어를 추가합니다.

원본 쿼리: "{query}"
{domain_context}

확장된 검색 쿼리만 반환해주세요 (따옴표 없이, 설명 없이).
원본 의미를 유지하면서 검색 커버리지를 높이는 키워드를 추가하세요.
"""

            # v18.0: asyncio.to_thread로 블로킹 호출 방지
            import asyncio
            response = await asyncio.to_thread(
                chat_completion,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            expanded = response.strip()

            # 원본과 너무 다르면 원본 반환
            if len(expanded) > len(query) * 5:
                return query

            logger.debug(f"Query expanded: '{query}' -> '{expanded}'")
            return expanded

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return None

    async def summarize_results(
        self,
        query: str,
        results: List[SearchResult],
        max_results: int = 5,
    ) -> str:
        """
        검색 결과를 자연어로 요약

        Args:
            query: 원본 검색 쿼리
            results: 검색 결과 목록
            max_results: 요약에 포함할 최대 결과 수

        Returns:
            자연어 요약 문자열
        """
        if not results:
            return f"'{query}'에 대한 검색 결과가 없습니다."

        # LLM 없으면 기본 요약
        if not self.llm_client:
            return self._summarize_results_basic(query, results, max_results)

        try:
            from ..model_config import get_v17_service_model
            from ...common.utils.llm import chat_completion

            model = get_v17_service_model("semantic_searcher")

            # 결과 포맷팅
            results_text = []
            for i, r in enumerate(results):
                results_text.append(
                    f"{i+1}. [{r.table_name}] {r.entity_id}: {r.matched_text} (유사도: {r.score:.2f})"
                )

            prompt = f"""검색 결과를 간결하게 요약해주세요.

검색어: "{query}"

검색 결과:
{chr(10).join(results_text)}

요약 요구사항:
1. 검색어와 결과의 관련성 설명
2. 주요 발견사항 2-3개
3. 2-3문장으로 간결하게

요약:"""

            # v18.0: asyncio.to_thread로 블로킹 호출 방지
            import asyncio
            response = await asyncio.to_thread(
                chat_completion,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            summary = response.strip()
            self.stats["llm_summarizations"] += 1

            return summary

        except Exception as e:
            logger.warning(f"Result summarization failed: {e}")
            return self._summarize_results_basic(query, results, max_results)

    def _summarize_results_basic(
        self,
        query: str,
        results: List[SearchResult],
        max_results: int,
    ) -> str:
        """LLM 없이 기본 요약 생성"""
        top_results = results[:max_results]

        tables = set(r.table_name for r in top_results)
        avg_score = sum(r.score for r in top_results) / len(top_results)

        summary_parts = [
            f"'{query}' 검색 결과: {len(results)}건 발견",
            f"관련 테이블: {', '.join(tables)}",
            f"평균 유사도: {avg_score:.2f}",
        ]

        if top_results:
            top = top_results[0]
            summary_parts.append(
                f"최상위 결과: [{top.table_name}] {top.entity_id}"
            )

        return " | ".join(summary_parts)

    async def search_with_summary(
        self,
        query: str,
        limit: int = 10,
        tables: Optional[List[str]] = None,
        expand_query: bool = True,
    ) -> Dict[str, Any]:
        """
        검색 + 자동 요약 (편의 메서드)

        Returns:
            {
                "results": List[SearchResult],
                "summary": str,
                "expanded_query": str (if expanded)
            }
        """
        # 검색 실행
        results = await self.search(
            query=query,
            limit=limit,
            tables=tables,
            expand_query=expand_query,
        )

        # 요약 생성
        summary = await self.summarize_results(query, results)

        return {
            "results": [r.to_dict() for r in results],
            "summary": summary,
            "result_count": len(results),
            "query": query,
        }
