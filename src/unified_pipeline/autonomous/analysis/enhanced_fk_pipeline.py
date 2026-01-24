# analysis/enhanced_fk_pipeline.py
"""
Enhanced FK Detection Pipeline

Multi-Signal + Direction + Semantic + LLM 통합 파이프라인
- Solution 1: FKSignalScorer (다중 신호 기반 점수)
- Solution 2: FKDirectionAnalyzer (확률적 방향 분석)
- Solution 3: SemanticEntityResolver (의미론적 관계 추론)
- Solution 4: LLMSemanticEnhancer (LLM 기반 semantic 보완)

v8.1: LLM Semantic Enhancer 통합 - 범용 FK 탐지
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import csv
import logging

from .fk_signal_scorer import FKSignalScorer, FKSignals
from .fk_direction_analyzer import FKDirectionAnalyzer, DirectionEvidence
from .semantic_entity_resolver import SemanticEntityResolver

# LLM Enhancer (optional)
try:
    from .llm_semantic_enhancer import LLMSemanticEnhancer, SemanticFKResult
    _LLM_ENHANCER_AVAILABLE = True
except ImportError:
    _LLM_ENHANCER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EnhancedFKCandidate:
    """Enhanced FK 후보"""
    from_table: str
    from_column: str
    to_table: str
    to_column: str

    # Confidence scores
    signal_confidence: float = 0.0
    direction_confidence: float = 0.0
    semantic_confidence: float = 0.0
    final_confidence: float = 0.0

    # Analysis details
    signal_breakdown: Dict = field(default_factory=dict)
    direction_evidences: List[Dict] = field(default_factory=list)
    semantic_analysis: Dict = field(default_factory=dict)

    # Flags
    direction_certain: bool = False
    is_bidirectional_pair: bool = False

    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column,
            "confidence": self.final_confidence,
            "signal_confidence": self.signal_confidence,
            "direction_confidence": self.direction_confidence,
            "semantic_confidence": self.semantic_confidence,
            "direction_certain": self.direction_certain,
            "signal_breakdown": self.signal_breakdown,
            "direction_evidences": self.direction_evidences,
            "semantic_analysis": self.semantic_analysis,
        }


class EnhancedFKPipeline:
    """
    개선된 FK Detection Pipeline

    Multi-Signal + Direction + Semantic + LLM Enhancement

    Hybrid 방식:
    1. Rule-based 탐지 (빠름, 저렴)
    2. LLM 기반 보완 (semantic role 추론) - optional
    """

    def __init__(
        self,
        signal_weight: float = 0.5,
        direction_weight: float = 0.25,
        semantic_weight: float = 0.25,
        min_confidence: float = 0.4,  # 조기 제외 threshold (낮게 유지)
        enable_llm: bool = True,  # LLM semantic enhancement
        llm_min_confidence: float = 0.6,  # LLM 결과 최소 신뢰도
        domain_context: str = "",  # 도메인 컨텍스트
    ):
        self.signal_scorer = FKSignalScorer()
        self.direction_analyzer = FKDirectionAnalyzer()
        self.semantic_resolver = SemanticEntityResolver()

        self.signal_weight = signal_weight
        self.direction_weight = direction_weight
        self.semantic_weight = semantic_weight
        self.min_confidence = min_confidence
        self.domain_context = domain_context

        # LLM Enhancer 초기화
        self.enable_llm = enable_llm and _LLM_ENHANCER_AVAILABLE
        self.llm_min_confidence = llm_min_confidence
        self._llm_enhancer = None

        if self.enable_llm:
            try:
                self._llm_enhancer = LLMSemanticEnhancer(
                    use_instructor=True,
                    use_dspy=False,
                    min_confidence=llm_min_confidence,
                )
                logger.info("LLM Semantic Enhancer initialized")
            except Exception as e:
                logger.warning(f"LLM Enhancer init failed: {e}")
                self.enable_llm = False

    def detect(
        self,
        data_path: str,
        return_all: bool = False,
    ) -> List[Dict]:
        """
        FK 탐지 실행

        Args:
            data_path: 데이터 디렉토리 경로
            return_all: True면 모든 후보 반환, False면 confident한 것만

        Returns:
            FK 후보 리스트
        """
        # 1. 데이터 로드
        tables_data = self._load_data(data_path)
        if not tables_data:
            logger.warning(f"No data loaded from {data_path}")
            return []

        logger.info(f"Loaded {len(tables_data)} tables from {data_path}")

        # 2. 모든 컬럼 쌍에 대해 분석 (Rule-based)
        candidates = []
        column_pairs = list(self._get_column_pairs(tables_data))
        logger.info(f"Analyzing {len(column_pairs)} column pairs...")

        for (table_a, col_a), (table_b, col_b) in column_pairs:
            candidate = self._analyze_pair(
                table_a, col_a,
                table_b, col_b,
                tables_data,
            )

            if candidate is None:
                continue

            # 조기 제외: 명확한 non-FK
            if candidate.final_confidence < self.min_confidence:
                continue

            candidates.append(candidate)

        logger.info(f"Found {len(candidates)} candidates after initial analysis")

        # 3. 양방향 쌍 처리
        candidates = self._resolve_bidirectional(candidates)
        logger.info(f"{len(candidates)} candidates after bidirectional resolution")

        # 4. LLM Semantic Enhancement (optional)
        if self.enable_llm and self._llm_enhancer:
            candidates = self._enhance_with_llm(candidates, tables_data)
            logger.info(f"{len(candidates)} candidates after LLM enhancement")

        # 5. confidence 순 정렬
        candidates.sort(key=lambda x: x.final_confidence, reverse=True)

        # 6. 결과 변환
        if return_all:
            return [c.to_dict() for c in candidates]
        else:
            # direction_certain이거나 높은 confidence만 반환
            confident = [c for c in candidates
                        if c.direction_certain or c.final_confidence >= 0.6]
            return [c.to_dict() for c in confident]

    def _enhance_with_llm(
        self,
        candidates: List[EnhancedFKCandidate],
        tables_data: Dict[str, Dict[str, List[Any]]],
    ) -> List[EnhancedFKCandidate]:
        """
        LLM 기반 Semantic Enhancement

        Rule-based semantic confidence가 낮은 후보들에 대해 LLM으로 재분석
        """
        enhanced = []
        llm_enhanced_count = 0

        for candidate in candidates:
            # Semantic confidence가 이미 높으면 스킵
            if candidate.semantic_confidence >= 0.7:
                enhanced.append(candidate)
                continue

            # LLM 사용 여부 결정
            if not self._llm_enhancer.should_use_llm(
                candidate.from_column,
                candidate.semantic_confidence
            ):
                enhanced.append(candidate)
                continue

            # 샘플 데이터 추출
            fk_samples = tables_data.get(candidate.from_table, {}).get(candidate.from_column, [])[:15]
            pk_samples = tables_data.get(candidate.to_table, {}).get(candidate.to_column, [])[:15]

            # 오버랩 계산
            fk_set = set(str(v) for v in fk_samples if v)
            pk_set = set(str(v) for v in pk_samples if v)
            overlap = len(fk_set & pk_set) / len(fk_set) if fk_set else 0.0

            try:
                # LLM 분석
                llm_result = self._llm_enhancer.analyze_semantic_fk(
                    fk_column=candidate.from_column,
                    fk_table=candidate.from_table,
                    pk_column=candidate.to_column,
                    pk_table=candidate.to_table,
                    fk_samples=[str(v) for v in fk_samples],
                    pk_samples=[str(v) for v in pk_samples],
                    overlap_ratio=overlap,
                    domain_context=self.domain_context,
                )

                # LLM 결과로 업데이트
                if llm_result.is_relationship and llm_result.confidence >= self.llm_min_confidence:
                    # Semantic confidence 업데이트
                    old_semantic = candidate.semantic_confidence
                    candidate.semantic_confidence = max(
                        candidate.semantic_confidence,
                        llm_result.confidence
                    )

                    # Final confidence 재계산
                    candidate.final_confidence = self._combine_confidences(
                        candidate.signal_confidence,
                        candidate.direction_confidence,
                        candidate.semantic_confidence,
                    )

                    # LLM 분석 결과 추가
                    candidate.semantic_analysis["llm_enhanced"] = True
                    candidate.semantic_analysis["llm_confidence"] = llm_result.confidence
                    candidate.semantic_analysis["llm_relationship"] = llm_result.semantic_relationship
                    candidate.semantic_analysis["llm_reasoning"] = llm_result.reasoning

                    llm_enhanced_count += 1
                    logger.info(
                        f"LLM enhanced: {candidate.from_table}.{candidate.from_column} → "
                        f"{candidate.to_table}.{candidate.to_column} "
                        f"(semantic: {old_semantic:.2f} → {candidate.semantic_confidence:.2f})"
                    )

            except Exception as e:
                logger.warning(f"LLM enhancement failed for {candidate.from_column}: {e}")

            enhanced.append(candidate)

        logger.info(f"LLM enhanced {llm_enhanced_count} candidates")
        return enhanced

    def _analyze_pair(
        self,
        table_a: str,
        col_a: str,
        table_b: str,
        col_b: str,
        tables_data: Dict[str, Dict[str, List[Any]]],
    ) -> Optional[EnhancedFKCandidate]:
        """단일 컬럼 쌍 분석"""

        values_a = tables_data.get(table_a, {}).get(col_a, [])
        values_b = tables_data.get(table_b, {}).get(col_b, [])

        if not values_a or not values_b:
            return None

        # === Step 1: Multi-Signal Scoring ===
        signals = self.signal_scorer.compute_signals(
            fk_values=values_a,
            pk_values=values_b,
            fk_column=col_a,
            pk_column=col_b,
            fk_table=table_a,
            pk_table=table_b,
        )
        signal_conf, signal_breakdown = self.signal_scorer.compute_confidence(signals)

        # 매우 낮은 signal은 조기 제외
        if signal_conf < 0.2:
            return None

        # === Step 2: Direction Analysis ===
        fk_col, pk_col, dir_conf, dir_evidences = self.direction_analyzer.analyze_direction(
            col_a, col_b,
            values_a, values_b,
            table_a, table_b,
        )

        direction_certain = fk_col is not None

        # 방향이 결정되었다면 해당 방향 사용
        if direction_certain:
            # fk_col format: "table.column"
            fk_parts = fk_col.split(".")
            pk_parts = pk_col.split(".")
            from_table, from_column = fk_parts[0], fk_parts[1]
            to_table, to_column = pk_parts[0], pk_parts[1]
        else:
            # 방향 불확실 - 일단 A→B로 설정
            from_table, from_column = table_a, col_a
            to_table, to_column = table_b, col_b

        # === Step 3: Semantic Analysis ===
        semantic = self.semantic_resolver.infer_fk_relationship(
            fk_column=from_column,
            fk_table=from_table,
            pk_column=to_column,
            pk_table=to_table,
        )
        semantic_conf = semantic.get("semantic_confidence", 0.0)

        # === Step 4: Combine Confidences ===
        final_conf = self._combine_confidences(
            signal_conf, dir_conf, semantic_conf
        )

        return EnhancedFKCandidate(
            from_table=from_table,
            from_column=from_column,
            to_table=to_table,
            to_column=to_column,
            signal_confidence=signal_conf,
            direction_confidence=dir_conf,
            semantic_confidence=semantic_conf,
            final_confidence=final_conf,
            signal_breakdown=signal_breakdown,
            direction_evidences=[e.__dict__ for e in dir_evidences],
            semantic_analysis=semantic,
            direction_certain=direction_certain,
        )

    def _combine_confidences(
        self,
        signal_conf: float,
        direction_conf: float,
        semantic_conf: float,
    ) -> float:
        """종합 confidence 계산"""
        # 가중 평균
        weighted = (
            signal_conf * self.signal_weight +
            direction_conf * self.direction_weight +
            semantic_conf * self.semantic_weight
        )

        # Bonus: 모든 신호가 긍정적이면 가점
        if signal_conf > 0.5 and direction_conf > 0.5 and semantic_conf > 0.5:
            weighted = min(weighted + 0.1, 1.0)

        return weighted

    def _resolve_bidirectional(
        self,
        candidates: List[EnhancedFKCandidate],
    ) -> List[EnhancedFKCandidate]:
        """양방향 쌍 해결"""

        # 쌍 그룹화
        pair_map: Dict[frozenset, List[EnhancedFKCandidate]] = {}

        for c in candidates:
            pair = frozenset([
                (c.from_table, c.from_column),
                (c.to_table, c.to_column)
            ])
            if pair not in pair_map:
                pair_map[pair] = []
            pair_map[pair].append(c)

        resolved = []

        for pair, group in pair_map.items():
            if len(group) == 1:
                # 단방향
                resolved.append(group[0])
            else:
                # 양방향 - 더 확실한 방향 선택
                # direction_certain인 것 우선
                certain = [c for c in group if c.direction_certain]
                if certain:
                    best = max(certain, key=lambda x: x.final_confidence)
                else:
                    # 모두 uncertain이면 confidence로
                    best = max(group, key=lambda x: x.final_confidence)

                best.is_bidirectional_pair = True
                resolved.append(best)

                logger.debug(f"Resolved bidirectional pair: "
                           f"{best.from_table}.{best.from_column} → "
                           f"{best.to_table}.{best.to_column}")

        return resolved

    def _load_data(
        self,
        data_path: str,
    ) -> Dict[str, Dict[str, List[Any]]]:
        """데이터 로드"""
        path = Path(data_path)

        if not path.exists():
            logger.error(f"Path does not exist: {data_path}")
            return {}

        tables_data = {}

        # CSV 파일들 로드
        csv_files = list(path.glob("*.csv"))

        for csv_file in csv_files:
            table_name = csv_file.stem
            tables_data[table_name] = {}

            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)

                    # 모든 행 읽기
                    rows = list(reader)

                    if not rows:
                        continue

                    # 컬럼별 값 추출
                    for col in rows[0].keys():
                        tables_data[table_name][col] = [row.get(col) for row in rows]

            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")

        return tables_data

    def _get_column_pairs(
        self,
        tables_data: Dict[str, Dict[str, List[Any]]],
    ):
        """모든 테이블 간 컬럼 쌍 생성"""

        tables = list(tables_data.keys())

        for i, table_a in enumerate(tables):
            cols_a = list(tables_data[table_a].keys())

            for j, table_b in enumerate(tables):
                if i == j:
                    continue  # 같은 테이블 스킵

                cols_b = list(tables_data[table_b].keys())

                for col_a in cols_a:
                    for col_b in cols_b:
                        # 기본 필터: ID-like 컬럼만
                        if self._is_potential_key(col_a) or self._is_potential_key(col_b):
                            yield (table_a, col_a), (table_b, col_b)

    def _is_potential_key(self, col_name: str) -> bool:
        """키 컬럼 가능성 체크"""
        col = col_name.lower()
        key_patterns = ['_id', '_code', '_key', '_no', '_ref', '_num', '_pk', '_fk']
        return any(pattern in col for pattern in key_patterns) or col == 'id'


# Factory function
def create_enhanced_detector(**kwargs) -> EnhancedFKPipeline:
    """Enhanced FK Detector 생성"""
    return EnhancedFKPipeline(**kwargs)


# Quick detection function
def detect_fks(data_path: str, **kwargs) -> List[Dict]:
    """
    빠른 FK 탐지

    Args:
        data_path: 데이터 디렉토리 경로

    Returns:
        FK 후보 리스트
    """
    pipeline = EnhancedFKPipeline(**kwargs)
    return pipeline.detect(data_path)
