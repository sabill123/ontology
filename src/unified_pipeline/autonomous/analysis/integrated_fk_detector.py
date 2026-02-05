"""
Integrated FK Detector (v16.0)

모든 FK 탐지기를 통합하는 진입점:
- EnhancedFKDetector (기존 단일 FK)
- CompositeFKDetector (복합 FK)
- HierarchyDetector (Self-reference FK)
- TemporalFKDetector (시간 기반 FK)

RelationshipDetectorAgent에서 사용:
    from ..analysis.integrated_fk_detector import detect_all_fks_and_update_context
    result = detect_all_fks_and_update_context(shared_context)

Target: HR 데이터셋 18/18 FK (100% Recall)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from ...shared_context import SharedContext

logger = logging.getLogger(__name__)


@dataclass
class IntegratedFKResult:
    """통합 FK 탐지 결과"""
    # 개별 결과
    single_fk_count: int = 0
    composite_fk_count: int = 0
    hierarchy_fk_count: int = 0
    temporal_fk_count: int = 0

    # 통합 결과
    total_unique_fks: int = 0
    by_confidence: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)

    # 상세 정보
    fk_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "single_fk_count": self.single_fk_count,
            "composite_fk_count": self.composite_fk_count,
            "hierarchy_fk_count": self.hierarchy_fk_count,
            "temporal_fk_count": self.temporal_fk_count,
            "total_unique_fks": self.total_unique_fks,
            "by_confidence": self.by_confidence,
            "by_type": self.by_type,
        }


def detect_all_fks_and_update_context(
    shared_context: "SharedContext",
    enable_single_fk: bool = True,
    enable_composite_fk: bool = True,
    enable_hierarchy: bool = True,
    enable_temporal: bool = True,
    min_score: float = 0.3,
) -> IntegratedFKResult:
    """
    모든 FK 탐지기를 실행하고 SharedContext 업데이트

    Args:
        shared_context: SharedContext 인스턴스
        enable_single_fk: 단일 FK 탐지 활성화
        enable_composite_fk: 복합 FK 탐지 활성화
        enable_hierarchy: 계층(Self-ref) FK 탐지 활성화
        enable_temporal: Temporal FK 탐지 활성화
        min_score: 최소 FK 점수

    Returns:
        IntegratedFKResult

    사용법:
        from ..analysis.integrated_fk_detector import detect_all_fks_and_update_context
        result = detect_all_fks_and_update_context(shared_context)

        # 결과 확인
        print(f"Total FKs: {result.total_unique_fks}")
        print(f"By type: {result.by_type}")
    """
    result = IntegratedFKResult()

    # 중복 제거용 Set
    seen_fks: Set[str] = set()  # "from_table.from_col -> to_table.to_col"

    def add_fk_if_new(fk_dict: Dict[str, Any], fk_type: str) -> bool:
        """중복이 아니면 추가"""
        fk_key = f"{fk_dict.get('from_table')}.{fk_dict.get('from_column')} -> {fk_dict.get('to_table')}.{fk_dict.get('to_column')}"

        if fk_key in seen_fks:
            return False

        seen_fks.add(fk_key)

        # SharedContext에 추가
        if fk_dict not in shared_context.enhanced_fk_candidates:
            fk_dict["detection_method"] = fk_type
            shared_context.enhanced_fk_candidates.append(fk_dict)

        # 통계
        confidence = fk_dict.get("confidence", "low")
        result.by_confidence[confidence] = result.by_confidence.get(confidence, 0) + 1
        result.by_type[fk_type] = result.by_type.get(fk_type, 0) + 1
        result.fk_details.append(fk_dict)

        return True

    # 테이블 데이터 준비
    tables = {}
    sample_data = {}

    for table_name, table_info in shared_context.tables.items():
        tables[table_name] = {
            "columns": table_info.columns if hasattr(table_info, 'columns') else table_info.get('columns', []),
            "row_count": table_info.row_count if hasattr(table_info, 'row_count') else table_info.get('row_count', 0),
            "primary_keys": table_info.primary_keys if hasattr(table_info, 'primary_keys') else table_info.get('primary_keys', []),
        }
        sample_data[table_name] = (
            table_info.sample_data if hasattr(table_info, 'sample_data')
            else table_info.get('sample_data', [])
        )

    logger.info(f"[IntegratedFKDetector] Starting detection on {len(tables)} tables")

    # =====================================================
    # 1. 단일 FK 탐지 (EnhancedFKDetector)
    # =====================================================
    if enable_single_fk:
        try:
            from .enhanced_fk_detector import EnhancedFKDetector, FKCandidate

            detector = EnhancedFKDetector()
            single_fks = detector.detect_foreign_keys(
                tables=tables,
                sample_data=sample_data,
                min_score=min_score,
            )

            for fk in single_fks:
                fk_dict = fk.to_dict() if hasattr(fk, 'to_dict') else {
                    "from_table": fk.from_table,
                    "from_column": fk.from_column,
                    "to_table": fk.to_table,
                    "to_column": fk.to_column,
                    "fk_score": fk.fk_score,
                    "confidence": fk.confidence,
                    "name_similarity": fk.name_similarity,
                    "value_inclusion": fk.value_inclusion,
                }
                if add_fk_if_new(fk_dict, "single_fk"):
                    result.single_fk_count += 1

            logger.info(f"[IntegratedFKDetector] Single FK: {result.single_fk_count} detected")

        except Exception as e:
            logger.warning(f"[IntegratedFKDetector] Single FK detection failed: {e}")

    # =====================================================
    # 2. 복합 FK 탐지 (CompositeFKDetector)
    # =====================================================
    if enable_composite_fk:
        try:
            from .composite_fk_detector import CompositeFKDetector, FKCandidate

            existing_fks = [
                FKCandidate(
                    from_table=d.get("from_table", ""),
                    from_column=d.get("from_column", ""),
                    to_table=d.get("to_table", ""),
                    to_column=d.get("to_column", ""),
                    fk_score=d.get("fk_score", 0),
                )
                for d in shared_context.enhanced_fk_candidates
                if isinstance(d, dict)
            ]

            detector = CompositeFKDetector()
            composite_fks = detector.detect_composite_fks(
                tables=tables,
                sample_data=sample_data,
                existing_fks=existing_fks,
            )

            for cfk in composite_fks:
                # 복합 FK는 개별 컬럼으로 분해하여 추가
                for single_fk in cfk.to_fk_candidates():
                    fk_dict = single_fk.to_dict() if hasattr(single_fk, 'to_dict') else {
                        "from_table": single_fk.from_table,
                        "from_column": single_fk.from_column,
                        "to_table": single_fk.to_table,
                        "to_column": single_fk.to_column,
                        "fk_score": single_fk.fk_score,
                        "confidence": single_fk.confidence,
                    }
                    fk_dict["is_composite"] = True
                    fk_dict["composite_key"] = list(cfk.from_columns)
                    if add_fk_if_new(fk_dict, "composite_fk"):
                        result.composite_fk_count += 1

            # dynamic_data에도 저장
            shared_context.set_dynamic("composite_fk_candidates", [
                cfk.to_dict() for cfk in composite_fks
            ])

            logger.info(f"[IntegratedFKDetector] Composite FK: {result.composite_fk_count} detected")

        except Exception as e:
            logger.warning(f"[IntegratedFKDetector] Composite FK detection failed: {e}")

    # =====================================================
    # 3. 계층 FK 탐지 (HierarchyDetector) - Self-reference
    # =====================================================
    if enable_hierarchy:
        try:
            from .hierarchy_detector import HierarchyDetector

            detector = HierarchyDetector()
            hierarchies = detector.detect_hierarchies(tables, sample_data)

            for h in hierarchies:
                fk_dict = h.to_fk_candidate().to_dict() if hasattr(h, 'to_fk_candidate') else {
                    "from_table": h.table,
                    "from_column": h.child_column,
                    "to_table": h.table,
                    "to_column": h.parent_column,
                    "fk_score": h.confidence,
                    "confidence": "high" if h.confidence >= 0.8 else "medium",
                }
                fk_dict["is_self_reference"] = True
                fk_dict["hierarchy_type"] = h.hierarchy_type
                fk_dict["max_depth"] = h.max_depth

                if add_fk_if_new(fk_dict, "hierarchy_fk"):
                    result.hierarchy_fk_count += 1

            # dynamic_data에도 저장
            shared_context.set_dynamic("hierarchy_candidates", [
                h.to_dict() for h in hierarchies
            ])

            logger.info(f"[IntegratedFKDetector] Hierarchy FK: {result.hierarchy_fk_count} detected")

        except Exception as e:
            logger.warning(f"[IntegratedFKDetector] Hierarchy FK detection failed: {e}")

    # =====================================================
    # 4. Temporal FK 탐지 (TemporalFKDetector)
    # =====================================================
    if enable_temporal:
        try:
            from .temporal_fk_detector import TemporalFKDetector

            # 기존 FK 목록
            existing_fks = []
            for d in shared_context.enhanced_fk_candidates:
                if isinstance(d, dict):
                    existing_fks.append(d)  # dict 형태 그대로 전달

            detector = TemporalFKDetector()
            temporal_fks = detector.detect_temporal_fks(tables, sample_data, existing_fks)

            for tfk in temporal_fks:
                fk_dict = tfk.to_fk_candidate().to_dict() if hasattr(tfk, 'to_fk_candidate') else {
                    "from_table": tfk.from_table,
                    "from_column": tfk.from_column,
                    "to_table": tfk.to_table,
                    "to_column": tfk.to_column,
                    "fk_score": tfk.confidence,
                    "confidence": "high" if tfk.confidence >= 0.8 else "medium",
                }
                fk_dict["is_temporal"] = True
                fk_dict["temporal_type"] = tfk.temporal_type.value
                fk_dict["valid_from_col"] = tfk.valid_from_col
                fk_dict["valid_to_col"] = tfk.valid_to_col

                if add_fk_if_new(fk_dict, "temporal_fk"):
                    result.temporal_fk_count += 1

            # dynamic_data에도 저장
            shared_context.set_dynamic("temporal_fk_candidates", [
                tfk.to_dict() for tfk in temporal_fks
            ])

            logger.info(f"[IntegratedFKDetector] Temporal FK: {result.temporal_fk_count} detected")

        except Exception as e:
            logger.warning(f"[IntegratedFKDetector] Temporal FK detection failed: {e}")

    # =====================================================
    # 결과 정리
    # =====================================================
    result.total_unique_fks = len(seen_fks)

    # 통합 결과를 dynamic_data에 저장
    shared_context.set_dynamic("integrated_fk_result", result.to_dict())

    logger.info(
        f"[IntegratedFKDetector] TOTAL: {result.total_unique_fks} unique FKs "
        f"(single={result.single_fk_count}, composite={result.composite_fk_count}, "
        f"hierarchy={result.hierarchy_fk_count}, temporal={result.temporal_fk_count})"
    )

    return result


def get_fk_detection_summary(shared_context: "SharedContext") -> Dict[str, Any]:
    """
    FK 탐지 결과 요약

    사용법:
        summary = get_fk_detection_summary(shared_context)
        print(summary)
    """
    result = shared_context.get_dynamic("integrated_fk_result", {})

    by_table = {}
    for fk in shared_context.enhanced_fk_candidates:
        if isinstance(fk, dict):
            from_table = fk.get("from_table", "unknown")
            by_table[from_table] = by_table.get(from_table, 0) + 1

    return {
        "total_fks": len(shared_context.enhanced_fk_candidates),
        "by_detection_method": result.get("by_type", {}),
        "by_confidence": result.get("by_confidence", {}),
        "by_source_table": by_table,
    }


# 모듈 초기화
__all__ = [
    "detect_all_fks_and_update_context",
    "get_fk_detection_summary",
    "IntegratedFKResult",
]
