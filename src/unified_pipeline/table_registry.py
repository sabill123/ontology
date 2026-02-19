"""
Table Registry

TableInfo / raw_data / data_directory / full-data access methods.
Extracted from SharedContext to reduce God Object complexity.
"""

from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .shared_context import TableInfo


class TableRegistry:
    """
    Table metadata and full-data access.

    Manages:
    - tables: Dict[str, TableInfo]
    - raw_data: Dict[str, Dict[str, Any]]
    - data_directory: Optional[str]
    """

    def __init__(self) -> None:
        # These references are set by SharedContext.__post_init__ to point
        # at the *same* objects that live on the dataclass instance.
        self.tables: Dict[str, "TableInfo"] = {}
        self.raw_data: Dict[str, Dict[str, Any]] = {}
        self.data_directory: Optional[str] = None
        # v28.6: 전체 데이터 캐시 — CSV를 한 번만 읽고 재사용
        self._full_data_cache: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Bind helpers (called once from SharedContext.__post_init__)
    # ------------------------------------------------------------------

    def bind(
        self,
        tables: Dict[str, "TableInfo"],
        raw_data: Dict[str, Dict[str, Any]],
        data_directory: Optional[str],
    ) -> None:
        """Bind to the dataclass-owned containers so mutations are shared."""
        self.tables = tables
        self.raw_data = raw_data
        self.data_directory = data_directory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_table(self, table_info: "TableInfo") -> None:
        self.tables[table_info.name] = table_info

    def get_data_directory(self) -> Optional[str]:
        return self.data_directory

    def set_data_directory(self, path: str) -> None:
        self.data_directory = path

    def get_full_data(self, table_name: str) -> List[Dict[str, Any]]:
        """
        v22.1: Return full (un-sampled) data for *table_name*.

        Priority:
        1) source CSV on disk
        2) data_directory / <table>.csv
        3) sample_data already stored in TableInfo

        v28.6: 결과를 인스턴스 캐시에 보관 — 같은 파이프라인 실행 중 CSV 중복 로딩 방지
        """
        # v28.6: 캐시 히트 → 즉시 반환
        if table_name in self._full_data_cache:
            return self._full_data_cache[table_name]

        import os

        table_info = self.tables.get(table_name)
        if not table_info:
            return []

        result = []

        # 1st: source path CSV
        source_path = getattr(table_info, "source", None) or ""
        if source_path and os.path.exists(source_path) and source_path.endswith(".csv"):
            try:
                import pandas as pd
                df = pd.read_csv(source_path)
                result = df.where(pd.notnull(df), None).to_dict("records")
            except Exception:
                pass

        # 2nd: data_directory + table_name.csv
        if not result and self.data_directory:
            for suffix in [".csv", "_utf8.csv"]:
                path = os.path.join(self.data_directory, f"{table_name}{suffix}")
                if os.path.exists(path):
                    try:
                        import pandas as pd
                        df = pd.read_csv(path)
                        result = df.where(pd.notnull(df), None).to_dict("records")
                        break
                    except Exception:
                        pass

        # 3rd: sample_data as-is
        if not result:
            result = table_info.sample_data or []

        # v28.6: 캐시 저장
        self._full_data_cache[table_name] = result
        return result

    def get_analysis_data(self, table_name: str, max_rows: int = 5000) -> List[Dict[str, Any]]:
        """
        v28.6: Business Insights / 통계 분석용 stratified sample 반환.

        전체 행이 max_rows 이하이면 전체 반환 (q2cut 1813행 → 변화 없음).
        초과하면 카테고리 비율 보존 stratified sample 반환.
        """
        full_data = self.get_full_data(table_name)
        if len(full_data) <= max_rows:
            return full_data

        try:
            import pandas as pd
            df = pd.DataFrame(full_data)

            # 가장 낮은 카디널리티(2-50) 카테고리 컬럼 기준 stratified sample
            cat_col = None
            for col in df.columns:
                n_unique = df[col].nunique()
                if 2 <= n_unique <= 50 and df[col].count() > len(df) * 0.5:
                    cat_col = col
                    break

            if cat_col:
                sampled = (
                    df.groupby(cat_col, group_keys=False)
                    .apply(lambda g: g.sample(
                        min(len(g), max(5, int(max_rows * len(g) / len(df)))),
                        random_state=42,
                    ))
                ).head(max_rows)
            else:
                sampled = df.sample(max_rows, random_state=42)

            return sampled.where(pd.notnull(sampled), None).to_dict("records")
        except Exception:
            return full_data[:max_rows]

    def get_all_analysis_data(self, max_rows: int = 5000) -> Dict[str, List[Dict[str, Any]]]:
        """v28.6: 모든 테이블의 분석용 샘플 데이터 반환."""
        return {
            table_name: self.get_analysis_data(table_name, max_rows)
            for table_name in self.tables
            if self.get_analysis_data(table_name, max_rows)
        }

    def get_all_full_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """v22.1: Full data for every table."""
        result: Dict[str, List[Dict[str, Any]]] = {}
        for table_name in self.tables:
            data = self.get_full_data(table_name)
            if data:
                result[table_name] = data
        return result
