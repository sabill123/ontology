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
        """
        import os

        table_info = self.tables.get(table_name)
        if not table_info:
            return []

        # 1st: source path CSV
        source_path = getattr(table_info, "source", None) or ""
        if source_path and os.path.exists(source_path) and source_path.endswith(".csv"):
            try:
                import pandas as pd
                df = pd.read_csv(source_path)
                return df.where(pd.notnull(df), None).to_dict("records")
            except Exception:
                pass

        # 2nd: data_directory + table_name.csv
        if self.data_directory:
            for suffix in [".csv", "_utf8.csv"]:
                path = os.path.join(self.data_directory, f"{table_name}{suffix}")
                if os.path.exists(path):
                    try:
                        import pandas as pd
                        df = pd.read_csv(path)
                        return df.where(pd.notnull(df), None).to_dict("records")
                    except Exception:
                        pass

        # 3rd: sample_data as-is
        return table_info.sample_data or []

    def get_all_full_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """v22.1: Full data for every table."""
        result: Dict[str, List[Dict[str, Any]]] = {}
        for table_name in self.tables:
            data = self.get_full_data(table_name)
            if data:
                result[table_name] = data
        return result
