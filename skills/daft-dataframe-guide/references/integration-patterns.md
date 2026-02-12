# Daft Integration Patterns

## Overview

This guide covers how to integrate Daft into existing Python projects, including the ontoloty pipeline.

## Installation

```bash
# Basic
pip install getdaft

# With Ray support
pip install "getdaft[ray]"

# With all extras
pip install "getdaft[all]"

# Specific extras
pip install "getdaft[iceberg,deltalake,sql]"
```

## Pattern 1: Replace Pandas in Existing Pipelines

```python
# Before (Pandas)
import pandas as pd
df = pd.read_csv("data.csv")
result = df.groupby("category")["revenue"].sum()

# After (Daft) - same result, scales to larger data
import daft
df = daft.read_csv("data.csv")
result = df.groupby("category").sum("revenue").collect()
pandas_result = result.to_pandas()  # Convert back if needed
```

## Pattern 2: Data Processing Service

```python
import daft

class DataProcessor:
    """Wrapper around Daft for project-specific data processing."""

    def __init__(self, use_ray: bool = False):
        if use_ray:
            daft.set_runner_ray()
        else:
            daft.set_runner_native()

    def load_entities(self, path: str) -> daft.DataFrame:
        """Load entity data from parquet files."""
        df = daft.read_parquet(path)
        return df.where(daft.col("status") == "active")

    def compute_relationships(self, entities_df: daft.DataFrame) -> daft.DataFrame:
        """Compute entity relationships."""
        return entities_df.groupby("entity_type").agg(
            daft.col("id").count().alias("count"),
            daft.col("confidence").mean().alias("avg_confidence"),
        )

    def export(self, df: daft.DataFrame, output_path: str):
        """Export results to parquet."""
        df.write_parquet(output_path)
```

## Pattern 3: Ontology Data Pipeline Integration

```python
import daft
from pydantic import BaseModel

class OntologyRelation(BaseModel):
    source: str
    target: str
    relation_type: str
    confidence: float

def process_ontology_data(input_path: str) -> daft.DataFrame:
    """Process ontology data using Daft for large-scale analysis."""

    # Load raw data
    df = daft.read_parquet(input_path)

    # Extract entities
    entities = df.select(
        daft.col("entity_id"),
        daft.col("entity_name"),
        daft.col("entity_type"),
    ).distinct()

    # Use AI for relationship extraction
    df = df.with_column(
        "relations",
        daft.functions.prompt(
            [
                "Extract ontological relationships from this text. "
                "Return source, target, relation_type, and confidence.",
                daft.col("text_content"),
            ],
            return_format=list[OntologyRelation],
            model="gpt-4o-mini",
            provider="openai",
        )
    )

    return df

def build_knowledge_graph(df: daft.DataFrame) -> dict:
    """Convert Daft DataFrame to knowledge graph structure."""
    result = df.select("entity_id", "relations").collect()

    graph = {"nodes": [], "edges": []}
    for row in result.to_pydict()["entity_id"]:
        graph["nodes"].append({"id": row})

    return graph
```

## Pattern 4: Batch AI Processing

```python
import daft

def batch_embed_documents(documents_path: str, output_path: str):
    """Embed documents at scale using Daft."""

    df = daft.read_parquet(documents_path)

    # Generate embeddings
    df = df.with_column(
        "embedding",
        daft.functions.embed(
            daft.col("content"),
            model="text-embedding-3-small",
            provider="openai",
        )
    )

    # Write results
    df.write_parquet(output_path)

def batch_classify_entities(entities_path: str, categories: list[str]):
    """Classify entities using LLM."""

    df = daft.read_parquet(entities_path)

    df = df.with_column(
        "category",
        daft.functions.classify(
            daft.col("description"),
            categories=categories,
            model="gpt-4o-mini",
            provider="openai",
        )
    )

    return df.collect()
```

## Pattern 5: Interop with Arrow/Pandas Ecosystem

```python
import daft
import pyarrow as pa
import pandas as pd

# From Arrow
arrow_table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
df = daft.from_arrow(arrow_table)

# Process with Daft
df = df.with_column("x_squared", daft.col("x") ** 2)

# Back to Arrow
result_arrow = df.collect().to_arrow()

# To Pandas
result_pandas = df.collect().to_pandas()

# From Pandas
pandas_df = pd.DataFrame({"x": [1, 2, 3]})
df = daft.from_pandas(pandas_df)
```

## Pattern 6: SQL Database Integration

```python
import daft

# Read from SQL database
df = daft.read_sql(
    "SELECT * FROM ontology_entities WHERE domain = 'healthcare'",
    "postgresql://user:pass@host:5432/dbname"
)

# Process
df = df.with_column(
    "enriched",
    daft.functions.prompt(
        "Enrich this entity description: ",
        daft.col("description"),
        model="gpt-4o-mini",
        provider="openai"
    )
)

# Write back to different format
df.write_parquet("enriched_entities/")
```

## Key Integration Points

| Entry Point | Use Case |
|-------------|----------|
| `daft.read_parquet()` | Load from data lake |
| `daft.read_sql()` | Load from databases |
| `daft.from_pandas()` | Convert existing Pandas code |
| `daft.from_arrow()` | Arrow ecosystem interop |
| `daft.from_pydict()` | From Python dictionaries |
| `.to_pandas()` | Export to Pandas for visualization |
| `.to_arrow()` | Export to Arrow for other tools |
| `.write_parquet()` | Persist to data lake |

## Dependencies to Add

```toml
# pyproject.toml
[project]
dependencies = [
    "getdaft>=0.4.0",
]

[project.optional-dependencies]
distributed = ["getdaft[ray]>=0.4.0"]
ai = ["getdaft>=0.4.0", "openai"]
```
