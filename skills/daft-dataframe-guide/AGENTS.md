# Daft DataFrame Engine - Complete Guide

Daft is a high-performance, Python-native distributed data engine built on Rust for processing multimodal data (images, audio, video, text) at scale. Source: https://github.com/Eventual-Inc/Daft

---

## Quick Start

```python
import daft

# Read data
df = daft.read_parquet("data.parquet")

# Transform
df = df.where(daft.col("status") == "active")
df = df.with_column("revenue_usd", daft.col("revenue") * daft.col("rate"))

# Aggregate
result = df.groupby("category").agg(
    daft.col("revenue_usd").sum().alias("total"),
    daft.col("id").count().alias("count"),
)

# Execute (lazy until this point)
result.show()
```

---

## 1. DataFrame API

### Creating DataFrames

```python
df = daft.from_pydict({"name": ["Alice", "Bob"], "age": [30, 25]})
df = daft.from_pylist([{"name": "Alice", "age": 30}])
df = daft.from_pandas(pd_df)
df = daft.from_arrow(pa_table)
df = daft.range(1000)
```

### Reading Data

```python
df = daft.read_csv("data.csv")
df = daft.read_parquet("data.parquet")
df = daft.read_parquet("s3://bucket/*.parquet")
df = daft.read_json("data.json")
df = daft.read_sql("SELECT * FROM table", db_url)
df = daft.read_iceberg("catalog.db.table")
df = daft.read_deltalake("path/to/delta")
df = daft.read_huggingface("dataset_name")
```

### Transformations

```python
df.select("col1", "col2")
df.with_column("new", daft.col("x") + 1)
df.with_columns({"a": expr1, "b": expr2})
df.where(daft.col("age") > 18)
df.filter(condition)
df.sort("col", desc=True)
df.limit(100)
df.distinct()
df.sample(fraction=0.1)
df.exclude("col")
df.rename({"old": "new"})
```

### Aggregations

```python
df.groupby("cat").sum("val")
df.groupby("cat").agg(
    daft.col("val").sum().alias("total"),
    daft.col("val").mean().alias("avg"),
    daft.col("id").count().alias("n"),
)
```

### Joins & Set Ops

```python
df1.join(df2, on="key")
df1.join(df2, left_on="a", right_on="b", how="left")  # inner/left/right/outer/semi/anti
df1.union(df2)
df1.intersect(df2)
```

### Window Functions

```python
from daft import Window
window = Window.partition_by("cat").order_by("date")
df.with_column("running_sum", daft.col("val").sum().over(window))
```

### Execution

```python
result = df.collect()      # Materialize
df.show(10)                # Display
df.explain()               # Query plan
df.to_pandas()             # -> Pandas
df.to_arrow()              # -> PyArrow
df.to_pydict()             # -> dict
```

### Writing

```python
df.write_parquet("out/")
df.write_csv("out/")
df.write_deltalake("delta/", mode="append")
df.write_iceberg(table)
```

---

## 2. Expressions

```python
from daft import col, lit

col("price") * col("qty")
col("price") * lit(1.1)
col("name").str.upper()
col("email").str.contains("@gmail")
col("val").is_null()
col("val").fill_null(0)
col("age").cast(daft.DataType.float64())
```

---

## 3. AI Functions

### LLM Prompting

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    category: str
    sentiment: str

df = df.with_column("result",
    daft.functions.prompt(
        ["Analyze:", daft.col("text")],
        return_format=Analysis,
        model="gpt-4o-mini",
        provider="openai"
    )
)
```

### Embeddings

```python
df = df.with_column("emb",
    daft.functions.embed(daft.col("text"), model="text-embedding-3-small", provider="openai")
)
```

### Classification

```python
df = df.with_column("cat",
    daft.functions.classify(daft.col("text"), categories=["pos","neg","neutral"], model="gpt-4o-mini", provider="openai")
)
```

### Providers: openai, google_genai, huggingface, vllm, lm_studio

---

## 4. Multimodal Processing

```python
# Download + decode images
df = df.with_column("bytes", daft.functions.download(df["url"], on_error="null"))
df = df.with_column("img", daft.functions.decode_image(df["bytes"], on_error="null"))
df = df.with_column("resized", daft.col("img").image.resize(224, 224))

# Video frames
df = daft.read_video_frames("video.mp4")
```

---

## 5. Distributed Execution

```python
# Local (default)
daft.set_runner_native()

# Distributed via Ray
daft.set_runner_ray()
daft.set_runner_ray(address="ray://cluster:10001")

# Config
daft.set_execution_config(num_partitions=16)
with daft.execution_config_ctx(num_partitions=32):
    result = df.collect()
```

---

## 6. SQL Support

```python
from daft.session import Session

session = Session()
session.create_temp_table("users", users_df)
result = session.sql("SELECT name, COUNT(*) FROM users GROUP BY name")
result.show()
```

---

## 7. UDFs

```python
@daft.udf(return_dtype=daft.DataType.string())
def categorize(age):
    return ["adult" if a >= 18 else "minor" for a in age.to_pylist()]

df = df.with_column("group", categorize(daft.col("age")))
```

Class-based UDF for expensive init (model loading):

```python
@daft.udf(return_dtype=daft.DataType.float64())
class Predictor:
    def __init__(self):
        self.model = load_model()
    def __call__(self, features):
        return self.model.predict(features.to_pylist())
```

---

## 8. IO Connectors

| Format | Read | Write |
|--------|------|-------|
| Parquet | `read_parquet()` | `write_parquet()` |
| CSV | `read_csv()` | `write_csv()` |
| JSON | `read_json()` | `write_json()` |
| SQL | `read_sql()` | - |
| Iceberg | `read_iceberg()` | `write_iceberg()` |
| Delta Lake | `read_deltalake()` | `write_deltalake()` |
| Hudi | `read_hudi()` | - |
| Hugging Face | `read_huggingface()` | - |
| Video | `read_video_frames()` | - |

Cloud: `s3://`, `gs://`, `az://` supported natively.

---

## 9. Integration with ontoloty

### Install

```bash
pip install getdaft
```

### Use in Pipeline

```python
import daft

# Load ontology entities
entities = daft.read_parquet("data/entities/*.parquet")

# AI-powered enrichment
entities = entities.with_column("embedding",
    daft.functions.embed(daft.col("description"), model="text-embedding-3-small", provider="openai")
)

# Relationship extraction via LLM
entities = entities.with_column("relations",
    daft.functions.prompt(
        ["Extract relationships:", daft.col("text")],
        model="gpt-4o-mini", provider="openai"
    )
)

# Aggregate by domain
summary = entities.groupby("domain").agg(
    daft.col("id").count().alias("entity_count"),
    daft.col("confidence").mean().alias("avg_confidence"),
)

# Export
summary.write_parquet("output/domain_summary/")
```

### Convert from/to Pandas

```python
pandas_df = daft_df.collect().to_pandas()
daft_df = daft.from_pandas(pandas_df)
```

---

## Dependencies

```
getdaft>=0.4.0          # Core
getdaft[ray]            # + Ray distributed
getdaft[sql]            # + SQL databases
getdaft[iceberg]        # + Apache Iceberg
getdaft[deltalake]      # + Delta Lake
getdaft[all]            # Everything
```

## Source Code

Local clone: `src/tools/Daft/`
