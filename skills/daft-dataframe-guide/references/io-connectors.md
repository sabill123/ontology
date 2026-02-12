# Daft IO Connectors

## Overview

Daft supports reading and writing from many data sources and formats, with native support for cloud storage (S3, GCS, Azure).

## File Formats

### Parquet (Recommended)
```python
# Read
df = daft.read_parquet("data.parquet")
df = daft.read_parquet("data/*.parquet")               # Glob
df = daft.read_parquet("s3://bucket/path/*.parquet")    # S3
df = daft.read_parquet("gs://bucket/path/*.parquet")    # GCS

# Write
df.write_parquet("output/")
df.write_parquet("s3://bucket/output/")
```

### CSV
```python
df = daft.read_csv("data.csv")
df = daft.read_csv("data.csv", delimiter="\t", has_header=True)
df.write_csv("output/")
```

### JSON / NDJSON
```python
df = daft.read_json("data.json")
df = daft.read_json("data.ndjson")
df.write_json("output/")
```

## Cloud Storage

### Amazon S3
```python
# Uses AWS credentials from environment or ~/.aws/
df = daft.read_parquet("s3://my-bucket/data/*.parquet")

# With explicit config
io_config = daft.io.IOConfig(
    s3=daft.io.S3Config(
        region_name="us-east-1",
        access_key_id="...",
        secret_access_key="...",
    )
)
df = daft.read_parquet("s3://bucket/data.parquet", io_config=io_config)
```

### Google Cloud Storage
```python
df = daft.read_parquet("gs://my-bucket/data/*.parquet")
```

### Azure Blob Storage
```python
df = daft.read_parquet("az://container/data/*.parquet")
```

## Table Formats

### Apache Iceberg
```python
# pip install "getdaft[iceberg]"
df = daft.read_iceberg("my_catalog.my_database.my_table")
df.write_iceberg(iceberg_table)
```

### Delta Lake
```python
# pip install "getdaft[deltalake]"
df = daft.read_deltalake("path/to/delta_table")
df = daft.read_deltalake("s3://bucket/delta_table")
df.write_deltalake("path/to/delta_table", mode="append")
```

### Apache Hudi
```python
# pip install "getdaft[hudi]"
df = daft.read_hudi("path/to/hudi_table")
```

## Database Sources

### SQL Databases
```python
# pip install "getdaft[sql]"
df = daft.read_sql(
    "SELECT * FROM users WHERE active = true",
    "postgresql://user:pass@host:5432/db"
)

# Supported: PostgreSQL, MySQL, SQLite, SQL Server, etc.
# Uses ConnectorX for high-performance reads
```

## Special Sources

### Hugging Face Datasets
```python
df = daft.read_huggingface("imdb")
df = daft.read_huggingface("squad", split="train")
```

### Video Frames
```python
df = daft.read_video_frames("video.mp4")
```

## IO Configuration

```python
io_config = daft.io.IOConfig(
    s3=daft.io.S3Config(
        region_name="us-east-1",
        endpoint_url="http://localhost:9000",  # MinIO
    ),
    http=daft.io.HTTPConfig(
        bearer_token="...",
    ),
)

df = daft.read_parquet("s3://bucket/data.parquet", io_config=io_config)
```

## Best Practices

1. **Use Parquet** for intermediate storage - columnar format, efficient compression, predicate pushdown
2. **Use glob patterns** to read multiple files: `data/**/*.parquet`
3. **Use cloud-native paths** (s3://, gs://) - Daft reads directly without downloading
4. **Use predicate pushdown** - filter in SQL or with `.where()` right after read for Parquet
5. **Use Delta/Iceberg** for mutable datasets with ACID transactions
