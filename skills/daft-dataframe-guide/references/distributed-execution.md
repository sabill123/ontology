# Daft Distributed Execution

## Overview

Daft supports two execution backends: Native (local, multi-threaded) and Ray (distributed across clusters). The same code runs on both backends.

## Native Runner (Default)

```python
import daft

# Explicitly set (or use default)
daft.set_runner_native()

# Configure execution
daft.set_execution_config(
    num_partitions=8,           # Number of partitions
    scan_tasks_min_size_bytes=64 * 1024 * 1024,  # Min partition size
)

df = daft.read_parquet("data/*.parquet")
result = df.groupby("category").sum("revenue").collect()
```

## Ray Runner (Distributed)

```python
import daft

# Switch to Ray backend
daft.set_runner_ray()

# Or with Ray address
daft.set_runner_ray(address="ray://cluster:10001")

# Same code as before - now runs distributed
df = daft.read_parquet("s3://bucket/data/*.parquet")
result = df.groupby("category").sum("revenue").collect()
```

### Ray Setup

```bash
# Install with Ray support
pip install getdaft[ray]

# Start local Ray cluster
ray start --head

# Or connect to existing cluster
export RAY_ADDRESS="ray://cluster:10001"
```

## Execution Configuration

```python
# Global config
daft.set_execution_config(
    num_partitions=16,
    csv_inflation_factor=10.0,
    parquet_inflation_factor=3.0,
)

# Context manager (temporary config)
with daft.execution_config_ctx(num_partitions=32):
    result = df.collect()

# Planning config
daft.set_planning_config(
    enable_pushdowns=True,      # Enable predicate pushdowns
)

with daft.planning_config_ctx(enable_io_fusion=True):
    df = daft.read_parquet("data.parquet")
```

## Partitioning

```python
# Repartition data
df = df.repartition(16)                        # Random repartition
df = df.repartition(16, daft.col("category"))  # Hash repartition by column

# View partition count
df.num_partitions()

# Into partitions (returns list of DataFrames)
partitions = df.into_partitions(4)
```

## Scaling Patterns

### Small Data (< 10GB)
```python
daft.set_runner_native()  # Default, single machine
df = daft.read_parquet("data.parquet")
result = df.collect()
```

### Medium Data (10-100GB)
```python
daft.set_runner_native()
daft.set_execution_config(num_partitions=16)
df = daft.read_parquet("data/*.parquet")
result = df.collect()  # Out-of-core processing handles memory
```

### Large Data (100GB+)
```python
daft.set_runner_ray()
df = daft.read_parquet("s3://bucket/data/**/*.parquet")
result = df.groupby("key").sum("value").collect()
```

## Best Practices

1. **Start with Native runner** - only move to Ray when data exceeds single-machine memory
2. **Partition by access patterns** - repartition by frequently-grouped columns
3. **Use predicate pushdowns** - filter early, especially when reading Parquet
4. **Monitor with `.explain()`** - inspect query plans to verify optimizations
5. **Control partition size** - aim for 64-256MB per partition for optimal throughput
