# Daft DataFrame API

## Overview

Daft DataFrames are lazy by default. Operations build a logical plan that is optimized and executed only when `.collect()` or `.show()` is called.

## Creating DataFrames

```python
import daft

# From Python data
df = daft.from_pydict({"name": ["Alice", "Bob"], "age": [30, 25]})
df = daft.from_pylist([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])

# From other frameworks
df = daft.from_pandas(pd_dataframe)
df = daft.from_arrow(pa_table)
df = daft.from_ray_dataset(ray_ds)

# Generate ranges
df = daft.range(1000)  # Creates DataFrame with column "id" from 0-999
```

## Reading Data

```python
df = daft.read_csv("data.csv")
df = daft.read_parquet("data.parquet")
df = daft.read_json("data.json")
df = daft.read_parquet("s3://bucket/path/*.parquet")  # Glob patterns supported
df = daft.read_sql("SELECT * FROM users", "postgresql://...")
```

## Column Selection & Transformation

```python
# Select columns
df = df.select("name", "age")
df = df.select(daft.col("name"), daft.col("age") + 1)

# Add/modify columns
df = df.with_column("age_plus_one", daft.col("age") + 1)
df = df.with_columns({
    "upper_name": daft.col("name").str.upper(),
    "is_adult": daft.col("age") >= 18,
})

# Rename columns
df = df.rename({"old_name": "new_name"})

# Drop columns
df = df.exclude("unwanted_col")
```

## Filtering

```python
df = df.where(daft.col("age") > 18)
df = df.filter(daft.col("name").str.contains("Ali"))
df = df.where(
    (daft.col("age") > 18) & (daft.col("status") == "active")
)
```

## Aggregations

```python
# Simple aggregations
df.count()
df.sum("revenue")
df.mean("age")
df.min("price")
df.max("price")

# GroupBy aggregations
df.groupby("category").agg(
    daft.col("revenue").sum().alias("total_revenue"),
    daft.col("orders").count().alias("num_orders"),
    daft.col("price").mean().alias("avg_price"),
)

# Multiple groupby keys
df.groupby("category", "region").sum("revenue")
```

## Joins

```python
# Inner join (default)
result = df1.join(df2, on="user_id")

# Left join
result = df1.join(df2, left_on="id", right_on="user_id", how="left")

# Supported: "inner", "left", "right", "outer", "semi", "anti"
```

## Sorting & Limiting

```python
df = df.sort("age", desc=True)
df = df.sort(daft.col("name").str.lower())
df = df.limit(100)
df = df.distinct()
df = df.sample(fraction=0.1)
```

## Set Operations

```python
df1.union(df2)       # Combine rows (like UNION ALL)
df1.intersect(df2)   # Common rows
df1.except_(df2)     # Rows in df1 but not df2
```

## Window Functions

```python
from daft import Window

window = Window.partition_by("category").order_by("date")

df = df.with_column(
    "running_total",
    daft.col("revenue").sum().over(window)
)

df = df.with_column(
    "rank",
    daft.col("revenue").rank().over(Window.partition_by("category"))
)
```

## Execution

```python
# Trigger execution
result = df.collect()         # Returns materialized DataFrame
df.show(10)                   # Display first 10 rows

# Inspect query plan
df.explain()                  # Show optimized plan
df.explain(show_all=True)     # Show all plan stages

# Convert results
pandas_df = df.to_pandas()
arrow_table = df.to_arrow()
py_dict = df.to_pydict()
```

## Writing Data

```python
df.write_parquet("output/")
df.write_csv("output/")
df.write_json("output/")
df.write_iceberg(table)
df.write_deltalake("delta_table/")
```

## Expressions

Expressions are the building blocks for column operations:

```python
from daft import col, lit

# Arithmetic
col("price") * col("quantity")
col("price") * lit(1.1)  # 10% increase

# String operations
col("name").str.upper()
col("email").str.contains("@gmail")
col("text").str.split(",")

# Null handling
col("value").is_null()
col("value").fill_null(0)
col("value").if_else(col("backup"), condition=col("value").is_null())

# Type casting
col("age").cast(daft.DataType.float64())
col("timestamp").cast(daft.DataType.date())

# Conditional
col("age").apply(lambda x: "adult" if x >= 18 else "minor", return_dtype=daft.DataType.string())
```

## Common Pitfalls

1. **Forgetting `.collect()`**: Operations are lazy - nothing executes until you call `.collect()` or `.show()`
2. **Using Python loops**: Use vectorized expressions instead of iterating rows
3. **Large `.to_pandas()`**: Converting huge DataFrames to Pandas defeats the purpose - filter/aggregate first
