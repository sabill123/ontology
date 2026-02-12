# Daft SQL Support

## Overview

Daft supports SQL queries via the `daft.sql()` function and `Session` objects, enabling SQL workflows over DataFrames.

## Basic SQL Queries

```python
import daft

# Register a DataFrame as a SQL table
df = daft.read_parquet("users.parquet")

# Query with SQL (auto-registered from variable name)
result = daft.sql("SELECT name, age FROM df WHERE age > 30")
result.show()
```

## Session-based SQL

```python
from daft.session import Session

session = Session()

# Create temp tables
users = daft.read_parquet("users.parquet")
orders = daft.read_parquet("orders.parquet")

session.create_temp_table("users", users)
session.create_temp_table("orders", orders)

# Run queries
result = session.sql("""
    SELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as revenue
    FROM users u
    JOIN orders o ON u.id = o.user_id
    WHERE u.status = 'active'
    GROUP BY u.name
    ORDER BY revenue DESC
    LIMIT 10
""")

result.show()
```

## Catalog Integration

```python
from daft.session import Session

session = Session()

# Unity Catalog
session.register_catalog("unity", unity_catalog)

# Iceberg
session.register_catalog("iceberg", iceberg_catalog)

# Query catalog tables
result = session.sql("SELECT * FROM iceberg.db.my_table WHERE date > '2024-01-01'")
```

## Supported SQL Features

- `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`, `LIMIT`
- `JOIN` (INNER, LEFT, RIGHT, OUTER, SEMI, ANTI)
- `UNION`, `INTERSECT`, `EXCEPT`
- Subqueries
- Window functions (`OVER`, `PARTITION BY`, `ORDER BY`)
- Common Table Expressions (CTEs)
- Aggregate functions: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`

## Mixing SQL and DataFrame API

```python
# Start with SQL, continue with DataFrame API
df = daft.sql("SELECT * FROM raw_data WHERE status = 'active'")
df = df.with_column("revenue_usd", daft.col("revenue") * daft.col("exchange_rate"))
df = df.groupby("region").sum("revenue_usd")
result = df.collect()
```
