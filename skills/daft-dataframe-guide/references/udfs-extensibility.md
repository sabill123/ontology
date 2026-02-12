# Daft UDFs & Extensibility

## Overview

Daft supports User-Defined Functions (UDFs) for custom logic that can't be expressed with built-in operations. UDFs run within the Daft execution engine and scale automatically.

## Basic UDFs

```python
import daft

@daft.udf(return_dtype=daft.DataType.string())
def categorize_age(age):
    """Categorize ages into groups."""
    return ["child" if a < 13 else "teen" if a < 20 else "adult" for a in age.to_pylist()]

df = daft.from_pydict({"name": ["Alice", "Bob", "Charlie"], "age": [10, 17, 35]})
df = df.with_column("age_group", categorize_age(daft.col("age")))
df.show()
```

## Multi-Column UDFs

```python
@daft.udf(return_dtype=daft.DataType.string())
def full_name(first, last):
    firsts = first.to_pylist()
    lasts = last.to_pylist()
    return [f"{f} {l}" for f, l in zip(firsts, lasts)]

df = df.with_column("full_name", full_name(daft.col("first_name"), daft.col("last_name")))
```

## Class-based UDFs (with state)

```python
@daft.udf(return_dtype=daft.DataType.float64())
class ModelPredictor:
    """UDF that loads a model once and runs inference on each batch."""

    def __init__(self):
        import torch
        self.model = torch.load("model.pt")
        self.model.eval()

    def __call__(self, features):
        import torch
        tensor = torch.tensor(features.to_pylist())
        with torch.no_grad():
            predictions = self.model(tensor)
        return predictions.numpy().tolist()

df = df.with_column("prediction", ModelPredictor(daft.col("features")))
```

## Map Partitions

```python
# Apply function to entire partition
@daft.udf(return_dtype=daft.DataType.python())
def process_partition(col_a, col_b):
    # Process entire partition at once
    a_list = col_a.to_pylist()
    b_list = col_b.to_pylist()
    return [a + b for a, b in zip(a_list, b_list)]
```

## Map Groups

```python
# Apply function to grouped data
result = df.groupby("category").map_groups(
    lambda group_df: group_df.with_column(
        "rank",
        daft.col("score").rank()
    )
)
```

## Best Practices

1. **Prefer built-in operations** over UDFs when possible - they run in Rust and are much faster
2. **Batch process** in UDFs - work with the entire column (Series) at once, not row by row
3. **Use class-based UDFs** when you need expensive initialization (model loading, DB connections)
4. **Return correct types** - always specify `return_dtype` accurately
5. **Handle nulls** - check for None values in input data
6. **Minimize Python overhead** - use numpy/pandas operations inside UDFs where possible
