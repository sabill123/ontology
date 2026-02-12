# Daft AI Functions

## Overview

Daft provides built-in AI functions for LLM prompting, embedding generation, and classification directly within DataFrame pipelines. These scale automatically across partitions.

## LLM Prompting

```python
import daft
from pydantic import BaseModel

# Simple text prompt
df = df.with_column(
    "summary",
    daft.functions.prompt(
        "Summarize this text: ",
        daft.col("text"),
        model="gpt-4o-mini",
        provider="openai"
    )
)

# Structured output with Pydantic
class ProductInfo(BaseModel):
    category: str
    sentiment: str
    key_features: list[str]

df = df.with_column(
    "analysis",
    daft.functions.prompt(
        ["Analyze this product review:", daft.col("review_text")],
        return_format=ProductInfo,
        model="gpt-4o-mini",
        provider="openai"
    )
)

# Multimodal prompt (text + image)
df = df.with_column(
    "description",
    daft.functions.prompt(
        ["Describe this image:", daft.col("image")],
        model="gpt-4o",
        provider="openai"
    )
)
```

## Embeddings

```python
# Text embeddings
df = df.with_column(
    "embedding",
    daft.functions.embed(
        daft.col("text"),
        model="text-embedding-3-small",
        provider="openai"
    )
)

# Image embeddings
df = df.with_column(
    "img_embedding",
    daft.functions.embed(
        daft.col("image"),
        model="clip-vit-base",
        provider="huggingface"
    )
)
```

## Classification

```python
df = df.with_column(
    "category",
    daft.functions.classify(
        daft.col("text"),
        categories=["positive", "negative", "neutral"],
        model="gpt-4o-mini",
        provider="openai"
    )
)

# Image classification
df = df.with_column(
    "object_type",
    daft.functions.classify(
        daft.col("image"),
        categories=["cat", "dog", "bird", "other"],
        model="gpt-4o-mini",
        provider="openai"
    )
)
```

## Supported Providers

| Provider | Models | Setup |
|----------|--------|-------|
| `openai` | gpt-4o, gpt-4o-mini, text-embedding-3-* | `OPENAI_API_KEY` env var |
| `google_genai` | gemini-2.0-flash, etc. | `GOOGLE_API_KEY` env var |
| `huggingface` | sentence-transformers, CLIP, etc. | Local or HF Hub |
| `vllm` | Any vLLM-served model | vLLM server URL |
| `lm_studio` | Any LM Studio model | Local LM Studio |

## Session-based Provider Config

```python
from daft.session import Session

session = Session()

# Register a provider
session.register_provider(
    name="my_openai",
    provider="openai",
    model="gpt-4o-mini",
    api_key="sk-..."
)

# Use registered provider
df = df.with_column(
    "result",
    daft.functions.prompt(
        "Analyze: ", daft.col("text"),
        provider="my_openai"
    )
)
```

## Best Practices

1. **Use structured outputs** (Pydantic models) for reliable parsing of LLM responses
2. **Start with smaller models** (gpt-4o-mini) for cost efficiency, upgrade if quality insufficient
3. **Batch processing**: Daft automatically batches API calls across partitions
4. **Error handling**: LLM calls can fail - consider wrapping in UDFs with retry logic for production
5. **Rate limits**: Configure partition counts to control concurrent API calls
6. **Cost awareness**: Each row = one API call. Filter data before AI operations to reduce cost
