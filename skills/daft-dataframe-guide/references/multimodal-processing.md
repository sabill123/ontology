# Daft Multimodal Processing

## Overview

Daft treats images, audio, and video as first-class data types alongside structured data. This enables processing unstructured media within standard DataFrame pipelines.

## Image Processing

```python
import daft

# Read images from URLs
df = daft.from_pydict({"url": [
    "https://example.com/img1.jpg",
    "https://example.com/img2.jpg",
]})

# Download image bytes
df = df.with_column(
    "image_bytes",
    daft.functions.download(df["url"], on_error="null")
)

# Decode to image type
df = df.with_column(
    "image",
    daft.functions.decode_image(df["image_bytes"], on_error="null")
)

# Image operations
df = df.with_column(
    "resized",
    daft.col("image").image.resize(224, 224)
)
df = df.with_column(
    "cropped",
    daft.col("image").image.crop(0, 0, 100, 100)  # x, y, w, h
)

# Get image metadata
df = df.with_column("width", daft.col("image").image.width())
df = df.with_column("height", daft.col("image").image.height())
df = df.with_column("channels", daft.col("image").image.num_channels())
```

## Video Processing

```python
import daft

# Extract frames from video files
df = daft.read_video_frames("video.mp4")
# Returns DataFrame with frame data

# Process frames as images
df = df.with_column(
    "resized_frame",
    daft.col("frame").image.resize(224, 224)
)
```

## Audio Processing

```python
import daft

# Read audio files
df = daft.from_pydict({
    "audio_path": ["audio1.wav", "audio2.mp3"]
})

# Download and process
df = df.with_column(
    "audio_bytes",
    daft.functions.download(df["audio_path"])
)
```

## File Downloads

```python
# Download any file type
df = df.with_column(
    "content",
    daft.functions.download(
        daft.col("url"),
        on_error="null",       # Return null on failure instead of error
        max_connections=32,    # Concurrent connections
    )
)
```

## Common Patterns

### Image Classification Pipeline

```python
import daft

df = daft.read_parquet("products.parquet")

# Download product images
df = df.with_column(
    "img_data",
    daft.functions.download(df["image_url"], on_error="null")
)

# Decode images
df = df.with_column(
    "image",
    daft.functions.decode_image(df["img_data"], on_error="null")
)

# Filter out failures
df = df.where(daft.col("image").is_not_null())

# Classify with AI
df = df.with_column(
    "category",
    daft.functions.classify(
        df["image"],
        categories=["electronics", "clothing", "food", "furniture"],
        model="gpt-4o-mini",
        provider="openai"
    )
)

result = df.collect()
```

### Batch Image Embedding

```python
df = df.with_column(
    "embedding",
    daft.functions.embed(
        df["image"],
        model="clip-vit-base",
        provider="huggingface"
    )
)
```

## Data Types

| Type | Description | Creation |
|------|-------------|----------|
| `daft.DataType.image()` | Image data (decoded) | `decode_image()` |
| `daft.DataType.binary()` | Raw bytes | `download()` |
| `daft.DataType.embedding(dim)` | Fixed-size float array | `embed()` |

## Best Practices

1. **Always use `on_error="null"`** for downloads and decoding to handle failures gracefully
2. **Filter nulls** after download/decode before further processing
3. **Resize images** before AI inference to reduce memory and cost
4. **Use partitioning** for large media datasets to control memory usage
