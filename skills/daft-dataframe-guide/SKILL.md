---
name: daft-dataframe-guide
description: Daft distributed DataFrame engine guide. Use this skill when writing data pipelines with Daft, processing multimodal data (images, audio, video), running AI/LLM inference at scale, or integrating Daft into Python projects.
license: Apache-2.0
metadata:
  author: Eventual Inc
  version: "0.5.0"
  organization: Eventual Inc
  date: January 2026
  abstract: Comprehensive guide for Daft - a Python-native distributed DataFrame engine with Rust backend. Covers DataFrame API, multimodal processing (images/audio/video), AI functions (LLM prompting, embeddings, classification), distributed execution via Ray, SQL support, UDFs, and integration patterns for embedding Daft into existing Python data pipelines.
---

# Daft DataFrame Engine Guide

Daft is a high-performance, Python-native distributed data engine built on a Rust backend for processing multimodal data (images, audio, video, text) alongside structured data at scale.

## When to Apply

Reference these guidelines when:
- Building data pipelines that process structured + unstructured data together
- Running batch AI/LLM inference on DataFrames
- Processing images, audio, or video at scale
- Replacing PySpark or Pandas with a faster, Python-native alternative
- Designing distributed data processing workflows (Ray)
- Writing SQL queries over DataFrame tables
- Integrating Daft into an existing Python/ontology project

## Key Capabilities

| Category | Description | Reference |
|----------|-------------|-----------|
| DataFrame API | Core read/transform/write operations | `references/dataframe-api.md` |
| Multimodal Processing | Image, audio, video as first-class types | `references/multimodal-processing.md` |
| AI Functions | LLM prompting, embeddings, classification | `references/ai-functions.md` |
| Distributed Execution | Ray runner, partitioning, scaling | `references/distributed-execution.md` |
| SQL Support | SQL queries, sessions, catalogs | `references/sql-support.md` |
| UDFs & Extensibility | Custom functions, map_partitions | `references/udfs-extensibility.md` |
| Integration Patterns | Embedding Daft into other projects | `references/integration-patterns.md` |
| IO Connectors | Parquet, CSV, JSON, Iceberg, Delta, S3 | `references/io-connectors.md` |

## How to Use

Read individual reference files for detailed API examples:

```
references/dataframe-api.md       # Core DataFrame operations
references/ai-functions.md        # LLM/embedding/classification
references/integration-patterns.md # How to embed Daft in your project
```

Each reference file contains:
- Feature overview and when to use it
- Code examples with correct patterns
- Common pitfalls and best practices
- Configuration options

## Full Compiled Document

For the complete guide with all sections: `AGENTS.md`

## References

- https://github.com/Eventual-Inc/Daft
- https://www.getdaft.io/projects/docs/en/stable/
- Source code: `src/tools/Daft/`
