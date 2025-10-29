---
title: GPT3dev OpenAI-Compatible API
emoji: "🚀"
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# GPT3dev OpenAI-Compatible API

A production-ready FastAPI server that mirrors the OpenAI REST API surface while proxying requests to Hugging Face causal language models. The service implements the `/v1/completions`, `/v1/models`, and `/v1/embeddings` endpoints with full support for streaming Server-Sent Events (SSE) and OpenAI-style usage accounting. A `/v1/chat/completions` stub is included but currently returns a structured 501 error because the available models are completion-only.

##The API is hosted on HuggingFace Spaces:
```bash
https://k050506koch-gpt3-dev-api.hf.space
```

## Features

- ✅ Drop-in compatible request/response schemas for OpenAI text completions.
- ✅ Streaming responses (`stream=true`) that emit OpenAI-formatted SSE frames ending with `data: [DONE]`.
- ✅ Configurable Hugging Face model registry with lazy loading, shared model cache, and automatic device placement.
- ✅ Prompt token counting via `tiktoken` when available (falls back to Hugging Face tokenizers).
- ✅ Structured OpenAI-style error payloads and health probe endpoint (`/healthz`).
- ✅ Dockerfile and pytest suite for rapid deployment and regression coverage.

## Getting Started

### Prerequisites

- Python 3.11+
- (Optional) `HF_TOKEN` environment variable for private model access.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> [!TIP]
> The project can't be deployed on Vercel (pytorch alone weights more than 250mb)

### Configuration

All configuration is driven via environment variables (see `app/core/settings.py`). Key options include:

| Variable | Description | Default |
| --- | --- | --- |
| `HF_TOKEN` | Hugging Face token for private models | `None` |
| `DEFAULT_DEVICE` | Preferred device (`auto`, `cpu`, `cuda`, `mps`) | `auto` |
| `MAX_CONTEXT_TOKENS` | Fallback max context window per model | `2048` |
| `MODEL_REGISTRY_PATH` | Optional path to JSON/YAML registry file | `None` |
| `MODEL_ALLOW_LIST` | Restrict registry to a comma-separated list of model IDs | `None` |
| `ENABLE_EMBEDDINGS_BACKEND` | Enable embeddings backend (returns 501 when `False`) | `False` |
| `CORS_ALLOW_ORIGINS` | Comma-separated list of allowed origins | empty |

The default in-memory registry (see `app/core/model_registry.py`) is disabled by default so CI/serverless environments do not
download checkpoints unexpectedly. To enable the built‑in registry locally, set `INCLUDE_DEFAULT_MODELS=1`. Alternatively, supply
your own registry via `MODEL_REGISTRY_PATH` (JSON or YAML) and optionally restrict with `MODEL_ALLOW_LIST`.

### Estimating Model Artifact Sizes

The helper script `scripts/model_size_report.py` queries Hugging Face for each registered model and prints the aggregated
artifact sizes. Supplying `--token` is recommended if any repositories are private:

```bash
python scripts/model_size_report.py --token "$HF_TOKEN"
```

This output is useful when planning how to shard heavier models across multiple deployments so that each serverless bundle stays

### Split Deployments for Large Models

To deploy several large models without breaching the serverless size limit, create separate Vercel projects (or environments)
that point to the same repository but configure distinct `MODEL_ALLOW_LIST` values. Each deployment then packages only the
models it serves while sharing the rest of the codebase. Combine this with the size report above to decide which models belong
on each instance.

### Running the API Server

Launch the FastAPI application with your preferred ASGI server (for example, `uvicorn` or `hypercorn`). A reference Docker workflow is shown below:

```bash
docker build -t gpt3dev-api .
docker run --rm -p 7860:7860 gpt3dev-api
```

## Usage Examples

### List Models

```bash
curl http://localhost:7860/v1/models
```

### Text Completion

```bash
curl http://localhost:7860/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "GPT3-dev-350m-2805",
        "prompt": "He is a doctor. His main goal is",
        "max_tokens": 64
      }'
```

### Chat Completions

The `/v1/chat/completions` endpoint is currently disabled and returns a 501 Not Implemented error instructing clients to use `/v1/completions` instead. I don't have any chat-tuned models now, but I plan to enable this endpoint later with openai harmony - tuned models.

### Embeddings

The `/v1/embeddings` endpoint returns a 501 Not Implemented error with actionable guidance unless an embeddings backend is configured.

## Testing

Can be configured with
```bash
export RUN_LIVE_API_TESTS=0 #or 1
```
```bash
pytest
pytest -k pytest -k live_more_models
```

## Project Structure

```
app/
  core/             # Settings, model registry, engine utilities
  routers/          # FastAPI routers for each OpenAI endpoint
  schemas/          # Pydantic request/response models
  main.py           # Application factory
tests/              # Pytest suite
```

## Extending the Model Registry

To override or extend the built-in registry, set `MODEL_REGISTRY_PATH` to a JSON or YAML file containing entries such as:

```json
[
  {
    "name": "GPT3-dev-350m-2805",
    "hf_repo": "k050506koch/GPT3-dev-350m-2805",
    "dtype": "float16",
    "device": "auto",
    "max_context_tokens": 4096
  }
]
```

## License

MIT
