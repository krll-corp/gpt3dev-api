# GPT3dev OpenAI-Compatible API

A production-ready FastAPI server that mirrors the OpenAI REST API surface while proxying requests to Hugging Face causal language models. The service implements the `/v1/completions`, `/v1/models`, and `/v1/embeddings` endpoints with full support for streaming Server-Sent Events (SSE) and OpenAI-style usage accounting. A `/v1/chat/completions` stub is included but currently returns a structured 501 error because the available models are completion-only.

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
> The requirements file pins the CPU-only PyTorch wheels via the official PyTorch index. This avoids compiling CUDA artifacts
> during deployment (for example on Vercel's 8 GB build workers) and keeps memory usage comfortably within the available limits.

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

The default in-memory registry (see `app/core/model_registry.py`) currently exposes no models. All bundled `ModelSpec`
definitions are commented out so Vercel can build the project without downloading Hugging Face checkpoints while we diagnose
the serverless size regression. Uncomment the relevant entries (for example, the 17M-parameter `GPT3-dev` checkpoint) or supply
a custom registry file once you are ready to serve a model. Use `MODEL_ALLOW_LIST` to restrict a deployment to a smaller subset
of model IDs when re-enabling them.

### Estimating Model Artifact Sizes

The helper script `scripts/model_size_report.py` queries Hugging Face for each registered model and prints the aggregated
artifact sizes. Supplying `--token` is recommended if any repositories are private:

```bash
python scripts/model_size_report.py --token "$HF_TOKEN"
```

This output is useful when planning how to shard heavier models across multiple deployments so that each serverless bundle stays
below the 250 MB uncompressed ceiling enforced by Vercel and other AWS Lambda-based platforms.

### Split Deployments for Large Models

To deploy several large models without breaching the serverless size limit, create separate Vercel projects (or environments)
that point to the same repository but configure distinct `MODEL_ALLOW_LIST` values. Each deployment then packages only the
models it serves while sharing the rest of the codebase. Combine this with the size report above to decide which models belong
on each instance.

### Running the API Server

Launch the FastAPI application with your preferred ASGI server (for example, `uvicorn` or `hypercorn`). A reference Docker workflow is shown below:

```bash
docker build -t gpt3dev-api .
docker run --rm -p 8000:8000 gpt3dev-api
```

## Usage Examples

### List Models

```bash
curl http://localhost:8000/v1/models
```

### Text Completion

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "GPT3-dev",
        "prompt": "Write a haiku about fast APIs",
        "max_tokens": 64
      }'
```

### Chat Completions

The `/v1/chat/completions` endpoint is currently disabled and returns a 501 Not Implemented error instructing clients to use `/v1/completions` instead. This keeps the API surface compatible for future fine-tuned chat models while avoiding confusing responses from the present completion-only models.

### Embeddings

The `/v1/embeddings` endpoint returns a 501 Not Implemented error with actionable guidance unless an embeddings backend is configured.

## Testing

```bash
pytest
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
