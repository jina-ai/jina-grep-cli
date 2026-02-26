"""FastAPI server for local embedding generation using MLX."""

import os
import signal
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# MLX checkpoints per model and task
MLX_MODELS = {
    "jina-embeddings-v5-small": {
        "retrieval": "jinaai/jina-embeddings-v5-text-small-retrieval-mlx",
        "text-matching": "jinaai/jina-embeddings-v5-text-small-text-matching-mlx",
        "clustering": "jinaai/jina-embeddings-v5-text-small-clustering-mlx",
        "classification": "jinaai/jina-embeddings-v5-text-small-classification-mlx",
    },
    "jina-embeddings-v5-nano": {
        "retrieval": "jinaai/jina-embeddings-v5-text-nano-retrieval-mlx",
        "text-matching": "jinaai/jina-embeddings-v5-text-nano-text-matching-mlx",
        "clustering": "jinaai/jina-embeddings-v5-text-nano-clustering-mlx",
        "classification": "jinaai/jina-embeddings-v5-text-nano-classification-mlx",
    },
}

VALID_TASKS = {"retrieval", "text-matching", "clustering", "classification"}
VALID_PROMPT_NAMES = {"query", "document"}  # only for retrieval task

app = FastAPI(title="Jina Grep Embedding Server")

# Global model cache: key = (model_name, task)
_models: dict = {}


class EmbeddingRequest(BaseModel):
    input: list[str]
    model: str = "jina-embeddings-v5-small"
    task: str = "retrieval"
    prompt_name: Optional[str] = "query"  # query or document, only for retrieval


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class UsageInfo(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: UsageInfo


def get_model(model_name: str, task: str):
    """Load or retrieve cached model for given task."""
    if model_name not in MLX_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {', '.join(MLX_MODELS)}")
    if task not in VALID_TASKS:
        raise ValueError(f"Unsupported task: {task}. Supported: {', '.join(VALID_TASKS)}")

    cache_key = (model_name, task)
    if cache_key not in _models:
        from sentence_transformers import SentenceTransformer

        hf_name = MLX_MODELS[model_name][task]

        # Try MLX backend first (Apple Silicon), fall back to default
        try:
            model = SentenceTransformer(hf_name, backend="mlx", trust_remote_code=True)
            print(f"Loaded {hf_name} with MLX backend")
        except Exception:
            try:
                model = SentenceTransformer(hf_name, trust_remote_code=True)
                print(f"Loaded {hf_name} with default backend")
            except Exception as e:
                raise RuntimeError(f"Failed to load {hf_name}: {e}")

        _models[cache_key] = model

    return _models[cache_key]


def count_tokens(texts: list[str]) -> int:
    """Approximate token count."""
    return int(sum(len(t.split()) for t in texts) * 1.3)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Generate embeddings for input texts."""
    if not request.input:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    task = request.task
    if task not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task: {task}. Must be one of: {', '.join(VALID_TASKS)}")

    try:
        model = get_model(request.model, task)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # For retrieval task, use prompt_name (query vs document)
    encode_kwargs = {"normalize_embeddings": True}
    if task == "retrieval" and request.prompt_name:
        if request.prompt_name not in VALID_PROMPT_NAMES:
            raise HTTPException(status_code=400, detail=f"Invalid prompt_name: {request.prompt_name}")
        encode_kwargs["prompt_name"] = request.prompt_name

    try:
        embeddings = model.encode(request.input, **encode_kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")

    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()

    data = [
        EmbeddingData(index=i, embedding=emb)
        for i, emb in enumerate(embeddings)
    ]

    token_count = count_tokens(request.input)

    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage=UsageInfo(prompt_tokens=token_count, total_tokens=token_count),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    return {"models": {name: list(tasks.keys()) for name, tasks in MLX_MODELS.items()}}


# --- PID management ---

def get_pid_file() -> Path:
    pid_dir = Path.home() / ".jina-grep"
    pid_dir.mkdir(exist_ok=True)
    return pid_dir / "server.pid"


def write_pid():
    get_pid_file().write_text(str(os.getpid()))


def read_pid() -> Optional[int]:
    pid_file = get_pid_file()
    if pid_file.exists():
        try:
            return int(pid_file.read_text().strip())
        except (ValueError, OSError):
            return None
    return None


def remove_pid():
    pid_file = get_pid_file()
    if pid_file.exists():
        pid_file.unlink()


def is_server_running() -> tuple[bool, Optional[int]]:
    pid = read_pid()
    if pid is None:
        return False, None
    try:
        os.kill(pid, 0)
        return True, pid
    except OSError:
        remove_pid()
        return False, None


def start_server(host: str = "127.0.0.1", port: int = 8089, daemon: bool = True):
    running, pid = is_server_running()
    if running:
        print(f"Server already running (PID: {pid})")
        return

    if daemon:
        import subprocess
        log_dir = Path.home() / ".jina-grep"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "server.log"

        # Use subprocess instead of fork (fork breaks Metal/MPS GPU access)
        with open(log_file, "a") as lf:
            proc = subprocess.Popen(
                [sys.executable, "-m", "jina_grep.server", "--host", host, "--port", str(port)],
                stdout=lf,
                stderr=lf,
                start_new_session=True,
            )
        print(f"Server starting in background (PID: {proc.pid})")
        return

    # Foreground mode
    import uvicorn
    write_pid()

    def cleanup(signum, frame):
        remove_pid()
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        remove_pid()


def stop_server():
    running, pid = is_server_running()
    if not running:
        print("Server is not running")
        return
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Server stopped (PID: {pid})")
        remove_pid()
    except OSError as e:
        print(f"Failed to stop server: {e}")
        remove_pid()


def server_status():
    running, pid = is_server_running()
    if running:
        print(f"Server is running (PID: {pid})")
    else:
        print("Server is not running")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8089)
    args = parser.parse_args()

    import uvicorn
    write_pid()

    def cleanup(signum, frame):
        remove_pid()
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        remove_pid()
