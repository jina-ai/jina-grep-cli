"""FastAPI server for local embedding generation."""

import os
import signal
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SUPPORTED_MODELS = {
    "jina-embeddings-v5-small": "jinaai/jina-embeddings-v5-text-small-retrieval",
    "jina-embeddings-v5-nano": "jinaai/jina-embeddings-v5-text-nano-retrieval",
}

# v5 retrieval models don't need task routing (single-task checkpoints)
# but keep the mapping for future multi-task model support
TASK_MAPPING = {
    "retrieval.query": "retrieval.query",
    "retrieval.passage": "retrieval.passage",
}

app = FastAPI(title="Jina Grep Embedding Server")

# Global model cache
_models: dict = {}


class EmbeddingRequest(BaseModel):
    input: list[str]
    model: str = "jina-embeddings-v5-small"
    task: str = "retrieval.query"


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


def get_model(model_name: str):
    """Load or retrieve cached model."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name not in _models:
        from sentence_transformers import SentenceTransformer

        hf_name = SUPPORTED_MODELS[model_name]

        # Try MLX backend first, fall back to default
        try:
            model = SentenceTransformer(hf_name, backend="mlx", trust_remote_code=True)
            print(f"Loaded {model_name} with MLX backend")
        except Exception:
            try:
                model = SentenceTransformer(hf_name, trust_remote_code=True)
                print(f"Loaded {model_name} with default backend")
            except Exception as e:
                raise RuntimeError(f"Failed to load model {model_name}: {e}")

        _models[model_name] = model

    return _models[model_name]


def count_tokens(texts: list[str]) -> int:
    """Approximate token count (words * 1.3)."""
    total_words = sum(len(t.split()) for t in texts)
    return int(total_words * 1.3)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Generate embeddings for input texts."""
    if not request.input:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    try:
        model = get_model(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        embeddings = model.encode(request.input, normalize_embeddings=True)
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
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    """List supported models."""
    return {"models": list(SUPPORTED_MODELS.keys())}


def get_pid_file() -> Path:
    """Get path to PID file."""
    pid_dir = Path.home() / ".jina-grep"
    pid_dir.mkdir(exist_ok=True)
    return pid_dir / "server.pid"


def write_pid():
    """Write current PID to file."""
    pid_file = get_pid_file()
    pid_file.write_text(str(os.getpid()))


def read_pid() -> Optional[int]:
    """Read PID from file."""
    pid_file = get_pid_file()
    if pid_file.exists():
        try:
            return int(pid_file.read_text().strip())
        except (ValueError, OSError):
            return None
    return None


def remove_pid():
    """Remove PID file."""
    pid_file = get_pid_file()
    if pid_file.exists():
        pid_file.unlink()


def is_server_running() -> tuple[bool, Optional[int]]:
    """Check if server is running."""
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
    """Start the embedding server."""
    import uvicorn

    running, pid = is_server_running()
    if running:
        print(f"Server already running (PID: {pid})")
        return

    if daemon:
        # Fork to background
        pid = os.fork()
        if pid > 0:
            # Parent process
            print(f"Server starting in background (PID: {pid})")
            return

        # Child process - daemonize
        os.setsid()

        # Redirect stdout/stderr to log file
        log_dir = Path.home() / ".jina-grep"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "server.log"

        sys.stdout = open(log_file, "a")
        sys.stderr = sys.stdout

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
    """Stop the embedding server."""
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
    """Get server status."""
    running, pid = is_server_running()
    if running:
        print(f"Server is running (PID: {pid})")
    else:
        print("Server is not running")
