"""FastAPI server for local embedding generation using pure MLX."""

import os
import signal
import sys
from pathlib import Path
from typing import Optional

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
    # Code models: single checkpoint, task handled via instruction prefix
    "jina-code-embeddings-0.5b": {
        "_all": "jinaai/jina-code-embeddings-0.5b-mlx",
    },
    "jina-code-embeddings-1.5b": {
        "_all": "jinaai/jina-code-embeddings-1.5b-mlx",
    },
}

# Code model names for dispatch
CODE_MODELS = {"jina-code-embeddings-0.5b", "jina-code-embeddings-1.5b"}

# Supported Matryoshka dimensions
MATRYOSHKA_DIMS = {32, 64, 128, 256, 512, 768, 1024}

VALID_TASKS = {"retrieval", "text-matching", "clustering", "classification"}
CODE_TASKS = {"nl2code", "qa", "code2code", "code2nl", "code2completion"}
ALL_TASKS = VALID_TASKS | CODE_TASKS
VALID_PROMPT_NAMES = {"query", "document", "passage"}

# Guardrails
MAX_BATCH_SIZE = 512
MAX_SEQ_LENGTH = {
    "jina-embeddings-v5-small": 32768,
    "jina-embeddings-v5-nano": 8192,
}

app = FastAPI(title="Jina Grep Embedding Server")

# Global model cache: key = (model_name, task) -> (model, tokenizer)
_models: dict = {}


class EmbeddingRequest(BaseModel):
    input: list[str]
    model: str = "jina-embeddings-v5-small"
    task: str = "retrieval"
    prompt_name: Optional[str] = "query"
    truncate_dim: Optional[int] = None


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
    """Load or retrieve cached MLX model and tokenizer for given task."""
    if model_name not in MLX_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {', '.join(MLX_MODELS)}")

    is_code = model_name in CODE_MODELS
    if is_code:
        if task not in CODE_TASKS:
            raise ValueError(f"Unsupported task for code model: {task}. Supported: {', '.join(CODE_TASKS)}")
    else:
        if task not in VALID_TASKS:
            raise ValueError(f"Unsupported task: {task}. Supported: {', '.join(VALID_TASKS)}")

    # Code models use single checkpoint for all tasks
    cache_key = (model_name, "_all") if is_code else (model_name, task)
    if cache_key not in _models:
        import importlib.util
        import json

        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        hf_name = MLX_MODELS[model_name]["_all"] if is_code else MLX_MODELS[model_name][task]

        # Download all model files
        model_dir = snapshot_download(hf_name)
        print(f"Downloaded {hf_name} to {model_dir}")

        # Load config
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)

        # Import model.py from the downloaded repo
        spec = importlib.util.spec_from_file_location(
            f"jina_mlx_model_{model_name}",
            os.path.join(model_dir, "model.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Create model and load weights
        model_class = mod.JinaCodeEmbeddingModel if is_code else mod.JinaEmbeddingModel
        model = model_class(config)
        weights = mx.load(os.path.join(model_dir, "model.safetensors"))
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        print(f"Loaded {hf_name} with pure MLX")

        # Load tokenizer
        tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))

        _models[cache_key] = (model, tokenizer)

    return _models[cache_key]


def count_tokens(texts: list[str]) -> int:
    """Approximate token count."""
    return int(sum(len(t.split()) for t in texts) * 1.3)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Generate embeddings for input texts."""
    if not request.input:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    if len(request.input) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.input)} exceeds maximum {MAX_BATCH_SIZE}",
        )

    task = request.task
    is_code = request.model in CODE_MODELS
    valid = CODE_TASKS if is_code else VALID_TASKS
    if task not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid task: {task}. Must be one of: {', '.join(sorted(valid))}")

    if request.truncate_dim is not None and request.truncate_dim not in MATRYOSHKA_DIMS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid truncate_dim: {request.truncate_dim}. Must be one of: {sorted(MATRYOSHKA_DIMS)}",
        )

    try:
        model, tokenizer = get_model(request.model, task)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        import mlx.core as mx

        if is_code:
            # Code models: task + prompt_type
            prompt_type = request.prompt_name or "query"
            if prompt_type == "document":
                prompt_type = "passage"
            embeddings = model.encode(
                request.input,
                tokenizer,
                task=task,
                prompt_type=prompt_type,
                truncate_dim=request.truncate_dim,
            )
        else:
            # v5 models: task_type string
            if task == "retrieval":
                prompt_name = request.prompt_name or "query"
                if prompt_name not in VALID_PROMPT_NAMES:
                    raise HTTPException(status_code=400, detail=f"Invalid prompt_name: {prompt_name}")
                task_type = f"retrieval.{prompt_name}" if prompt_name == "query" else "retrieval.passage"
            else:
                task_type = task
            embeddings = model.encode(
                request.input,
                tokenizer,
                task_type=task_type,
                truncate_dim=request.truncate_dim,
            )
        mx.eval(embeddings)
        embeddings = embeddings.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")

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
    return {
        "models": {name: list(tasks.keys()) for name, tasks in MLX_MODELS.items()},
        "matryoshka_dims": sorted(MATRYOSHKA_DIMS),
        "max_seq_length": MAX_SEQ_LENGTH,
    }


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

        with open(log_file, "a") as lf:
            proc = subprocess.Popen(
                [sys.executable, "-m", "jina_grep.server", "--host", host, "--port", str(port)],
                stdout=lf,
                stderr=lf,
                start_new_session=True,
            )
        print(f"Server starting in background (PID: {proc.pid})")
        return

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
