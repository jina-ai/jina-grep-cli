"""In-process MLX embedding for serverless mode."""

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import sys
from typing import Optional

import numpy as np

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

# Global model cache: key = (model_name, task) -> (model, tokenizer)
_models: dict = {}

_first_load = True


def get_model(model_name: str, task: str):
    """Load or retrieve cached MLX model and tokenizer for given task."""
    global _first_load

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
        if _first_load:
            print("Loading model...", file=sys.stderr, flush=True)
            _first_load = False

        import importlib.util
        import json

        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        hf_name = MLX_MODELS[model_name]["_all"] if is_code else MLX_MODELS[model_name][task]

        # Download all model files
        model_dir = snapshot_download(hf_name)

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

        # Load tokenizer
        tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))

        _models[cache_key] = (model, tokenizer)

    return _models[cache_key]


class LocalEmbedder:
    """In-process MLX embedding, same interface as EmbeddingClient."""

    def embed(
        self,
        texts: list[str],
        model: str = "jina-embeddings-v5-small",
        task: str = "retrieval",
        prompt_name: str = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Get embeddings for texts using in-process MLX inference."""
        import mlx.core as mx

        model_obj, tokenizer = get_model(model, task)
        is_code = model in CODE_MODELS

        if is_code:
            prompt_type = prompt_name or "query"
            if prompt_type == "document":
                prompt_type = "passage"
            embeddings = model_obj.encode(
                texts,
                tokenizer,
                task=task,
                prompt_type=prompt_type,
            )
        else:
            if task == "retrieval":
                pn = prompt_name or "query"
                if pn not in VALID_PROMPT_NAMES:
                    raise ValueError(f"Invalid prompt_name: {pn}")
                task_type = f"retrieval.{pn}" if pn == "query" else "retrieval.passage"
            else:
                task_type = task
            embeddings = model_obj.encode(
                texts,
                tokenizer,
                task_type=task_type,
            )

        mx.eval(embeddings)
        return np.array(embeddings.tolist())

    def health_check(self) -> bool:
        return True

    def close(self):
        pass
