"""In-process MLX embedding for serverless mode."""

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # suppress tokenizers warning

import sys
from typing import Optional

import numpy as np

# Unified MLX repos with dynamic LoRA adapter switching
MLX_MODELS = {
    "jina-embeddings-v5-small": "jinaai/jina-embeddings-v5-text-small-mlx",
    "jina-embeddings-v5-nano": "jinaai/jina-embeddings-v5-text-nano-mlx",
}

# Code models: single checkpoint, no LoRA switching
CODE_MODELS_MAP = {
    "jina-code-embeddings-0.5b": "jinaai/jina-code-embeddings-0.5b-mlx",
    "jina-code-embeddings-1.5b": "jinaai/jina-code-embeddings-1.5b-mlx",
}

CODE_MODELS = set(CODE_MODELS_MAP.keys())

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

# Global model cache: key = model_name -> JinaMultiTaskModel or (code_model, tokenizer)
_models: dict = {}

_first_load = True


def get_model(model_name: str, task: str):
    """Load or retrieve cached MLX model for the given model name.

    For v5 models, returns a JinaMultiTaskModel with dynamic adapter switching.
    For code models, returns (model, tokenizer) tuple.
    """
    global _first_load

    is_code = model_name in CODE_MODELS

    if not is_code and model_name not in MLX_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {', '.join(list(MLX_MODELS) + list(CODE_MODELS_MAP))}")

    if is_code:
        if task not in CODE_TASKS:
            raise ValueError(f"Unsupported task for code model: {task}. Supported: {', '.join(CODE_TASKS)}")
    else:
        if task not in VALID_TASKS:
            raise ValueError(f"Unsupported task: {task}. Supported: {', '.join(VALID_TASKS)}")

    if model_name not in _models:
        if _first_load:
            print("Loading model...", file=sys.stderr, flush=True)
            _first_load = False

        if is_code:
            import importlib.util
            import json

            import mlx.core as mx
            from huggingface_hub import snapshot_download
            from tokenizers import Tokenizer

            model_dir = snapshot_download(CODE_MODELS_MAP[model_name])

            with open(os.path.join(model_dir, "config.json")) as f:
                config = json.load(f)

            spec = importlib.util.spec_from_file_location(
                f"jina_mlx_model_{model_name}",
                os.path.join(model_dir, "model.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            model = mod.JinaCodeEmbeddingModel(config)
            weights = mx.load(os.path.join(model_dir, "model.safetensors"))
            model.load_weights(list(weights.items()))
            mx.eval(model.parameters())

            tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
            _models[model_name] = (model, tokenizer)
        else:
            # v5 models: use unified repo with dynamic LoRA
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(MLX_MODELS[model_name])

            # Import utils.py from the downloaded repo
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                f"jina_mlx_utils_{model_name}",
                os.path.join(model_dir, "utils.py"),
            )
            utils_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils_mod)

            multi_model = utils_mod.load_model(model_dir)
            _models[model_name] = multi_model

    return _models[model_name]


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

        cached = get_model(model, task)
        is_code = model in CODE_MODELS

        if is_code:
            model_obj, tokenizer = cached
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
            # JinaMultiTaskModel with dynamic LoRA switching
            multi_model = cached
            multi_model.switch_task(task)

            if task == "retrieval":
                pn = prompt_name or "query"
                if pn not in VALID_PROMPT_NAMES:
                    raise ValueError(f"Invalid prompt_name: {pn}")
                task_type = f"retrieval.{pn}" if pn == "query" else "retrieval.passage"
            else:
                task_type = task

            embeddings = multi_model.encode(texts, task_type=task_type)

        mx.eval(embeddings)
        return np.array(embeddings.tolist())

    def health_check(self) -> bool:
        return True

    def close(self):
        pass
