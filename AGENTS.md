# jina-grep: Semantic File Search

Local semantic grep using Jina embedding models on MLX. No API key needed.

## grep vs rg vs jina-grep

| Tool | How it matches | Speed | Best for |
|------|---------------|-------|----------|
| `grep` | Exact text / regex | Instant | Known patterns |
| `rg` (ripgrep) | Exact text / regex, faster | Instant | Known patterns, large codebases |
| `jina-grep` | Embedding similarity | ~5-10s | "I know the concept but not the keyword" |

Use `grep`/`rg` when you know what to search. Use `jina-grep` when you don't.

Example: searching for "WebSocket channel plugin implementation" finds `ChannelPlugin` type definitions, `connectionMode: websocket` configs, and plugin registry code - none of which contain the literal query string.

## Usage

```bash
# Semantic search (serverless, auto-start/stop)
jina-grep -r --top-k 5 --threshold 0.3 "search query" src/

# Code search
jina-grep -r --model jina-code-embeddings-0.5b --task nl2code "print hello world" src/

# Pipe: pre-filter with rg, rerank semantically
rg -n "channel" src/ | jina-grep "WebSocket plugin"

# Persistent server (for batch usage)
jina-grep serve start
jina-grep -r "query" src/    # ~10ms per query
jina-grep serve stop
```

## Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--top-k` | 10 | Max results |
| `--threshold` | 0.5 | Min similarity (0.3 for exploration, 0.5+ for precision) |
| `--granularity` | token | line / paragraph / sentence / token |
| `--model` | jina-embeddings-v5-small | Embedding model |
| `--task` | retrieval | retrieval / text-matching / nl2code / code2code |
| `-r` | off | Recursive directory search |
| `--include` | all | File glob (e.g. `--include="*.py"`) |

## Models

| Model | Best for |
|-------|----------|
| `jina-embeddings-v5-small` | General text (default) |
| `jina-code-embeddings-0.5b` | Code search (fast) |
| `jina-code-embeddings-1.5b` | Code search (quality) |
