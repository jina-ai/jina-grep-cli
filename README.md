# jina-grep

Semantic grep powered by Jina embeddings v5 running locally on Apple Silicon (MLX).

Two modes: pipe grep output for semantic reranking, or search files directly.

## Install

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

## Usage

Start the local embedding server (downloads model on first run):

```bash
jina-grep serve start
```

### Pipe mode: rerank grep output

Use grep for text matching, pipe to jina-grep for semantic reranking:

```bash
grep -rn "error" src/ | jina-grep "error handling logic"
grep -rn "def.*test" . | jina-grep "unit tests for authentication"
grep -rn "TODO" . | jina-grep "performance optimization"
```

### Standalone mode: direct semantic search

When you don't have a keyword to grep for:

```bash
jina-grep "memory leak" src/
jina-grep -r --threshold=0.6 "database connection pooling" .
jina-grep --top-k=5 "retry with exponential backoff" *.py
```

### Server management

```
jina-grep serve start [--port 8089] [--host 127.0.0.1] [--foreground]
jina-grep serve stop
jina-grep serve status
```

## Options

```
-r, -R          Recursive search (standalone mode)
-l              Print only filenames with matches
-L              Print only filenames without matches
-c              Print match count per file
-n              Print line numbers (default: on)
-H / --no-filename   Show / hide filename
-A/-B/-C NUM    Context lines after/before/both
--include=GLOB  Search only matching files
--exclude=GLOB  Skip matching files
--exclude-dir   Skip matching directories
--color=WHEN    never/always/auto
-v              Invert match (lowest similarity)
-m NUM          Max matches per file
-q              Quiet mode
--threshold     Cosine similarity threshold (default: 0.5)
--top-k         Max results (default: 10)
--model         Model name (default: jina-embeddings-v5-small)
--task          retrieval/text-matching/clustering/classification (default: retrieval)
--server        Server URL (default: http://localhost:8089)
--granularity   line/paragraph/sentence (default: line)
```

In pipe mode, most file-related flags are ignored since grep handles file traversal.
Regex flags (`-E`, `-F`, `-G`, `-P`, `-w`, `-x`) are not needed: use grep for pattern matching, jina-grep for meaning.

## Models

- `jina-embeddings-v5-small` (default) - 568M params, 1024 dims
- `jina-embeddings-v5-nano` - 207M params, 768 dims
