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

- `jina-embeddings-v5-small` (default) - 677M params, 1024 dims
- `jina-embeddings-v5-nano` - 239M params, 768 dims

Each model has per-task MLX checkpoints (retrieval, text-matching, clustering, classification) that are loaded on demand.

## Benchmark (M3 Ultra, MLX)

```
Config                    Batch ~Tokens   Avg ms   P50 ms      Tok/s
---------------------------------------------------------------------------
1x short (~8 tok)             1       9     17.7     17.3        508
1x medium (~130 tok)          1     117     28.2     23.9       4156
1x long (~520 tok)            1     520     60.4     50.2       8605
8x short (~64 tok)            8      72     47.1     23.9       1530
32x short (~256 tok)         32     291     53.8     42.2       5406
128x short (~1K tok)        128    1164    160.0    159.2       7275
```

Single short text: ~18ms latency. Batch throughput peaks at ~8.6K tok/s.
