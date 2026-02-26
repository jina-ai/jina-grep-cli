# jina-grep

Semantic grep powered by Jina embeddings v5, running locally on Apple Silicon via MLX.

Two modes: pipe grep output for semantic reranking, or search files directly.

## Install

```bash
git clone https://github.com/jina-ai/jina-grep-cli.git && cd jina-grep-cli
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

Requirements: Python 3.10+, Apple Silicon Mac.

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
jina-grep -r --threshold 0.3 "database connection pooling" .
jina-grep --top-k 5 "retry with exponential backoff" *.py
```

### Server management

```
jina-grep serve start [--port 8089] [--host 127.0.0.1] [--foreground]
jina-grep serve stop
jina-grep serve status
```

## Options

```
Grep-compatible flags:
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

Semantic flags:
  --threshold     Cosine similarity threshold (default: 0.5)
  --top-k         Max results (default: 10)
  --model         Model name (default: jina-embeddings-v5-small)
  --task          retrieval/text-matching/clustering/classification
  --server        Server URL (default: http://localhost:8089)
  --granularity   line/paragraph/sentence (default: line)
```

In pipe mode, file-related flags are ignored since grep handles file traversal.
Regex flags (`-E`, `-F`, `-G`, `-P`, `-w`, `-x`) are not needed: use grep for pattern matching, jina-grep for meaning.

## Models

| Model | Params | Dims | Max Seq Length | Matryoshka |
|-------|--------|------|----------------|------------|
| jina-embeddings-v5-small | 677M | 1024 | 32768 | 32, 64, 128, 256, 512, 768, 1024 |
| jina-embeddings-v5-nano | 239M | 768 | 32768 | 32, 64, 128, 256, 512, 768 |

Each model has per-task MLX checkpoints (retrieval, text-matching, clustering, classification) loaded on demand.

### Quantization

Three weight variants per model, selectable via API `quantization` field:

| Level | v5-small Size | Speed | Quality (vs float32) |
|-------|--------------|-------|---------------------|
| float32 | 2.28 GB | baseline | 1.0000 |
| 8bit | 639 MB | ~1.5-2x | >= 0.9999 |
| 4bit | 355 MB | ~2-3x | >= 0.99 |

## Benchmark (M3 Ultra, v5-small float32)

```
Config                    Batch ~Tokens   Avg ms   P50 ms      Tok/s
---------------------------------------------------------------------------
1x short (~8 tok)             1       9     17.8     17.9        505
1x medium (~130 tok)          1     117     22.3     22.2       5245
1x long (~520 tok)            1     520     45.4     45.6      11464
8x short (~64 tok)            8      72     23.4     23.3       3080
32x short (~256 tok)         32     291     36.8     36.9       7903
128x short (~1K tok)        128    1164     99.7     98.8      11676
```

Single query: ~18ms. Batch throughput: ~11.7K tok/s peak.

## Architecture

```
jina-grep "query" files/  -----> HTTP -----> jina-grep serve (MLX on Metal GPU)
grep ... | jina-grep "query"                   |
                                               v
                                     model.py + safetensors
                                     (pure MLX, no PyTorch)
```

- Server loads MLX checkpoints directly (model.py from HuggingFace repo)
- No PyTorch, no transformers, no sentence-transformers dependency
- Cosine similarity computed with NumPy (server returns L2-normalized embeddings)
- Large inputs auto-batched (256 per request)

## License

Apache-2.0
