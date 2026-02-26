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

## Benchmark (M3 Ultra, pure MLX)

### v5-small (677M params, 1024 dims)

```
Config                    Batch ~Tokens   Avg ms   P50 ms      Tok/s
---------------------------------------------------------------------------
1x short                      1       9     17.5     17.5        514
1x medium                     1     117     21.6     21.5       5416
1x long (~520 tok)            1     624     47.2     47.0      13231
1x very long (~2.6K tok)      1    2470    275.2    274.3       8975
8x short                      8      72     21.8     21.8       3296
32x short                    32     291     36.0     35.7       8084
128x short                  128    1164     98.3     97.6      11843
256x short                  256    2329    180.9    181.0      12872
8x long                       8    4992    265.3    264.3      18814
32x long                     32   19968   1113.0   1101.9      17941
```

Single query: ~18ms. Peak throughput: **18.8K tok/s**.

### v5-nano (239M params, 768 dims)

```
Config                    Batch ~Tokens   Avg ms   P50 ms      Tok/s
---------------------------------------------------------------------------
1x short                      1       9      3.3      3.2       2730
1x medium                     1     117      4.7      4.6      25134
1x long (~520 tok)            1     624     10.3     10.2      60564
1x very long (~2.6K tok)      1    2470     42.2     42.2      58549
8x short                      8      72      6.1      6.1      11763
32x short                    32     291     11.5     11.0      25381
128x short                  128    1164     29.5     29.5      39491
256x short                  256    2329     54.4     54.1      42808
8x long                       8    4992     53.4     53.4      93442
32x long                     32   19968    200.9    201.1      99410
```

Single query: **3.3ms**. Peak throughput: **99.4K tok/s**.

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
