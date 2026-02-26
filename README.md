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

## Benchmark (M3 Ultra, pure MLX, optimized model.py)

Uses `mx.fast.rope` and `mx.fast.scaled_dot_product_attention` for Metal-accelerated inference.
Verified numerically identical to original (MSE < 2e-9, cosine > 0.9999).

### v5-small (677M params, 1024 dims)

```
Config                    Batch ~Tokens   Avg ms   P50 ms      Tok/s
---------------------------------------------------------------------------
1x short                      1       9      7.3      7.3       1228
1x medium                     1     117     12.1     12.1       9633
1x long (~520 tok)            1     624     33.2     32.4      18790
1x very long (~2.6K tok)      1    2470    164.1    164.0      15051
8x short                      8      72     11.7     11.5       6128
32x short                    32     291     23.7     23.7      12279
128x short                  128    1164     70.1     70.6      16614
256x short                  256    2329    132.2    133.2      17614
8x long                       8    4992    197.3    197.1      25298
32x long                     32   19968    810.7    774.0      24631
```

Single query: **7ms**. Peak throughput: **25.3K tok/s**. (~1.4x faster than unoptimized)

### v5-nano (239M params, 768 dims)

```
Config                    Batch ~Tokens   Avg ms   P50 ms      Tok/s
---------------------------------------------------------------------------
1x short                      1       9      2.9      2.8       3145
1x medium                     1     117      4.4      4.4      26353
1x long (~520 tok)            1     624     10.1     10.1      61660
1x very long (~2.6K tok)      1    2470     41.0     40.9      60193
8x short                      8      72      5.2      5.1      13836
32x short                    32     291      8.7      8.6      33344
128x short                  128    1164     22.6     22.3      51399
256x short                  256    2329     41.7     42.0      55829
8x long                       8    4992     52.2     51.7      95668
32x long                     32   19968    202.2    201.7      98740
```

Single query: **2.9ms**. Peak throughput: **98.7K tok/s**.

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
