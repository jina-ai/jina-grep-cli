# <img src="logo.svg?v=2" alt="" width="28" height="28" style="vertical-align: middle;"/> jina-grep

Semantic grep powered by Jina embeddings, running locally on Apple Silicon via MLX.

Four modes: pipe grep output for semantic reranking, search files directly with natural language, zero-shot classification, or code search.


| Model | Params | Dims | Max Seq | Matryoshka | Tasks |
|-------|--------|------|---------|------------|-------|
| jina-embeddings-v5-small | 677M | 1024 | 32768 | 32-1024 | retrieval, text-matching, clustering, classification |
| jina-embeddings-v5-nano | 239M | 768 | 8192 | 32-768 | retrieval, text-matching, clustering, classification |
| jina-code-embeddings-1.5b | 1.54B | 1536 | 32768 | 128-1536 | nl2code, code2code, code2nl, code2completion, qa |
| jina-code-embeddings-0.5b | 0.49B | 896 | 32768 | 64-896 | nl2code, code2code, code2nl, code2completion, qa |

Per-task MLX checkpoints (v5) or single checkpoint with instruction prefixes (code) loaded on demand from HuggingFace. No PyTorch, no transformers - pure MLX on Metal GPU. Server auto-batches large inputs (up to 256 per request).

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

```bash
grep -rn "error" src/ | jina-grep "error handling logic"
grep -rn "def.*test" . | jina-grep "unit tests for authentication"
grep -rn "TODO" . | jina-grep "performance optimization"
```

### Standalone mode: direct semantic search

```bash
jina-grep "memory leak" src/
jina-grep -r --threshold 0.3 "database connection pooling" .
jina-grep --top-k 5 "retry with exponential backoff" *.py
```

### Code search

Use `--model` to switch to code embeddings and `--task` for code-specific tasks:

```bash
# Natural language to code: find code that matches a description
jina-grep --model jina-code-embeddings-1.5b --task nl2code "sort a list in descending order" src/

# Code to code: find similar code snippets
jina-grep --model jina-code-embeddings-0.5b --task code2code "for i in range(len(arr))" src/

# Code to natural language: find comments/docs matching code
jina-grep --model jina-code-embeddings-1.5b --task code2nl "def binary_search(arr, target):" src/

# Pipe mode works too
grep -rn "def " src/ | jina-grep --model jina-code-embeddings-1.5b --task nl2code "HTTP retry with backoff"
```

Code tasks:
- `nl2code` - natural language query to code (default for code models)
- `code2code` - find similar code snippets
- `code2nl` - find comments/docs for code
- `code2completion` - find completions for partial code
- `qa` - question answering over code

### Zero-shot classification

Use `-e` to specify labels. Each line gets classified to the best matching label.

```bash
# Classify code by category
jina-grep -e "database" -e "error handling" -e "data processing" -e "configuration" src/

# Read labels from file
echo -e "bug\nfeature\nrefactor\ndocs" > labels.txt
jina-grep -f labels.txt src/

# Output only the label (pipe-friendly)
jina-grep -o -e "positive" -e "negative" -e "neutral" reviews.txt

# Count per label
jina-grep -c -e "bug" -e "feature" -e "docs" src/
```

Output shows all label scores, best label highlighted:

```
src/main.py:10:def handle_error(error_code, message):  [error handling:0.744 data processing:0.756 ...]
src/config.py:1:# Configuration settings  [configuration:0.210 database:0.217 ...]
```

### Server management

```bash
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
  --task          v5: retrieval/text-matching/clustering/classification
                  code: nl2code/code2code/code2nl/code2completion/qa
  --server        Server URL (default: http://localhost:8089)
  --granularity   line/paragraph/sentence (default: line)
```

## Benchmark (M3 Ultra)

### v5-small (677M, 1024d)

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

Single query: **7ms**. Peak throughput: **25.3K tok/s**.

### v5-nano (239M, 768d)

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

## License

Apache-2.0
