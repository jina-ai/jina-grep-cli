# jina-grep

Semantic grep powered by Jina embeddings v5 running locally on Apple Silicon (MLX).

## Install

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

## Usage

Start the local embedding server:

```bash
jina-grep serve start
```

Search:

```bash
jina-grep "error handling" src/
jina-grep -r --threshold=0.6 "database connection" .
jina-grep --top-k=5 -n "authentication logic" *.py
jina-grep -l "memory leak" --include="*.c" -r .
```

Stop the server:

```bash
jina-grep serve stop
```

## Server

```
jina-grep serve start [--port 8089] [--host 127.0.0.1] [--foreground]
jina-grep serve stop
jina-grep serve status
```

Exposes `POST /v1/embeddings` (Jina/OpenAI-compatible format). Models are downloaded on first run.

## Options

```
-r, -R          Recursive search
-l              Print only filenames with matches
-L              Print only filenames without matches
-c              Print match count per file
-n              Print line numbers (default: on)
-H / -h         Show / hide filename
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
--server        Server URL (default: http://localhost:8089)
--granularity   line/paragraph/sentence (default: line)
```

Regex flags (`-E`, `-F`, `-G`, `-P`, `-w`, `-x`) are not supported -- semantic search operates on meaning, not patterns.

## Models

- `jina-embeddings-v5-small` (default)
- `jina-embeddings-v5-nano`
