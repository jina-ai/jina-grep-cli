"""Client for embedding server and semantic search logic."""

import fnmatch
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import httpx
import numpy as np


@dataclass
class SearchResult:
    """A single search result."""
    filepath: str
    line_number: int
    line: str
    score: float
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)
    original_prefix: str = ""  # preserve original grep prefix for pipe mode


@dataclass
class SearchOptions:
    """Search configuration options."""
    recursive: bool = False
    files_with_matches: bool = False
    files_without_match: bool = False
    count: bool = False
    line_number: bool = True
    with_filename: bool = True
    after_context: int = 0
    before_context: int = 0
    include_patterns: list[str] = None
    exclude_patterns: list[str] = None
    exclude_dir_patterns: list[str] = None
    color: bool = True
    invert_match: bool = False
    max_count: Optional[int] = None
    quiet: bool = False
    threshold: float = 0.5
    top_k: int = 10
    model: str = "jina-embeddings-v5-small"
    task: str = "retrieval"
    server_url: str = "http://localhost:8089"
    granularity: str = "line"

    def __post_init__(self):
        if self.include_patterns is None:
            self.include_patterns = []
        if self.exclude_patterns is None:
            self.exclude_patterns = []
        if self.exclude_dir_patterns is None:
            self.exclude_dir_patterns = []


class EmbeddingClient:
    """Client for the local embedding server."""

    def __init__(self, server_url: str = "http://localhost:8089"):
        self.server_url = server_url.rstrip("/")
        self.client = httpx.Client(timeout=120.0)

    def embed(
        self,
        texts: list[str],
        model: str = "jina-embeddings-v5-small",
        task: str = "retrieval",
        prompt_name: str = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Get embeddings for texts. Auto-batches large inputs."""
        if len(texts) <= batch_size:
            return self._embed_batch(texts, model, task, prompt_name)

        # Batch large inputs to avoid HTTP payload issues
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = self._embed_batch(batch, model, task, prompt_name)
            all_embeddings.append(embs)
        return np.vstack(all_embeddings)

    def _embed_batch(
        self,
        texts: list[str],
        model: str,
        task: str,
        prompt_name: str = None,
    ) -> np.ndarray:
        """Embed a single batch."""
        payload = {"input": texts, "model": model, "task": task}
        if prompt_name:
            payload["prompt_name"] = prompt_name
        response = self.client.post(
            f"{self.server_url}/v1/embeddings",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = [d["embedding"] for d in data["data"]]
        return np.array(embeddings)

    def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            response = self.client.get(f"{self.server_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        """Close the client."""
        self.client.close()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vectors."""
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)

    if a_norm.ndim == 1:
        a_norm = a_norm.reshape(1, -1)

    return np.dot(a_norm, b_norm.T).flatten()


# ---------------------------------------------------------------------------
# Pipe mode: parse grep output, rerank semantically
# ---------------------------------------------------------------------------

# Match grep output: filename:linenum:content or filename:content or just content
GREP_LINE_RE = re.compile(
    r"^(?:(?P<file>[^:]+):)?(?:(?P<lineno>\d+):)?(?P<content>.*)$"
)

# Strip ANSI escape sequences for embedding
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_grep_line(line: str) -> tuple[str, int, str, str]:
    """Parse a grep output line. Returns (filepath, lineno, content, raw_line)."""
    m = GREP_LINE_RE.match(line)
    if not m:
        return ("", 0, line, line)

    filepath = m.group("file") or ""
    lineno = int(m.group("lineno")) if m.group("lineno") else 0
    content = m.group("content") or ""
    return (filepath, lineno, content, line)


def pipe_rerank(
    pattern: str,
    options: SearchOptions,
) -> int:
    """Read grep output from stdin, rerank by semantic similarity."""
    client = EmbeddingClient(options.server_url)

    if not client.health_check():
        print(
            f"Error: Cannot connect to embedding server at {options.server_url}",
            file=sys.stderr,
        )
        print("Start the server with: jina-grep serve start", file=sys.stderr)
        return 2

    # Read all stdin lines
    raw_lines = []
    for line in sys.stdin:
        line = line.rstrip("\n")
        if line:
            raw_lines.append(line)

    if not raw_lines:
        return 1

    # Parse grep output
    parsed = [parse_grep_line(l) for l in raw_lines]
    contents = [ANSI_RE.sub("", p[2]) for p in parsed]

    # Filter out empty content
    valid = [(i, c) for i, c in enumerate(contents) if c.strip()]
    if not valid:
        return 1

    valid_indices = [v[0] for v in valid]
    valid_contents = [v[1] for v in valid]

    # Embed query
    try:
        query_kwargs = {"task": options.task}
        if options.task == "retrieval":
            query_kwargs["prompt_name"] = "query"
        query_emb = client.embed([pattern], model=options.model, **query_kwargs)[0]
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    # Embed contents (batch)
    try:
        doc_kwargs = {"task": options.task}
        if options.task == "retrieval":
            doc_kwargs["prompt_name"] = "document"
        content_embs = client.embed(valid_contents, model=options.model, **doc_kwargs)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    client.close()

    # Compute similarity
    scores = cosine_similarity(query_emb, content_embs)

    # Build scored results
    scored = list(zip(valid_indices, scores))

    if options.invert_match:
        scored.sort(key=lambda x: x[1])
    else:
        scored.sort(key=lambda x: -x[1])

    # Apply threshold and top_k
    results = []
    for idx, score in scored:
        if not options.invert_match and score < options.threshold:
            continue
        if options.invert_match and score >= options.threshold:
            continue
        results.append((idx, score))
        if len(results) >= options.top_k:
            break

    if options.quiet:
        return 0 if results else 1

    # Output: original grep line + score annotation
    for idx, score in results:
        raw_line = raw_lines[idx]
        if options.color:
            # Append score in dim text
            print(f"{raw_line}  \033[2m[{score:.3f}]\033[0m")
        else:
            print(f"{raw_line}  [{score:.3f}]")

    return 0 if results else 1


# ---------------------------------------------------------------------------
# Standalone mode: read files directly, semantic search
# ---------------------------------------------------------------------------

def split_into_chunks(
    content: str,
    granularity: str = "line",
) -> list[tuple[int, str]]:
    """Split content into chunks with line numbers (1-indexed)."""
    lines = content.splitlines()

    if granularity == "line":
        return [(i + 1, line) for i, line in enumerate(lines) if line.strip()]

    elif granularity == "paragraph":
        chunks = []
        current_chunk = []
        start_line = 1

        for i, line in enumerate(lines):
            if line.strip():
                if not current_chunk:
                    start_line = i + 1
                current_chunk.append(line)
            elif current_chunk:
                chunks.append((start_line, "\n".join(current_chunk)))
                current_chunk = []

        if current_chunk:
            chunks.append((start_line, "\n".join(current_chunk)))

        return chunks

    elif granularity == "sentence":
        chunks = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            # Split on sentence-ending punctuation (English and CJK)
            sentences = re.split(r"(?<=[.!?。！？])\s*", line)
            for sentence in sentences:
                if sentence.strip():
                    chunks.append((i + 1, sentence.strip()))
        return chunks

    else:
        return [(i + 1, line) for i, line in enumerate(lines) if line.strip()]


def should_include_file(filepath: Path, options: SearchOptions) -> bool:
    """Check if file should be included based on patterns."""
    filename = filepath.name

    if options.include_patterns:
        if not any(fnmatch.fnmatch(filename, p) for p in options.include_patterns):
            return False

    if options.exclude_patterns:
        if any(fnmatch.fnmatch(filename, p) for p in options.exclude_patterns):
            return False

    return True


def should_exclude_dir(dirpath: Path, options: SearchOptions) -> bool:
    """Check if directory should be excluded."""
    dirname = dirpath.name

    default_excludes = {".git", ".svn", ".hg", "__pycache__", "node_modules", ".venv", "venv"}
    if dirname in default_excludes:
        return True

    if options.exclude_dir_patterns:
        if any(fnmatch.fnmatch(dirname, p) for p in options.exclude_dir_patterns):
            return True

    return False


def get_files(paths: list[Path], options: SearchOptions) -> Iterator[Path]:
    """Get files to search."""
    for path in paths:
        if path.is_file():
            if should_include_file(path, options):
                yield path
        elif path.is_dir():
            if options.recursive:
                for item in sorted(path.iterdir()):
                    if item.is_dir():
                        if not should_exclude_dir(item, options):
                            yield from get_files([item], options)
                    elif item.is_file():
                        if should_include_file(item, options):
                            yield item
            else:
                for item in sorted(path.iterdir()):
                    if item.is_file() and should_include_file(item, options):
                        yield item


def read_file_safely(filepath: Path) -> Optional[str]:
    """Read file content, handling encoding errors."""
    try:
        return filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return filepath.read_text(encoding="latin-1")
        except Exception:
            return None
    except Exception:
        return None


def search_file(
    filepath: Path,
    query_embedding: np.ndarray,
    client: EmbeddingClient,
    options: SearchOptions,
) -> list[SearchResult]:
    """Search a single file for semantic matches."""
    content = read_file_safely(filepath)
    if content is None:
        return []

    chunks = split_into_chunks(content, options.granularity)
    if not chunks:
        return []

    chunk_texts = [c[1] for c in chunks]

    try:
        doc_kwargs = {"task": options.task}
        if options.task == "retrieval":
            doc_kwargs["prompt_name"] = "document"
        chunk_embeddings = client.embed(
            chunk_texts,
            model=options.model,
            **doc_kwargs,
        )
    except Exception:
        return []

    similarities = cosine_similarity(query_embedding, chunk_embeddings)

    results = []
    lines = content.splitlines()

    for idx, (line_num, chunk_text) in enumerate(chunks):
        score = float(similarities[idx])

        if options.invert_match:
            if score >= options.threshold:
                continue
        else:
            if score < options.threshold:
                continue

        context_before = []
        context_after = []

        if options.before_context > 0:
            start = max(0, line_num - 1 - options.before_context)
            context_before = lines[start : line_num - 1]

        if options.after_context > 0:
            end = min(len(lines), line_num + options.after_context)
            context_after = lines[line_num : end]

        results.append(
            SearchResult(
                filepath=str(filepath),
                line_number=line_num,
                line=chunk_text,
                score=score,
                context_before=context_before,
                context_after=context_after,
            )
        )

    if options.invert_match:
        results.sort(key=lambda r: r.score)
    else:
        results.sort(key=lambda r: -r.score)

    if options.max_count is not None:
        results = results[: options.max_count]

    return results


def format_result(result: SearchResult, options: SearchOptions) -> str:
    """Format a search result for output."""
    parts = []

    if options.with_filename:
        if options.color:
            parts.append(f"\033[35m{result.filepath}\033[0m")
        else:
            parts.append(result.filepath)

    if options.line_number:
        if options.color:
            parts.append(f"\033[32m{result.line_number}\033[0m")
        else:
            parts.append(str(result.line_number))

    if parts:
        prefix = ":".join(parts) + ":"
    else:
        prefix = ""

    line = result.line
    if options.color:
        line = f"\033[1m{line}\033[0m"

    # Score annotation
    if options.color:
        score_str = f"  \033[2m[{result.score:.3f}]\033[0m"
    else:
        score_str = f"  [{result.score:.3f}]"

    output_lines = []

    for ctx_line in result.context_before:
        ctx_prefix = prefix.replace(":", "-") if prefix else ""
        output_lines.append(f"{ctx_prefix}{ctx_line}")

    output_lines.append(f"{prefix}{line}{score_str}")

    for ctx_line in result.context_after:
        ctx_prefix = prefix.replace(":", "-") if prefix else ""
        output_lines.append(f"{ctx_prefix}{ctx_line}")

    return "\n".join(output_lines)


def semantic_classify(
    labels: list[str],
    paths: list[Path],
    options: SearchOptions,
    only_matching: bool = False,
) -> int:
    """Zero-shot classification: assign best label to each line/chunk."""
    client = EmbeddingClient(options.server_url)

    if not client.health_check():
        if not options.quiet:
            print(
                f"Error: Cannot connect to embedding server at {options.server_url}",
                file=sys.stderr,
            )
            print("Start the server with: jina-grep serve start", file=sys.stderr)
        return 2

    # Embed all labels
    try:
        label_embs = client.embed(labels, model=options.model, task="classification")
    except Exception as e:
        if not options.quiet:
            print(f"Error embedding labels: {e}", file=sys.stderr)
        return 2

    files = list(get_files(paths, options))
    if not files:
        if not options.quiet:
            print("No files to search", file=sys.stderr)
        return 1

    has_output = False

    if options.count:
        # Count mode: tally per label
        label_counts = {label: 0 for label in labels}

    for filepath in files:
        content = read_file_safely(filepath)
        if content is None:
            continue

        chunks = split_into_chunks(content, options.granularity)
        if not chunks:
            continue

        chunk_texts = [c[1] for c in chunks]

        try:
            chunk_embs = client.embed(chunk_texts, model=options.model, task="classification")
        except Exception:
            continue

        # Compute similarity: each chunk against all labels
        # chunk_embs: [N, D], label_embs: [L, D]
        chunk_norm = chunk_embs / (np.linalg.norm(chunk_embs, axis=-1, keepdims=True) + 1e-9)
        label_norm = label_embs / (np.linalg.norm(label_embs, axis=-1, keepdims=True) + 1e-9)
        scores_matrix = np.dot(chunk_norm, label_norm.T)  # [N, L]

        for idx, (line_num, chunk_text) in enumerate(chunks):
            best_label_idx = int(np.argmax(scores_matrix[idx]))
            best_score = float(scores_matrix[idx, best_label_idx])
            best_label = labels[best_label_idx]

            if best_score < options.threshold:
                continue

            has_output = True

            if options.quiet:
                continue

            if options.count:
                label_counts[best_label] += 1
                continue

            if only_matching:
                print(best_label)
                continue

            # Format output: filepath:linenum:content [label:score]
            parts = []
            if options.with_filename:
                if options.color:
                    parts.append(f"\033[35m{filepath}\033[0m")
                else:
                    parts.append(str(filepath))
            if options.line_number:
                if options.color:
                    parts.append(f"\033[32m{line_num}\033[0m")
                else:
                    parts.append(str(line_num))

            prefix = ":".join(parts) + ":" if parts else ""
            line_display = f"\033[1m{chunk_text}\033[0m" if options.color else chunk_text

            # Show all label scores
            all_scores = []
            for li, label in enumerate(labels):
                s = float(scores_matrix[idx, li])
                if li == best_label_idx:
                    if options.color:
                        all_scores.append(f"\033[1;33m{label}:{s:.3f}\033[0m")
                    else:
                        all_scores.append(f"{label}:{s:.3f}")
                else:
                    if options.color:
                        all_scores.append(f"\033[2m{label}:{s:.3f}\033[0m")
                    else:
                        all_scores.append(f"{label}:{s:.3f}")

            score_str = "  [" + " ".join(all_scores) + "]"
            print(f"{prefix}{line_display}{score_str}")

    client.close()

    if options.quiet:
        return 0 if has_output else 1

    if options.count:
        for label, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"{label}: {cnt}")
        return 0 if has_output else 1

    return 0 if has_output else 1


def semantic_grep(
    pattern: str,
    paths: list[Path],
    options: SearchOptions,
) -> int:
    """Main semantic grep function. Returns exit code."""
    client = EmbeddingClient(options.server_url)

    if not client.health_check():
        if not options.quiet:
            print(
                f"Error: Cannot connect to embedding server at {options.server_url}",
                file=sys.stderr,
            )
            print(
                "Start the server with: jina-grep serve start",
                file=sys.stderr,
            )
        return 2

    try:
        query_kwargs = {"task": options.task}
        if options.task == "retrieval":
            query_kwargs["prompt_name"] = "query"
        query_embedding = client.embed([pattern], model=options.model, **query_kwargs)[0]
    except Exception as e:
        if not options.quiet:
            print(f"Error getting query embedding: {e}", file=sys.stderr)
        return 2

    files = list(get_files(paths, options))
    if not files:
        if not options.quiet:
            print("No files to search", file=sys.stderr)
        return 1

    file_results: dict[str, list[SearchResult]] = {}
    all_results: list[SearchResult] = []

    for filepath in files:
        results = search_file(filepath, query_embedding, client, options)
        if results:
            file_results[str(filepath)] = results
            all_results.extend(results)

    client.close()

    if options.quiet:
        return 0 if all_results else 1

    if options.files_with_matches:
        for filepath in file_results:
            print(filepath)
        return 0 if file_results else 1

    if options.files_without_match:
        matched_files = set(file_results.keys())
        for filepath in files:
            if str(filepath) not in matched_files:
                print(filepath)
        return 0

    if options.count:
        for filepath, results in file_results.items():
            if options.with_filename:
                print(f"{filepath}:{len(results)}")
            else:
                print(len(results))
        return 0 if file_results else 1

    # Sort all results by score and take top_k
    if options.invert_match:
        all_results.sort(key=lambda r: r.score)
    else:
        all_results.sort(key=lambda r: -r.score)

    all_results = all_results[: options.top_k]

    for result in all_results:
        print(format_result(result, options))

    return 0 if all_results else 1
