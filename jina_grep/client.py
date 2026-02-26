"""Client for embedding server and semantic search logic."""

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import httpx
import numpy as np


@dataclass
class SearchResult:
    """A single search result."""
    filepath: Path
    line_number: int
    line: str
    score: float
    context_before: list[str]
    context_after: list[str]


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
        task: str = "retrieval.query",
    ) -> np.ndarray:
        """Get embeddings for texts."""
        response = self.client.post(
            f"{self.server_url}/v1/embeddings",
            json={"input": texts, "model": model, "task": task},
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
    # Vectors should already be normalized, but normalize just in case
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)

    if a_norm.ndim == 1:
        a_norm = a_norm.reshape(1, -1)

    return np.dot(a_norm, b_norm.T).flatten()


def split_into_chunks(
    content: str,
    granularity: str = "line",
) -> list[tuple[int, str]]:
    """Split content into chunks with line numbers.

    Returns list of (line_number, chunk_text) tuples.
    Line numbers are 1-indexed.
    """
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
        # Simple sentence splitting
        chunks = []
        current_line = 1

        for i, line in enumerate(lines):
            if not line.strip():
                current_line = i + 2
                continue

            # Split on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", line)
            for sentence in sentences:
                if sentence.strip():
                    chunks.append((i + 1, sentence.strip()))

        return chunks

    else:
        return [(i + 1, line) for i, line in enumerate(lines) if line.strip()]


def should_include_file(filepath: Path, options: SearchOptions) -> bool:
    """Check if file should be included based on patterns."""
    filename = filepath.name

    # Check include patterns
    if options.include_patterns:
        if not any(fnmatch.fnmatch(filename, p) for p in options.include_patterns):
            return False

    # Check exclude patterns
    if options.exclude_patterns:
        if any(fnmatch.fnmatch(filename, p) for p in options.exclude_patterns):
            return False

    return True


def should_exclude_dir(dirpath: Path, options: SearchOptions) -> bool:
    """Check if directory should be excluded."""
    dirname = dirpath.name

    # Always exclude common non-text directories
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
                for item in path.iterdir():
                    if item.is_dir():
                        if not should_exclude_dir(item, options):
                            yield from get_files([item], options)
                    elif item.is_file():
                        if should_include_file(item, options):
                            yield item
            else:
                for item in path.iterdir():
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
    all_lines: list[str],
) -> list[SearchResult]:
    """Search a single file for semantic matches."""
    content = read_file_safely(filepath)
    if content is None:
        return []

    chunks = split_into_chunks(content, options.granularity)
    if not chunks:
        return []

    # Get embeddings for chunks
    chunk_texts = [c[1] for c in chunks]

    try:
        chunk_embeddings = client.embed(
            chunk_texts,
            model=options.model,
            task="retrieval.passage",
        )
    except Exception:
        return []

    # Compute similarities
    similarities = cosine_similarity(query_embedding, chunk_embeddings)

    # Build results
    results = []
    lines = content.splitlines()

    for idx, (line_num, chunk_text) in enumerate(chunks):
        score = float(similarities[idx])

        # Apply threshold (inverted for invert_match)
        if options.invert_match:
            if score >= options.threshold:
                continue
        else:
            if score < options.threshold:
                continue

        # Get context lines
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
                filepath=filepath,
                line_number=line_num,
                line=chunk_text,
                score=score,
                context_before=context_before,
                context_after=context_after,
            )
        )

    # Sort by score
    if options.invert_match:
        results.sort(key=lambda r: r.score)  # Lowest first
    else:
        results.sort(key=lambda r: -r.score)  # Highest first

    # Apply max_count per file
    if options.max_count is not None:
        results = results[: options.max_count]

    return results


def format_result(result: SearchResult, options: SearchOptions) -> str:
    """Format a search result for output."""
    parts = []

    # Filename
    if options.with_filename:
        if options.color:
            parts.append(f"\033[35m{result.filepath}\033[0m")
        else:
            parts.append(str(result.filepath))

    # Line number
    if options.line_number:
        if options.color:
            parts.append(f"\033[32m{result.line_number}\033[0m")
        else:
            parts.append(str(result.line_number))

    # Build the line with separators
    if parts:
        prefix = ":".join(parts) + ":"
    else:
        prefix = ""

    # Main line content
    line = result.line
    if options.color:
        # Highlight the matching line
        line = f"\033[1m{line}\033[0m"

    output_lines = []

    # Context before
    for ctx_line in result.context_before:
        ctx_prefix = prefix.replace(":", "-") if prefix else ""
        output_lines.append(f"{ctx_prefix}{ctx_line}")

    # Main match
    output_lines.append(f"{prefix}{line}")

    # Context after
    for ctx_line in result.context_after:
        ctx_prefix = prefix.replace(":", "-") if prefix else ""
        output_lines.append(f"{ctx_prefix}{ctx_line}")

    return "\n".join(output_lines)


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
                file=__import__("sys").stderr,
            )
            print(
                "Start the server with: jina-grep serve start",
                file=__import__("sys").stderr,
            )
        return 2

    # Get query embedding
    try:
        query_embedding = client.embed([pattern], model=options.model, task="retrieval.query")[0]
    except Exception as e:
        if not options.quiet:
            print(f"Error getting query embedding: {e}", file=__import__("sys").stderr)
        return 2

    # Collect all files
    files = list(get_files(paths, options))
    if not files:
        if not options.quiet:
            print("No files to search", file=__import__("sys").stderr)
        return 1

    # Track results per file for special modes
    file_results: dict[Path, list[SearchResult]] = {}
    all_results: list[SearchResult] = []

    for filepath in files:
        content = read_file_safely(filepath)
        if content is None:
            continue

        results = search_file(
            filepath,
            query_embedding,
            client,
            options,
            content.splitlines(),
        )

        if results:
            file_results[filepath] = results
            all_results.extend(results)

    client.close()

    # Handle special output modes
    if options.quiet:
        return 0 if all_results else 1

    if options.files_with_matches:
        for filepath in file_results:
            print(filepath)
        return 0 if file_results else 1

    if options.files_without_match:
        matched_files = set(file_results.keys())
        for filepath in files:
            if filepath not in matched_files:
                print(filepath)
        return 0

    if options.count:
        for filepath, results in file_results.items():
            if options.with_filename:
                print(f"{filepath}:{len(results)}")
            else:
                print(len(results))
        return 0 if file_results else 1

    # Normal output: sort all results by score and take top_k
    if options.invert_match:
        all_results.sort(key=lambda r: r.score)
    else:
        all_results.sort(key=lambda r: -r.score)

    all_results = all_results[: options.top_k]

    for result in all_results:
        print(format_result(result, options))

    return 0 if all_results else 1
