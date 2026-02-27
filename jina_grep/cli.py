"""CLI interface for jina-grep."""

import sys
from pathlib import Path

import click

from .client import SearchOptions, pipe_rerank, semantic_grep, semantic_classify
from .server import server_status, start_server, stop_server


@click.group()
def serve():
    """Manage the embedding server."""
    pass


@serve.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", type=int, default=8089, help="Port to bind to")
@click.option("--foreground", is_flag=True, help="Run in foreground")
def start(host, port, foreground):
    """Start the embedding server."""
    start_server(host=host, port=port, daemon=not foreground)


@serve.command()
def stop():
    """Stop the embedding server."""
    stop_server()


@serve.command()
def status():
    """Check embedding server status."""
    server_status()


def main():
    """Entry point: routes to 'serve' subcommand or grep mode."""
    args = sys.argv[1:]

    if args and args[0] == "serve":
        serve(args[1:], standalone_mode=True)
        return

    grep_main(args)


def grep_main(args=None):
    """Grep CLI."""

    @click.command()
    @click.option("-r", "-R", "--recursive", is_flag=True, help="Recursive search")
    @click.option("-l", "--files-with-matches", is_flag=True, help="Print only filenames with matches")
    @click.option("-L", "--files-without-match", is_flag=True, help="Print only filenames without matches")
    @click.option("-c", "--count", is_flag=True, help="Print match count per file")
    @click.option("-n", "--line-number", is_flag=True, default=True, help="Print line numbers")
    @click.option("-H", "--with-filename", is_flag=True, default=None, help="Print filename with matches")
    @click.option("--no-filename", is_flag=True, help="Suppress filename")
    @click.option("-A", "--after-context", type=int, default=0, help="Lines after match")
    @click.option("-B", "--before-context", type=int, default=0, help="Lines before match")
    @click.option("-C", "--context", type=int, default=0, help="Lines before and after match")
    @click.option("--include", multiple=True, help="Search only files matching GLOB")
    @click.option("--exclude", multiple=True, help="Skip files matching GLOB")
    @click.option("--exclude-dir", multiple=True, help="Skip directories matching GLOB")
    @click.option("--color", type=click.Choice(["never", "always", "auto"]), default="auto")
    @click.option("-v", "--invert-match", is_flag=True, help="Invert match (lowest similarity)")
    @click.option("-m", "--max-count", type=int, help="Max matches per file")
    @click.option("-q", "--quiet", is_flag=True, help="Quiet mode")
    @click.option("-o", "--only-matching", is_flag=True, help="Print only matching label (classification mode)")
    @click.option("-e", "--regexp", multiple=True, help="Classification labels (multiple -e for zero-shot classification)")
    @click.option("-f", "--file", "label_file", type=click.Path(exists=True), help="Read classification labels from file (one per line)")
    @click.option("--threshold", type=float, default=0.5, help="Similarity threshold (default: 0.5)")
    @click.option("--top-k", type=int, default=10, help="Max results (default: 10)")
    @click.option("--model", default="jina-embeddings-v5-nano", help="Model name")
    @click.option("--task", type=click.Choice(["retrieval", "text-matching", "clustering", "classification", "nl2code", "qa", "code2code", "code2nl", "code2completion"]), default=None, help="Embedding task (auto-detected)")
    @click.option("--server", default="http://localhost:8089", help="Server URL")
    @click.option("--granularity", type=click.Choice(["line", "paragraph", "sentence", "token"]), default="token")
    @click.argument("pattern", required=False, default=None)
    @click.argument("files", nargs=-1, type=click.Path())
    def _grep(
        recursive, files_with_matches, files_without_match, count, line_number,
        with_filename, no_filename, after_context, before_context, context,
        include, exclude, exclude_dir, color, invert_match, max_count, quiet,
        only_matching, regexp, label_file, threshold, top_k, model, task,
        server, granularity, pattern, files,
    ):
        """Semantic grep using Jina embeddings.

        PATTERN is a natural language query (not regex).

        \b
        Pipe mode (rerank grep output):
            grep -rn "error" src/ | jina-grep "error handling"

        \b
        Standalone mode (direct semantic search):
            jina-grep "error handling" src/
            jina-grep -r --threshold=0.6 "database connection" .

        \b
        Zero-shot classification:
            jina-grep -e "bug" -e "feature" -e "docs" src/
            jina-grep -f labels.txt src/

        \b
        Server:
            jina-grep serve start    Start embedding server
            jina-grep serve stop     Stop embedding server
            jina-grep serve status   Check server status
        """
        # Collect labels from -e and -f
        labels = list(regexp)
        if label_file:
            with open(label_file) as lf:
                labels.extend(line.strip() for line in lf if line.strip())

        # Classification mode: multiple labels provided
        if labels:
            if not files and pattern:
                # pattern is actually a file path
                files = (pattern,) + files if pattern else files
                if pattern:
                    files = (pattern,)
            elif not files:
                files = (".",)

            use_color = color == "always" or (color == "auto" and sys.stdout.isatty())
            show_fn = not no_filename and (with_filename or len(files) > 1 or recursive)

            classify_options = SearchOptions(
                recursive=recursive,
                files_with_matches=files_with_matches,
                files_without_match=files_without_match,
                count=count,
                line_number=line_number,
                with_filename=show_fn,
                after_context=after_context if context == 0 else context,
                before_context=before_context if context == 0 else context,
                include_patterns=list(include),
                exclude_patterns=list(exclude),
                exclude_dir_patterns=list(exclude_dir),
                color=use_color,
                invert_match=invert_match,
                max_count=max_count,
                quiet=quiet,
                threshold=threshold,
                top_k=top_k,
                model=model,
                task="classification",
                server_url=server,
                granularity=granularity,
            )

            paths = []
            for f in files:
                p = Path(f)
                if not p.exists():
                    print(f"jina-grep: {f}: No such file or directory", file=sys.stderr)
                    continue
                paths.append(p)

            if not paths:
                sys.exit(2)

            exit_code = semantic_classify(labels, paths, classify_options, only_matching=only_matching)
            sys.exit(exit_code)

        # Standard grep mode requires a pattern
        if not pattern:
            print("Error: PATTERN is required (or use -e for classification)", file=sys.stderr)
            sys.exit(2)

        if no_filename:
            show_filename = False
        elif with_filename is not None:
            show_filename = with_filename
        else:
            show_filename = len(files) > 1 or recursive

        if context > 0:
            after_context = context
            before_context = context

        use_color = color == "always" or (color == "auto" and sys.stdout.isatty())

        options = SearchOptions(
            recursive=recursive,
            files_with_matches=files_with_matches,
            files_without_match=files_without_match,
            count=count,
            line_number=line_number,
            with_filename=show_filename,
            after_context=after_context,
            before_context=before_context,
            include_patterns=list(include),
            exclude_patterns=list(exclude),
            exclude_dir_patterns=list(exclude_dir),
            color=use_color,
            invert_match=invert_match,
            max_count=max_count,
            quiet=quiet,
            threshold=threshold,
            top_k=top_k,
            model=model,
            task=task or "retrieval",
            server_url=server,
            granularity=granularity,
        )

        # Pipe mode: stdin has data and is not a TTY
        # Also check if files were provided -- if so, prefer standalone mode
        if not sys.stdin.isatty() and not files:
            import select
            if select.select([sys.stdin], [], [], 0.0)[0]:
                exit_code = pipe_rerank(pattern, options)
                sys.exit(exit_code)

        # Standalone mode: need files
        if not files:
            files = (".",)

        paths = []
        for f in files:
            p = Path(f)
            if not p.exists():
                print(f"jina-grep: {f}: No such file or directory", file=sys.stderr)
                continue
            paths.append(p)

        if not paths:
            sys.exit(2)

        exit_code = semantic_grep(pattern, paths, options)
        sys.exit(exit_code)

    _grep(args, standalone_mode=True)


if __name__ == "__main__":
    main()
