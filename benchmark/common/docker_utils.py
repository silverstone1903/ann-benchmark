"""Thin helpers to bring one engine's docker-compose stack up/down and wait for it to be ready.

Each engine gets its own compose file (engines/<name>/docker-compose.yml) so runs can be done
one engine at a time and torn down (`down -v`) between runs, instead of one shared stack.
"""

import pathlib
import socket
import subprocess
import time


def compose_up(compose_file: pathlib.Path) -> None:
    subprocess.run(["docker", "compose", "-f", str(compose_file), "up", "-d"], check=True)


def compose_down(compose_file: pathlib.Path) -> None:
    subprocess.run(["docker", "compose", "-f", str(compose_file), "down", "-v"], check=True)


def wait_tcp(host: str, port: int, timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError as e:
            last_err = e
            time.sleep(1)
    raise TimeoutError(f"{host}:{port} not reachable after {timeout_s}s ({last_err})")


def container_dir_size_bytes(container_name: str, path: str) -> int:
    """Bytes on disk under `path` inside a running container, via `du -sb`.

    Used as the index_size_bytes() fallback for engines whose client API doesn't cleanly
    report on-disk size (Qdrant, Weaviate, Chroma) — measuring actual filesystem bytes keeps
    the metric comparable across engines instead of trusting divergent self-reported stats.
    """
    result = subprocess.run(
        ["docker", "exec", container_name, "du", "-sb", path],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.split()[0])
