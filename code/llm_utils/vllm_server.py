"""vLLM OpenAI-compatible server lifecycle management.
Manages starting, health-checking, and stopping local vLLM servers.
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional

import requests


def start_vllm_server(
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    api_key: str = "token-abc123",
    dtype: str = "auto",
    extra_args: Optional[list[str]] = None,
) -> subprocess.Popen:
    """Start the vLLM OpenAI-compatible API server as a subprocess.

    Returns the Popen handle so the caller can manage its lifecycle.
    """
    if extra_args is None:
        extra_args = []

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        dtype,
        "--api-key",
        api_key,
        *extra_args,
    ]

    print(f"[vllm] Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    return proc


def wait_for_vllm(
    base_url: str,
    api_key: str,
    timeout: float = 360.0,
    interval: float = 2.0,
) -> None:
    """Poll the vLLM server at /v1/models until it responds or timeout."""
    url = base_url.rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    start = time.time()
    while True:
        try:
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                print("[vllm] Server is ready.")
                return
        except requests.RequestException:
            pass

        if time.time() - start > timeout:
            raise TimeoutError(
                f"vLLM server at {base_url} not ready after {timeout}s"
            )
        time.sleep(interval)


def stop_vllm_server(proc: subprocess.Popen, timeout: float = 15.0) -> None:
    """Terminate vLLM server process gracefully, force-kill if needed."""
    if proc.poll() is not None:
        return
    print("[vllm] Stopping server...")
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("[vllm] Force killing server...")
        proc.kill()
        proc.wait(timeout=5)
