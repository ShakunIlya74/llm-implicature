import subprocess
import time
from contextlib import contextmanager
from typing import Optional

import requests
from langchain_openai import ChatOpenAI


def start_vllm_server(
    # model: str = "Qwen/Qwen2.5-3B-Instruct",
    model: str = "Qwen/Qwen3-VL-4B-Instruct",
    host: str = "0.0.0.0",
    port: int = 8000,
    api_key: str = "token-abc123",
    dtype: str = "auto",
    extra_args: Optional[list[str]] = None,
) -> subprocess.Popen:
    """
    Start the vLLM OpenAI-compatible API server as a subprocess.

    Returns the Popen object so you can terminate it later.
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

    print("Starting vLLM server with command:")
    print(" ".join(cmd))

    # Inherit stdout/stderr; change to subprocess.PIPE if you want to capture.
    proc = subprocess.Popen(cmd)
    return proc


def wait_for_vllm(
    base_url: str,
    api_key: str,
    timeout: float = 360.0,
    interval: float = 1.0,
) -> None:
    """
    Poll the vLLM OpenAI-compatible server until it's ready or timeout.

    Uses the /v1/models endpoint to check readiness.
    """
    url = base_url.rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    start = time.time()
    while True:
        try:
            r = requests.get(url, headers=headers, timeout=3)
            if r.status_code == 200:
                print("vLLM server is up.")
                return
        except requests.RequestException:
            pass

        if time.time() - start > timeout:
            raise TimeoutError(
                f"vLLM server at {base_url} did not become ready within {timeout} seconds"
            )

        time.sleep(interval)


def get_local_qwen_llm(
    # model: str = "Qwen/Qwen2.5-3B-Instruct",
    model: str = "Qwen/Qwen3-VL-4B-Instruct",
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str = "token-abc123",
    temperature: float = 0,
    max_tokens: int = 256,
) -> ChatOpenAI:
    """
    Return a LangChain ChatOpenAI instance pointing to a local vLLM server.

    Assumes the server is already running and serving OpenAI-compatible API at /v1.
    """
    base_url = f"http://{host}:{port}/v1"

    llm = ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return llm


@contextmanager
def local_qwen_llm(
    # model: str = "Qwen/Qwen2.5-3B-Instruct",
    model: str = "Qwen/Qwen3-VL-4B-Instruct",
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str = "token-abc123",
    dtype: str = "auto",
    temperature: float = 0.2,
    max_tokens: int = 256,
    extra_args: Optional[list[str]] = None,
):
    """
    Context manager that:

    - starts a vLLM server for Qwen
    - waits until it's ready
    - yields a LangChain ChatOpenAI instance
    - shuts down the server on exit
    """
    if extra_args is None:
        extra_args = []

    # Start server
    proc = start_vllm_server(
        model=model,
        host=host,
        port=port,
        api_key=api_key,
        dtype=dtype,
        extra_args=extra_args,
    )

    base_url = f"http://{host}:{port}/v1"

    try:
        # Wait until server is ready
        wait_for_vllm(base_url=base_url[:-3], api_key=api_key)  # strip /v1

        # Build LangChain client
        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        yield llm
    finally:
        print("Shutting down vLLM server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()