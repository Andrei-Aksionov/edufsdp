import os
import socket
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys

import pytest


def get_free_port() -> int:
    """Finds and returns an available port allocated by the OS kernel."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "torchrun(nproc=2, timeout=30): marks test to be executed via torch.distributed.run"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    marker = item.get_closest_marker("torchrun")

    if not marker:
        return None

    # PATH A: Parent Pytest Process (Not inside torchrun yet)
    if os.environ.get("IN_TORCHRUN_WORKER") != "1":
        nproc = marker.kwargs.get("nproc", 2)
        timeout = marker.kwargs.get("timeout", 30)

        # Allow explicit port override from marker; otherwise dynamically assign a free port
        port = marker.kwargs.get("port") or get_free_port()

        test_node = item.nodeid

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            f"--nproc-per-node={nproc}",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            "-m",
            "pytest",
            test_node,
            "-s",
            "-o",
            "addopts=",
        ]

        env = os.environ.copy()
        env["IN_TORCHRUN_WORKER"] = "1"

        try:
            result = subprocess.run(cmd, env=env, timeout=timeout)

            if result.returncode != 0:
                pytest.fail(
                    f"\n❌ [torchrun failed with exit code {result.returncode} on port {port} for node {test_node}]",
                    pytrace=False,
                )

        except subprocess.TimeoutExpired:
            pytest.fail(
                f"\n⏳ [DEADLOCK / TIMEOUT DETECTED]: torchrun exceeded {timeout}s limit for test {test_node}!",
                pytrace=False,
            )

        return True

    # PATH B: Inside torchrun Worker Process
    return None
