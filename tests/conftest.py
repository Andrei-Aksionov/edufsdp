import os
import socket

import pytest
import torch

# Ensures child processes inherit the debugpy agent cleanly
if torch.multiprocessing.get_start_method(allow_none=True) != "spawn":
    torch.multiprocessing.set_start_method("spawn", force=True)


def get_free_port() -> str:
    """Allocates an available free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


# def _dist_worker(
#     rank: int,
#     world_size: int,
#     master_port: str,
#     fn: Callable,
#     args: tuple,
#     kwargs: dict,
# ) -> None:
#     """Worker function executed inside spawned processes."""
#     os.environ["MASTER_ADDR"] = "127.0.0.1"
#     os.environ["MASTER_PORT"] = master_port

#     # Initialize GLOO for CPU-based multi-process testing
#     dist.init_process_group(
#         backend="gloo",
#         rank=rank,
#         world_size=world_size,
#     )
#     try:
#         # Pass rank and world_size into the test if requested by function parameters
#         if "rank" in fn.__code__.co_varnames:
#             kwargs["rank"] = rank
#         if "world_size" in fn.__code__.co_varnames:
#             kwargs["world_size"] = world_size

#         fn(*args, **kwargs)
#     finally:
#         dist.destroy_process_group()


def _spawn_worker(
    rank: int,
    world_size: int,
    port: str,
    test_func,
    test_instance,
    args,
    kwargs,
) -> None:
    """Target function executed by mp.spawn inside each worker process."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    # Explicitly tell setup_method that this is a spawned worker.
    os.environ["PYTEST_TORCH_MP_SPAWN_WORKER"] = "1"

    # Re-bind bound method to the instance inside the worker process
    if test_instance is not None:
        # Run setup_method manually inside child process if present
        test_instance.setup_method(test_func)

        try:
            test_func(test_instance, *args, **kwargs)
        finally:
            test_instance.teardown_method(test_func)
    else:
        test_func(*args, **kwargs)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "multi_rank(world_size=2): run test across multiple CPU processes via torch.multiprocessing.spawn",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Intercepts test function call before execution."""
    marker = pyfuncitem.get_closest_marker("multi_rank")
    if not marker:
        return None  # Fallback to standard single-process pytest execution

    world_size = marker.kwargs.get("world_size", 2)
    port = get_free_port()

    # Retrieve the raw, un-wrapped top-level function object directly from the test item
    test_func = pyfuncitem.obj.__func__ if hasattr(pyfuncitem.obj, "__func__") else pyfuncitem.obj
    test_instance = pyfuncitem.instance

    test_args = ()
    test_kwargs = {
        name: pyfuncitem._fixtureinfo.argvalues[name]
        for name in pyfuncitem.fixturenames
        # if name in pyfuncitem._fixtureinfo.argvalues
        if name in pyfuncitem._fixtureinfo.argnames
    }

    # mp.spawn receives the top-level un-wrapped function reference
    torch.multiprocessing.spawn(
        _spawn_worker,
        args=(world_size, port, test_func, test_instance, test_args, test_kwargs),
        nprocs=world_size,
        join=True,
    )

    return True  # Indicates PyTest that the test execution has been handled
