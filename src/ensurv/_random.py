"""Prediction random-state management."""

from contextlib import contextmanager
import numpy as np
import torch


@contextmanager
def _prediction_rng(seed):
    r"""Temporarily seed prediction draws and restore the saved random states.

    :param seed: Temporary NumPy and PyTorch seed. If None, use the current random states
        without saving or reseeding them.
    :type seed: int or None

    :returns: A context in which prediction draws use the requested seed.
    :rtype: context manager

    With a numeric seed, NumPy, PyTorch CPU, and available CUDA random states are restored even
    if prediction raises an exception.
    """
    if seed is None:
        yield
        return
    np_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda_states is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

