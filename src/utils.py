import random
import time
from contextlib import contextmanager


@contextmanager
def time_block():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def set_seed(seed: int = 42):
    import torch, numpy
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)