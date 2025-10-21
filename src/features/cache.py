import os
from joblib import Memory


def get_cache(cache_dir: str | None = None, verbose: int = 0) -> Memory:
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), ".cache")
    
    os.makedirs(cache_dir, exist_ok=True)
    return Memory(cache_dir, verbose=verbose)