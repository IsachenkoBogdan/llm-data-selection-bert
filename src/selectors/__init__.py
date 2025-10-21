from .base import BaseSelector, FullSel, RandomSel
from .diversity import HerdingSel, KCenterSel, KMeansSel
from .quality import PerplexitySel


REGISTRY = {
    "random": RandomSel,
    "full": FullSel,
    "kcenter": KCenterSel,
    "kmeans": KMeansSel,
    "herding": HerdingSel,
    "perplexity": PerplexitySel,
}
