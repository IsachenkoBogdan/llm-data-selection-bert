from .base import FullSel, RandomSel
from .diversity import HerdingSel, KCenterSel, KMeansSel
from .quality import PerplexitySel
from .statistical import PredictiveEntropySel, WordPieceRatioSel


REGISTRY = {
    "random": RandomSel,
    "full": FullSel,
    "kcenter": KCenterSel,
    "kmeans": KMeansSel,
    "herding": HerdingSel,
    "perplexity": PerplexitySel,
	"entropy": PredictiveEntropySel,         # новый метод
    "wp_ratio": WordPieceRatioSel,          # новый метод
}
