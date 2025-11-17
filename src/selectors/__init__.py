from .base import FullSel, RandomSel
from .diversity import HerdingSel, KCenterSel, KMeansSel
from .quality import PerplexitySel
from .statistical import PredictiveEntropySel, WordPieceRatioSel
from .llm_quality import LLMQualitySel


REGISTRY = {
    "random": RandomSel,
    "full": FullSel,
    "kcenter": KCenterSel,
    "kmeans": KMeansSel,
    "herding": HerdingSel,
    "perplexity": PerplexitySel,
	"entropy": PredictiveEntropySel,
    "wp_ratio": WordPieceRatioSel,
    "llm_quality": LLMQualitySel,
}
