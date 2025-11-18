from .base import RandomSel, FullSel
from .diversity import KCenterSel, KMeansSel, HerdingSel
from .statistical import PredictiveEntropySel, WordPieceRatioSel
from .llm_quality import LLMQualitySel
from .datadiet import DataDietSel

REGISTRY = {
    "random": RandomSel,
    "full": FullSel,
    "kcenter": KCenterSel,
    "kmeans": KMeansSel,
    "herding": HerdingSel,
    "predictive_entropy": PredictiveEntropySel,
    "wordpiece_ratio": WordPieceRatioSel,
    "llm_quality": LLMQualitySel,
    "datadiet": DataDietSel,
}
