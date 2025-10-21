import os

# Force a non-interactive backend for headless environments (e.g. Snakemake runs)
os.environ["MPLBACKEND"] = "Agg"

import pandas as pd
import textstat
from lexicalrichness import LexicalRichness


def compute_text_metrics(texts: pd.Series) -> pd.DataFrame:
    texts = texts.astype(str)
    
    word_lists = texts.str.split()
    token_counts = word_lists.apply(len).astype(float)
    char_counts = texts.str.len().astype(float)
    
    def compute_lexical_diversity(text: str) -> tuple[float, float, float]:
        if not text.strip() or len(text.split()) < 2:
            return 0.0, 0.0, 0.0
        try:
            lex = LexicalRichness(text)
            ttr = lex.ttr
            mtld = lex.mtld(threshold=0.72)
            mattr = lex.mattr(window_size=25)
            return ttr, mtld, mattr
        except (ValueError, ZeroDivisionError):
            return 0.0, 0.0, 0.0
    
    lexical_stats = texts.apply(compute_lexical_diversity)
    type_token_ratio = lexical_stats.apply(lambda x: x[0])
    mtld_scores = lexical_stats.apply(lambda x: x[1])
    mattr_scores = lexical_stats.apply(lambda x: x[2])
    
    avg_word_len = texts.apply(
        lambda t: sum(len(w) for w in t.split()) / max(len(t.split()), 1)
    )
    
    flesch_scores = texts.apply(lambda t: textstat.flesch_reading_ease(t) if t.strip() else 0.0)
    syllable_counts = texts.apply(lambda t: textstat.syllable_count(t) if t.strip() else 0)
    sentence_counts = texts.apply(lambda t: textstat.sentence_count(t) if t.strip() else 1)
    
    return pd.DataFrame({
        "token_count": token_counts,
        "char_count": char_counts,
        "type_token_ratio": type_token_ratio,
        "mtld": mtld_scores,
        "mattr": mattr_scores,
        "avg_word_len": avg_word_len,
        "flesch_reading_ease": flesch_scores,
        "syllable_count": syllable_counts,
        "sentence_count": sentence_counts,
    }, index=texts.index)
