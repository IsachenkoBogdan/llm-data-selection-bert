import pandas as pd
import evaluate


def compute_perplexity(
    texts: pd.Series,
    model_name: str = "Qwen/Qwen3-1.7B-Base",
    max_length: int = 256,
    batch_size: int | None = None,
    add_start_token: bool = False,
) -> pd.Series:
    if texts.empty:
        return pd.Series([], index=texts.index, name="pppl")

    metric = evaluate.load("perplexity")
    text_list = texts.astype(str).tolist()
    kwargs = {"predictions": text_list, "model_id": model_name}
    if max_length is not None:
        kwargs["max_length"] = max_length
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    kwargs["add_start_token"] = add_start_token
    result = metric.compute(**kwargs)

    return pd.Series(result["perplexities"], index=texts.index, name="pppl")
