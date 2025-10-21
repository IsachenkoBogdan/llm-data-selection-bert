import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np


def compute_modernbert_embeddings(
    texts: pd.Series,
    model_name: str = "answerdotai/ModernBERT-base",
    max_length: int = 128,
    batch_size: int = 256,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    text_list = texts.tolist()
    embeddings: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(text_list), batch_size):
            batch_texts = text_list[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            
            cls_hidden = outputs.last_hidden_state[:, 0, :].detach().cpu()
            embeddings.append(cls_hidden)

    return torch.cat(embeddings, dim=0).numpy()
