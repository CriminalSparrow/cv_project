"""Инференс multilabel-модели без аудиоэмбеддингов."""

from typing import Dict

import numpy as np
import pandas as pd
import torch

from common_utils.path import resolve_path
from inference.text_encoder import encode_text
from inference.video_encoder import encode_video, get_video_duration_s


@torch.no_grad()
def recommend_tracks_for_video(
    video_path: str,
    video_title: str,
    user_query: str,
    model,
    tokenizer,
    text_model,
    video_processor,
    video_model,
    idx2track: Dict[int, str],
    device: torch.device,
    project_root: str | None = None,
    top_k: int = 10,
    query_max_length: int = 64,
    title_max_length: int = 64,
    num_video_frames: int = 16,
    image_size: int = 224,
    hf_pooling: str = "mean",
) -> pd.DataFrame:
    """
    Считает embeddings с нуля и возвращает top-K track_id.

    Модель не использует audio embeddings и не обращается к track catalog.
    """
    video_path = resolve_path(video_path, project_root)
    duration_s = get_video_duration_s(video_path)

    query_emb = encode_text(
        text=user_query,
        tokenizer=tokenizer,
        text_model=text_model,
        device=device,
        prefix="query",
        max_length=query_max_length,
    )

    title_emb = encode_text(
        text=video_title,
        tokenizer=tokenizer,
        text_model=text_model,
        device=device,
        prefix="passage",
        max_length=title_max_length,
    )

    video_emb = encode_video(
        video_path=video_path,
        processor=video_processor,
        video_model=video_model,
        device=device,
        num_frames=num_video_frames,
        image_size=image_size,
        pooling=hf_pooling,
    )

    batch = {
        "query_features": query_emb.unsqueeze(0).to(device),
        "title_features": title_emb.unsqueeze(0).to(device),
        "video_features": video_emb.unsqueeze(0).to(device),
        "duration": torch.tensor(
            [duration_s],
            dtype=torch.float32,
            device=device,
        ),
    }

    output = model(batch)

    logits = output["logits"]
    scores = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()

    top_indices = np.argsort(scores)[-top_k:][::-1]

    rows = []

    for rank, class_idx in enumerate(top_indices, start=1):
        track_id = str(idx2track[int(class_idx)])

        rows.append(
            {
                "rank": rank,
                "track_id": track_id,
                "model_score": float(scores[class_idx]),
                "duration_s": duration_s,
                "input_video_title": video_title,
                "user_query": user_query,
                "input_video_path": str(video_path),
            }
        )

    return pd.DataFrame(rows)
