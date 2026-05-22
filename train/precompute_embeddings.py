"""
precompute_embeddings.py

Предрасчёт эмбеддингов для модели подбора музыки.

Сохраняет:
- query_embeddings.npy
- title_embeddings.npy
- video_embeddings.npy
- query2idx.json
- title2idx.json
- video2idx.json
- metadata.json
- train_with_emb_idx.csv
- val_with_emb_idx.csv
- test_with_emb_idx.csv

Используется:
- HF text encoder для user_query и video_title;
- HF video encoder + HF processor для video_path.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoImageProcessor,
    AutoProcessor,
)
from common_utils.video_utils import (
    resolve_video_path,
    sample_video_frames_rgb
)


def mean_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean pooling token embeddings с учётом attention mask."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)

    return summed / counts


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Возвращает device. Если не указан — cuda при наличии, иначе cpu."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


@torch.no_grad()
def encode_texts(
    texts: List[str],
    model_name: str = "intfloat/multilingual-e5-small",
    batch_size: int = 64,
    max_length: int = 64,
    device: Optional[Union[str, torch.device]] = None,
) -> np.ndarray:
    """Кодирует список текстов в L2-нормализованные эмбеддинги."""
    device = resolve_device(device)

    if len(texts) == 0:
        raise ValueError("texts is empty.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    use_amp = device.type == "cuda"

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch_texts = texts[start:start + batch_size]

        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(**tokens)

            embeddings = mean_pooling(
                outputs.last_hidden_state,
                tokens["attention_mask"],
            )

            embeddings = F.normalize(embeddings, dim=-1)

        all_embeddings.append(
            embeddings.detach().cpu().float().numpy()
        )

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)

# ============================================================
# HF video encoding
# ============================================================


def load_hf_video_processor(
    model_name: str,
    trust_remote_code: bool = True,
):
    """Загружает HF processor для video/image model."""
    try:
        return AutoImageProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        return AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )


def prepare_single_hf_video_inputs(
    processor,
    frames: List[np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Готовит входы для одного видео.
    frames: List[np.ndarray], где каждый кадр [H, W, 3].
    """
    call_variants = [
        lambda: processor(frames, return_tensors="pt"),
        lambda: processor(videos=frames, return_tensors="pt"),
        lambda: processor(images=frames, return_tensors="pt"),
    ]

    last_error = None

    for fn in call_variants:
        try:
            inputs = fn()

            inputs.pop("bool_masked_pos", None)
            inputs.pop("mask", None)

            return {
                key: value.to(device)
                for key, value in inputs.items()
                if torch.is_tensor(value)
            }

        except Exception as e:
            last_error = e

    raise RuntimeError(
        "Could not prepare HF video inputs. "
        f"Last processor error: {last_error}"
    )


def collate_hf_video_inputs(
    inputs_list: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Склеивает inputs отдельных видео в batch.
    Обычно для VideoMAE ключ — pixel_values.
    """
    batch = {}

    keys = inputs_list[0].keys()

    for key in keys:
        values = [inputs[key] for inputs in inputs_list]

        # У каждого отдельного видео обычно shape [1, T, C, H, W]
        batch[key] = torch.cat(values, dim=0)

    return batch


def extract_hf_video_embedding(
    outputs,
    pooling: str = "auto",
) -> torch.Tensor:
    """
    Извлекает один embedding на видео из outputs HF-модели.

    pooling:
    - auto
    - pooler
    - cls
    - mean
    """
    if pooling in {"auto", "pooler"}:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output:
            return outputs.pooler_output

    if pooling == "auto":
        if hasattr(outputs, "video_embeds") and outputs.video_embeds:
            return outputs.video_embeds

        if hasattr(outputs, "image_embeds") and outputs.image_embeds:
            return outputs.image_embeds

    if not hasattr(outputs, "last_hidden_state"):
        raise ValueError(
            "Cannot extract embedding: no pooler_output, video_embeds, "
            "image_embeds or last_hidden_state in model outputs."
        )

    hidden = outputs.last_hidden_state

    if pooling == "cls":
        return hidden[:, 0]

    if pooling in {"auto", "mean"}:
        return hidden.mean(dim=1)

    raise ValueError(f"Unknown pooling: {pooling}")


@torch.no_grad()
def encode_videos_hf(
    video_paths: List[str],
    video_root: str,
    model_name: str,
    batch_size: int = 1,
    num_frames: int = 16,
    image_size: int = 224,
    device: Optional[Union[str, torch.device]] = None,
    missing_video: str = "error",
    pooling: str = "auto",
    trust_remote_code: bool = True,
) -> np.ndarray:
    """
    Кодирует видео через HF video encoder.

    Важно: каждое видео отдельно прогоняется через processor,
    чтобы HF processor не воспринял весь batch как одно видео.
    """
    device = resolve_device(device)

    if len(video_paths) == 0:
        raise ValueError("video_paths is empty.")

    processor = load_hf_video_processor(
        model_name=model_name,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    ).to(device)

    model.eval()

    all_embeddings = []
    use_amp = device.type == "cuda"

    for start in tqdm(
        range(0, len(video_paths), batch_size),
        desc=f"Encoding videos: {model_name}",
    ):
        batch_paths = video_paths[start:start + batch_size]

        inputs_list = []

        for rel_path in batch_paths:
            path = resolve_video_path(rel_path, video_root)

            frames = sample_video_frames_rgb(
                video_path=path,
                num_frames=num_frames,
                image_size=image_size,
                missing_video=missing_video,
            )

            single_inputs = prepare_single_hf_video_inputs(
                processor=processor,
                frames=frames,
                device=device,
            )

            inputs_list.append(single_inputs)

        inputs = collate_hf_video_inputs(inputs_list)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(**inputs)

            embeddings = extract_hf_video_embedding(
                outputs=outputs,
                pooling=pooling,
            )

            if embeddings.shape[0] != len(batch_paths):
                raise RuntimeError(
                    f"HF video encoder returned wrong batch size. "
                    f"Expected {len(batch_paths)}, got {embeddings.shape[0]}. "
                    f"Inputs shapes: "
                    f"{ {k: tuple(v.shape) for k, v in inputs.items()} }"
                )

            embeddings = F.normalize(embeddings, dim=-1)

        all_embeddings.append(
            embeddings.detach().cpu().float().numpy()
        )

    video_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    if video_embeddings.shape[0] != len(video_paths):
        raise RuntimeError(
            f"Wrong number of video embeddings: "
            f"expected {len(video_paths)}, got {video_embeddings.shape[0]}"
        )

    return video_embeddings


# ============================================================
# DataFrame utils
# ============================================================

def _check_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    name: str,
) -> None:
    """Проверяет наличие нужных колонок."""
    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )


# ============================================================
# Main
# ============================================================

def build_precomputed_embeddings(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    video_root: str,
    text_model_name: str = "intfloat/multilingual-e5-small",
    video_model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
    query_col: str = "user_query",
    query_id_col: str = "query_id",
    title_col: str = "video_title",
    video_id_col: str = "video_id",
    video_path_col: str = "video_path",
    text_batch_size: int = 64,
    video_batch_size: int = 2,
    query_max_length: int = 64,
    title_max_length: int = 64,
    num_frames: int = 16,
    image_size: int = 224,
    device: Optional[Union[str, torch.device]] = None,
    missing_video: str = "error",
    hf_pooling: str = "auto",
    hf_trust_remote_code: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Считает эмбеддинги query, title и video.

    Возвращает train/val/test с колонками:
    - query_emb_idx
    - title_emb_idx
    - video_emb_idx
    """
    device = resolve_device(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = [
        query_id_col,
        query_col,
        video_id_col,
        title_col,
        video_path_col,
    ]

    _check_required_columns(train_df, required_cols, "train_df")
    _check_required_columns(val_df, required_cols, "val_df")
    _check_required_columns(test_df, required_cols, "test_df")

    all_df = pd.concat(
        [train_df, val_df, test_df],
        axis=0,
        ignore_index=True,
    )

    # -------------------------
    # Queries
    # -------------------------

    query_table = (
        all_df[[query_id_col, query_col]]
        .drop_duplicates(query_id_col)
        .reset_index(drop=True)
    )

    query_ids = query_table[query_id_col].astype(str).tolist()

    query_texts = [
        f"query: {text}"
        for text in query_table[query_col].fillna("").astype(str).tolist()
    ]

    query2idx: Dict[str, int] = {
        query_id: idx
        for idx, query_id in enumerate(query_ids)
    }

    print("Unique queries:", len(query_table))

    query_embeddings = encode_texts(
        texts=query_texts,
        model_name=text_model_name,
        batch_size=text_batch_size,
        max_length=query_max_length,
        device=device,
    )

    np.save(output_dir / "query_embeddings.npy", query_embeddings)

    with open(output_dir / "query2idx.json", "w", encoding="utf-8") as f:
        json.dump(query2idx, f, ensure_ascii=False, indent=2)

    # -------------------------
    # Titles
    # -------------------------

    title_table = (
        all_df[[video_id_col, title_col]]
        .drop_duplicates(video_id_col)
        .reset_index(drop=True)
    )

    title_video_ids = title_table[video_id_col].astype(str).tolist()

    title_texts = [
        f"passage: {text}"
        for text in title_table[title_col].fillna("").astype(str).tolist()
    ]

    title2idx: Dict[str, int] = {
        video_id: idx
        for idx, video_id in enumerate(title_video_ids)
    }

    print("Unique video titles:", len(title_table))

    title_embeddings = encode_texts(
        texts=title_texts,
        model_name=text_model_name,
        batch_size=text_batch_size,
        max_length=title_max_length,
        device=device,
    )

    np.save(output_dir / "title_embeddings.npy", title_embeddings)

    with open(output_dir / "title2idx.json", "w", encoding="utf-8") as f:
        json.dump(title2idx, f, ensure_ascii=False, indent=2)

    # -------------------------
    # Videos
    # -------------------------

    video_table = (
        all_df[[video_id_col, video_path_col]]
        .drop_duplicates(video_id_col)
        .reset_index(drop=True)
    )

    video_ids = video_table[video_id_col].astype(str).tolist()
    video_paths = video_table[video_path_col].astype(str).tolist()

    video2idx: Dict[str, int] = {
        video_id: idx
        for idx, video_id in enumerate(video_ids)
    }

    print("Unique videos:", len(video_table))

    video_embeddings = encode_videos_hf(
        video_paths=video_paths,
        video_root=video_root,
        model_name=video_model_name,
        batch_size=video_batch_size,
        num_frames=num_frames,
        image_size=image_size,
        device=device,
        missing_video=missing_video,
        pooling=hf_pooling,
        trust_remote_code=hf_trust_remote_code,
    )

    np.save(output_dir / "video_embeddings.npy", video_embeddings)

    with open(output_dir / "video2idx.json", "w", encoding="utf-8") as f:
        json.dump(video2idx, f, ensure_ascii=False, indent=2)

    # -------------------------
    # Metadata
    # -------------------------

    metadata = {
        "text_model_name": text_model_name,
        "video_model_name": video_model_name,
        "query_dim": int(query_embeddings.shape[1]),
        "title_dim": int(title_embeddings.shape[1]),
        "video_dim": int(video_embeddings.shape[1]),
        "num_queries": int(query_embeddings.shape[0]),
        "num_titles": int(title_embeddings.shape[0]),
        "num_videos": int(video_embeddings.shape[0]),
        "num_frames": int(num_frames),
        "image_size": int(image_size),
        "hf_pooling": hf_pooling,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # -------------------------
    # Add indices
    # -------------------------

    def add_indices(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["query_emb_idx"] = df[query_id_col].astype(str).map(query2idx)
        df["title_emb_idx"] = df[video_id_col].astype(str).map(title2idx)
        df["video_emb_idx"] = df[video_id_col].astype(str).map(video2idx)

        for col in ["query_emb_idx", "title_emb_idx", "video_emb_idx"]:
            if df[col].isna().any():
                bad_rows = df[df[col].isna()].head()
                raise ValueError(f"NaN in {col}. Example rows:\n{bad_rows}")

            df[col] = df[col].astype(int)

        return df.reset_index(drop=True)

    train_df = add_indices(train_df)
    val_df = add_indices(val_df)
    test_df = add_indices(test_df)

    train_df.to_csv(output_dir / "train_with_emb_idx.csv", index=False)
    val_df.to_csv(output_dir / "val_with_emb_idx.csv", index=False)
    test_df.to_csv(output_dir / "test_with_emb_idx.csv", index=False)

    print("Saved precomputed embeddings to:", output_dir)

    return train_df, val_df, test_df