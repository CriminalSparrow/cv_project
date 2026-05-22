# loaders.py

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


@dataclass
class PrecomputedMultilabelLoaderConfig:
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True

    query_emb_idx_col: str = "query_emb_idx"
    title_emb_idx_col: str = "title_emb_idx"
    video_emb_idx_col: str = "video_emb_idx"

    duration_col: str = "duration_s"
    labels_col: str = "labels"


class PrecomputedMultilabelMusicDataset(Dataset):
    def __init__(self, df, query_embeddings, title_embeddings,
                 video_embeddings, config):

        self.df = df.reset_index(drop=True)
        self.query_embeddings = query_embeddings
        self.title_embeddings = title_embeddings
        self.video_embeddings = video_embeddings
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        query_idx = int(row[self.config.query_emb_idx_col])
        title_idx = int(row[self.config.title_emb_idx_col])
        video_idx = int(row[self.config.video_emb_idx_col])
        labels = row[self.config.labels_col]

        if isinstance(labels, str):
            import ast
            labels = np.asarray(ast.literal_eval(labels), dtype=np.float32)
        else:
            labels = np.asarray(labels, dtype=np.float32)

        return {
            "query_features": self.query_embeddings[query_idx],
            "title_features": self.title_embeddings[title_idx],
            "video_features": self.video_embeddings[video_idx],
            "duration": torch.tensor(float(row[self.config.duration_col]),
                                     dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


def create_precomputed_multilabel_loaders(
    train_df,
    val_df,
    test_df,
    precomputed_dir,
    config=None,
):
    if config is None:
        config = PrecomputedMultilabelLoaderConfig()

    query_embeddings = torch.tensor(
        np.load(f"{precomputed_dir}/query_embeddings.npy"),
        dtype=torch.float32,
    )

    title_embeddings = torch.tensor(
        np.load(f"{precomputed_dir}/title_embeddings.npy"),
        dtype=torch.float32,
    )

    video_embeddings = torch.tensor(
        np.load(f"{precomputed_dir}/video_embeddings.npy"),
        dtype=torch.float32,
    )

    train_dataset = PrecomputedMultilabelMusicDataset(
        train_df,
        query_embeddings,
        title_embeddings,
        video_embeddings,
        config,
    )

    val_dataset = PrecomputedMultilabelMusicDataset(
        val_df,
        query_embeddings,
        title_embeddings,
        video_embeddings,
        config,
    )

    test_dataset = PrecomputedMultilabelMusicDataset(
        test_df,
        query_embeddings,
        title_embeddings,
        video_embeddings,
        config,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, test_loader
