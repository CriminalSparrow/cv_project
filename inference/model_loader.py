"""Загрузка обученной multilabel-модели без precomputed embeddings."""

import torch

from train.architacture import (
    FusionModelConfig,
    PrecomputedTextPairBackbone,
    PrecomputedVideoBackbone,
    MusicMatchingModel,
)

from inference.load_track_mapping import load_track_mappings


def create_inference_multilabel_model(
    num_labels: int,
    query_input_dim: int = 384,
    title_input_dim: int = 384,
    video_input_dim: int = 768,
    text_dim: int = 256,
    video_dim: int = 256,
    duration_dim: int = 32,
    fusion_hidden_dim: int = 512,
    fusion_output_dim: int = 256,
    dropout: float = 0.2,
):
    """
    Создаёт модель той же архитектуры, что использовалась при обучении.

    На вход модель получает уже посчитанные embeddings:
    - query_features
    - title_features
    - video_features
    - duration
    """
    config = FusionModelConfig(
        text_dim=text_dim,
        video_dim=video_dim,
        duration_dim=duration_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_output_dim=fusion_output_dim,
        num_labels=num_labels,
        dropout=dropout,
        task_type="multilabel",
    )

    text_backbone = PrecomputedTextPairBackbone(
        query_input_dim=query_input_dim,
        title_input_dim=title_input_dim,
        encoder_output_dim=text_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_output_dim=config.text_dim,
        dropout=dropout,
        use_interactions=True,
    )

    video_backbone = PrecomputedVideoBackbone(
        input_dim=video_input_dim,
        output_dim=config.video_dim,
    )

    model = MusicMatchingModel(
        text_backbone=text_backbone,
        video_backbone=video_backbone,
        config=config,
    )

    return model


def load_multilabel_model(
    checkpoint_path: str,
    device: torch.device,
    query_input_dim: int = 384,
    title_input_dim: int = 384,
    video_input_dim: int = 768,
    text_dim: int = 256,
    video_dim: int = 256,
    duration_dim: int = 32,
    fusion_hidden_dim: int = 512,
    fusion_output_dim: int = 256,
    dropout: float = 0.2,
):
    """
    Загружает checkpoint один раз и возвращает:
    - model
    - track2idx
    - idx2track
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    track2idx, idx2track = load_track_mappings(checkpoint)

    model = create_inference_multilabel_model(
        num_labels=len(track2idx),
        query_input_dim=query_input_dim,
        title_input_dim=title_input_dim,
        video_input_dim=video_input_dim,
        text_dim=text_dim,
        video_dim=video_dim,
        duration_dim=duration_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_output_dim=fusion_output_dim,
        dropout=dropout,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint must contain 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    return model, track2idx, idx2track
