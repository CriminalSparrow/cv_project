from dataclasses import dataclass
from typing import Dict
from transformers import AutoModel

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================

@dataclass
class FusionModelConfig:
    text_dim: int = 256
    video_dim: int = 256
    duration_dim: int = 32

    fusion_hidden_dim: int = 512
    fusion_output_dim: int = 256

    num_labels: int = 1
    dropout: float = 0.2

    # "multilabel" — для жанров / настроений / инструментов
    # "binary" — для задачи подходит / не подходит трек
    task_type: str = "multilabel"


# ============================================================
# Pooling
# ============================================================

class MeanPooling(nn.Module):
    """
    Усреднение токенов с учетом attention_mask.
    Подходит для BERT-like моделей.
    """
    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts


# ============================================================
# Text backbones
# ============================================================

class BaseTextBackbone(nn.Module):
    """
    Базовый интерфейс для текстового backbone.
    Любой новый текстовый encoder должен возвращать вектор размера output_dim.
    """

    output_dim: int

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class HFTextPairBackbone(BaseTextBackbone):
    """
    Считает отдельно:
    1. эмбеддинг пользовательского запроса
    2. эмбеддинг названия видео

    Потом объединяет их через отдельный FFN.

    Можно использовать:
    - один общий encoder для query и title
    - два разных encoder'а
    """

    def __init__(
        self,
        query_model_name: str,
        title_model_name: str | None = None,
        encoder_output_dim: int = 256,
        fusion_hidden_dim: int = 512,
        fusion_output_dim: int = 256,
        dropout: float = 0.2,
        freeze_query_encoder: bool = False,
        freeze_title_encoder: bool = False,
        share_encoder: bool = True,
        use_interactions: bool = True,
    ):
        super().__init__()

        self.share_encoder = share_encoder
        self.pooling = MeanPooling()

        self.query_encoder = AutoModel.from_pretrained(query_model_name)

        if share_encoder:
            self.title_encoder = self.query_encoder
        else:
            if title_model_name is None:
                title_model_name = query_model_name

            self.title_encoder = AutoModel.from_pretrained(title_model_name)

        query_hidden_size = self.query_encoder.config.hidden_size
        title_hidden_size = self.title_encoder.config.hidden_size

        self.query_projection = nn.Linear(
            query_hidden_size,
            encoder_output_dim,
        )

        self.title_projection = nn.Linear(
            title_hidden_size,
            encoder_output_dim,
        )

        self.text_fusion = TextFusionFFN(
            query_dim=encoder_output_dim,
            title_dim=encoder_output_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim,
            dropout=dropout,
            use_interactions=use_interactions,
        )

        self.output_dim = fusion_output_dim

        if freeze_query_encoder:
            for p in self.query_encoder.parameters():
                p.requires_grad = False

        if freeze_title_encoder and not share_encoder:
            for p in self.title_encoder.parameters():
                p.requires_grad = False

    def encode_query(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.query_encoder(
            input_ids=batch["query_input_ids"],
            attention_mask=batch["query_attention_mask"],
        )

        pooled = self.pooling(
            outputs.last_hidden_state,
            batch["query_attention_mask"],
        )

        query_emb = self.query_projection(pooled)
        query_emb = F.normalize(query_emb, dim=-1)

        return query_emb

    def encode_title(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.title_encoder(
            input_ids=batch["title_input_ids"],
            attention_mask=batch["title_attention_mask"],
        )

        pooled = self.pooling(
            outputs.last_hidden_state,
            batch["title_attention_mask"],
        )

        title_emb = self.title_projection(pooled)
        title_emb = F.normalize(title_emb, dim=-1)

        return title_emb

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        query_emb = self.encode_query(batch)
        title_emb = self.encode_title(batch)

        text_emb = self.text_fusion(
            query_emb=query_emb,
            title_emb=title_emb,
        )

        return text_emb


class PrecomputedTextPairBackbone(BaseTextBackbone):
    """
    Для случая, когда query_embedding и title_embedding
    уже заранее посчитаны.
    """

    def __init__(
        self,
        query_input_dim: int,
        title_input_dim: int,
        encoder_output_dim: int = 256,
        fusion_hidden_dim: int = 512,
        fusion_output_dim: int = 256,
        dropout: float = 0.2,
        use_interactions: bool = True,
    ):
        super().__init__()

        self.query_projection = nn.Sequential(
            nn.Linear(query_input_dim, encoder_output_dim),
            nn.LayerNorm(encoder_output_dim),
            nn.GELU(),
        )

        self.title_projection = nn.Sequential(
            nn.Linear(title_input_dim, encoder_output_dim),
            nn.LayerNorm(encoder_output_dim),
            nn.GELU(),
        )

        self.text_fusion = TextFusionFFN(
            query_dim=encoder_output_dim,
            title_dim=encoder_output_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim,
            dropout=dropout,
            use_interactions=use_interactions,
        )

        self.output_dim = fusion_output_dim

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        query_emb = self.query_projection(
            batch["query_features"].float()
        )

        title_emb = self.title_projection(
            batch["title_features"].float()
        )

        query_emb = F.normalize(query_emb, dim=-1)
        title_emb = F.normalize(title_emb, dim=-1)

        text_emb = self.text_fusion(
            query_emb=query_emb,
            title_emb=title_emb,
        )

        return text_emb


class TextFusionFFN(nn.Module):
    """
    Объединяет эмбеддинг пользовательского запроса
    и эмбеддинг названия видео.
    """

    def __init__(
        self,
        query_dim: int,
        title_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.2,
        use_interactions: bool = True,
    ):
        super().__init__()

        self.use_interactions = use_interactions

        if use_interactions:
            input_dim = query_dim + title_dim + query_dim
        else:
            input_dim = query_dim + title_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        self.output_dim = output_dim

    def forward(
        self,
        query_emb: torch.Tensor,
        title_emb: torch.Tensor,
    ) -> torch.Tensor:

        if self.use_interactions:
            # Дополнительный признак взаимодействия:
            # насколько похожи признаки запроса и названия по измерениям
            interaction = query_emb * title_emb

            x = torch.cat(
                [
                    query_emb,
                    title_emb,
                    interaction,
                ],
                dim=-1,
            )
        else:
            x = torch.cat(
                [
                    query_emb,
                    title_emb,
                ],
                dim=-1,
            )

        text_emb = self.net(x)
        text_emb = F.normalize(text_emb, dim=-1)

        return text_emb

# ============================================================
# Video backbones
# ============================================================


class BaseVideoBackbone(nn.Module):
    """
    Базовый интерфейс для видео backbone.
    Любой новый video encoder должен возвращать вектор размера output_dim.
    """

    output_dim: int

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class PrecomputedVideoBackbone(BaseVideoBackbone):
    """
    Вариант для случая, если видео-эмбеддинги уже заранее извлечены.
    Например, CLIP/VideoMAE/S3D/TimeSformer embeddings.
    """

    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        self.output_dim = output_dim

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        video_features = batch["video_features"].float()
        video_emb = self.projection(video_features)
        video_emb = F.normalize(video_emb, dim=-1)

        return video_emb


class ModuleVideoBackbone(BaseVideoBackbone):
    """
    Обертка над любой torch-моделью для видео.

    Ожидается, что video_model возвращает:
    - tensor [B, D]
    - или tensor [B, T, D]
    - или dict с ключом "last_hidden_state" / "pooler_output" / "features"

    Это позволяет легко подключить VideoMAE, TimeSformer, CLIP-video,
    torchvision video models и т.д.
    """

    def __init__(
        self,
        video_model: nn.Module,
        input_dim: int,
        output_dim: int = 256,
        freeze: bool = False,
    ):
        super().__init__()

        self.video_model = video_model
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        self.output_dim = output_dim

        if freeze:
            for p in self.video_model.parameters():
                p.requires_grad = False

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        frames = batch["video_frames"]

        # [B, T, C, H, W] -> [B, C, T, H, W]
        frames = frames.permute(0, 2, 1, 3, 4).contiguous()

        outputs = self.video_model(frames)

        if isinstance(outputs, dict):
            if "pooler_output" in outputs:
                features = outputs["pooler_output"]
            elif "features" in outputs:
                features = outputs["features"]
            elif "last_hidden_state" in outputs:
                features = outputs["last_hidden_state"].mean(dim=1)
            else:
                raise ValueError(f"Unknown video model output keys: {outputs.keys()}")
        else:
            features = outputs

        if features.ndim == 3:
            features = features.mean(dim=1)

        video_emb = self.projection(features)
        video_emb = F.normalize(video_emb, dim=-1)

        return video_emb


# ============================================================
# Duration encoder
# ============================================================

class DurationEncoder(nn.Module):
    """
    Кодирует длительность видео в отдельный небольшой эмбеддинг.
    На вход подается длительность в секундах.
    """

    def __init__(self, output_dim: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        self.output_dim = output_dim

    def forward(self, duration: torch.Tensor) -> torch.Tensor:
        if duration.ndim == 1:
            duration = duration.unsqueeze(-1)

        duration = duration.float()

        # log1p помогает сгладить большой разброс длительностей
        duration = torch.log1p(duration)

        return self.encoder(duration)


# ============================================================
# Fusion head
# ============================================================

class FusionHead(nn.Module):
    """
    Объединяет:
    - текстовый эмбеддинг запроса + названия видео
    - видео-эмбеддинг
    - эмбеддинг длительности
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Main model
# ============================================================

class MusicMatchingModel(nn.Module):
    """
    Основная модель.

    Вход:
    - текст: запрос пользователя + название видео
    - видео: video_features или video_frames
    - длительность: duration
    - labels: разметка, если идет обучение

    Выход:
    - logits
    - probabilities
    - fused_embedding
    - loss, если labels переданы
    """

    def __init__(
        self,
        text_backbone: BaseTextBackbone,
        video_backbone: BaseVideoBackbone,
        config: FusionModelConfig,
    ):
        super().__init__()

        self.text_backbone = text_backbone
        self.video_backbone = video_backbone
        self.config = config

        self.duration_encoder = DurationEncoder(
            output_dim=config.duration_dim,
        )

        fusion_input_dim = (
            text_backbone.output_dim
            + video_backbone.output_dim
            + config.duration_dim
        )

        self.fusion = FusionHead(
            input_dim=fusion_input_dim,
            hidden_dim=config.fusion_hidden_dim,
            output_dim=config.fusion_output_dim,
            dropout=config.dropout,
        )

        self.classifier = nn.Linear(
            config.fusion_output_dim,
            config.num_labels,
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        text_emb = self.text_backbone(batch)
        video_emb = self.video_backbone(batch)
        duration_emb = self.duration_encoder(batch["duration"])

        fused_input = torch.cat(
            [
                text_emb,
                video_emb,
                duration_emb,
            ],
            dim=-1,
        )

        fused_embedding = self.fusion(fused_input)
        logits = self.classifier(fused_embedding)

        output = {
            "logits": logits,
            "fused_embedding": fused_embedding,
        }

        if self.config.task_type in {"multilabel", "binary"}:
            probabilities = torch.sigmoid(logits)
            output["probabilities"] = probabilities
        else:
            probabilities = torch.softmax(logits, dim=-1)
            output["probabilities"] = probabilities

        if "labels" in batch and batch["labels"] is not None:
            if self.config.task_type in {"multilabel", "binary"}:
                labels = batch["labels"].float()
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            else:
                labels = batch["labels"].long()
                class_weights = getattr(self, "class_weights", None)

                loss = F.cross_entropy(
                    logits,
                    labels,
                    weight=class_weights,
                    label_smoothing=0.05,
                )

            output["loss"] = loss

        return output
