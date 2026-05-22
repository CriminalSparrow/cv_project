"""Текстовый энкодер для получения эмбеддингов текстов."""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def mean_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean pooling token embeddings с учётом attention mask."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)

    return summed / counts


def load_text_encoder(
    model_name: str,
    device: torch.device,
):
    """Загружает HF tokenizer и text encoder."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    return tokenizer, model


@torch.no_grad()
def encode_text(
    text: str,
    tokenizer,
    text_model,
    device: torch.device,
    prefix: str,
    max_length: int = 64,
) -> torch.Tensor:
    """
    Кодирует один текст в embedding.

    Для intfloat/multilingual-e5-small:
        user_query  -> prefix="query"
        video_title -> prefix="passage"
    """
    text = f"{prefix}: {text}"

    tokens = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    tokens = {
        key: value.to(device)
        for key, value in tokens.items()
    }

    with torch.amp.autocast(
        device_type=device.type,
        enabled=device.type == "cuda",
    ):
        outputs = text_model(**tokens)

        emb = mean_pooling(
            outputs.last_hidden_state,
            tokens["attention_mask"],
        )

        emb = F.normalize(emb, dim=-1)

    return emb.squeeze(0).detach().cpu()
