from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoProcessor


def get_video_duration_s(video_path: Union[str, Path]) -> float:
    """Возвращает длительность видео в секундах."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    if fps is None or fps <= 0:
        raise RuntimeError(f"Could not read FPS for video: {video_path}")

    return float(frame_count / fps)


def sample_video_frames_rgb(
    video_path: Union[str, Path],
    num_frames: int = 16,
    image_size: int = 224,
) -> List[np.ndarray]:
    """
    Возвращает список RGB-кадров для HF processor.

    Каждый кадр: np.ndarray [H, W, 3], uint8.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frame count: {video_path}")

    frame_indices = np.linspace(
        0,
        frame_count - 1,
        num_frames,
    ).astype(int)

    frames = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(
            frame,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return frames[:num_frames]


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


def load_video_encoder(
    model_name: str,
    device: torch.device,
    trust_remote_code: bool = True,
):
    """Загружает HF video processor и video encoder."""
    processor = load_hf_video_processor(
        model_name=model_name,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    ).to(device)

    model.eval()

    return processor, model


def prepare_single_hf_video_inputs(
    processor,
    frames: List[np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Готовит входы для одного видео.

    bool_masked_pos/mask удаляются: для feature extraction они не нужны.
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


def extract_hf_video_embedding(
    outputs,
    pooling: str = "mean",
) -> torch.Tensor:
    """Извлекает embedding видео из outputs HF-модели."""
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
            "image_embeds or last_hidden_state."
        )

    hidden = outputs.last_hidden_state

    if pooling == "cls":
        return hidden[:, 0]

    if pooling in {"auto", "mean"}:
        return hidden.mean(dim=1)

    raise ValueError(f"Unknown pooling: {pooling}")


@torch.no_grad()
def encode_video(
    video_path: Union[str, Path],
    processor,
    video_model,
    device: torch.device,
    num_frames: int = 16,
    image_size: int = 224,
    pooling: str = "mean",
) -> torch.Tensor:
    """Кодирует одно видео в L2-нормализованный embedding."""
    frames = sample_video_frames_rgb(
        video_path=video_path,
        num_frames=num_frames,
        image_size=image_size,
    )

    inputs = prepare_single_hf_video_inputs(
        processor=processor,
        frames=frames,
        device=device,
    )

    with torch.amp.autocast(
        device_type=device.type,
        enabled=device.type == "cuda",
    ):
        outputs = video_model(**inputs)

        emb = extract_hf_video_embedding(
            outputs=outputs,
            pooling=pooling,
        )

        emb = F.normalize(emb, dim=-1)

    return emb.squeeze(0).detach().cpu()
