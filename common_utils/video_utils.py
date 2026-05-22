from pathlib import Path

import cv2
import numpy as np
from typing import List, Union


def resolve_video_path(
    video_path: str,
    video_root: Union[str, Path],
) -> Path:
    """Разрешает относительный video_path относительно video_root."""
    path = Path(str(video_path).replace("\\", "/"))

    if path.is_absolute():
        return path

    return Path(video_root) / path


def sample_video_frames_rgb(
    video_path: Union[str, Path],
    num_frames: int = 8,
    image_size: int = 224,
    missing_video: str = "error",
) -> List[np.ndarray]:
    """
    Возвращает список RGB-кадров для HF processor.

    Каждый кадр: np.ndarray [H, W, 3], uint8.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        if missing_video == "zeros":
            return [
                np.zeros((image_size, image_size, 3), dtype=np.uint8)
                for _ in range(num_frames)
            ]

        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()

        if missing_video == "zeros":
            return [
                np.zeros((image_size, image_size, 3), dtype=np.uint8)
                for _ in range(num_frames)
            ]

        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()

        if missing_video == "zeros":
            return [
                np.zeros((image_size, image_size, 3), dtype=np.uint8)
                for _ in range(num_frames)
            ]

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
        if missing_video == "zeros":
            return [
                np.zeros((image_size, image_size, 3), dtype=np.uint8)
                for _ in range(num_frames)
            ]

        raise RuntimeError(f"No frames read from: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return frames[:num_frames]
