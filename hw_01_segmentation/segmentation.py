from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


MODEL = Path(__file__).resolve().parent / "models" / "selfie_segmenter.tflite"


def composite(frame, background, mask):
    alpha = mask[..., None]
    return np.clip(frame * alpha + background * (1 - alpha), 0, 255).astype(np.uint8)


def fit_background(image, width, height):
    """Заполнить кадр картинкой без изменения пропорций."""
    scale = max(width / image.shape[1], height / image.shape[0])
    image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    x = (image.shape[1] - width) // 2
    y = (image.shape[0] - height) // 2
    return image[y:y + height, x:x + width]


class BackgroundRemover:
    def __init__(self, width, height, mode="color", color=(80, 160, 80),
                 background=None, smoothing=0.25):
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(MODEL),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=False,
            output_confidence_masks=True,
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)
        self.mode = mode
        self.smoothing = smoothing
        self.previous_mask = None
        if background is not None:
            self.background = fit_background(background, width, height)
        else:
            self.background = np.full((height, width, 3), color, dtype=np.uint8)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        # У Selfie Segmenter единственный канал — вероятность переднего плана.
        mask = result.confidence_masks[0].numpy_view().copy()
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = np.clip((mask - 0.25) / 0.5, 0, 1)
        if self.previous_mask is not None:
            mask = (1 - self.smoothing) * mask + self.smoothing * self.previous_mask
        self.previous_mask = mask

        background = self.background
        if self.mode == "blur":
            small = cv2.resize(frame, None, fx=0.25, fy=0.25)
            small = cv2.GaussianBlur(small, (21, 21), 0)
            background = cv2.resize(small, (frame.shape[1], frame.shape[0]))
        return composite(frame, background, mask)

    def close(self):
        self.segmenter.close()
