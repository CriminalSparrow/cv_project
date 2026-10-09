import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from main import parse_args, run
from segmentation import composite, fit_background


class SegmentationTests(unittest.TestCase):
    def test_composite_foreground_background_and_soft_edge(self):
        frame = np.full((1, 3, 3), 200, dtype=np.uint8)
        background = np.full_like(frame, 100)
        mask = np.array([[0, 0.5, 1]], dtype=np.float32)
        result = composite(frame, background, mask)
        np.testing.assert_array_equal(result[0, :, 0], [100, 150, 200])
        self.assertEqual(result.dtype, np.uint8)

    def test_background_center_crop(self):
        image = np.zeros((10, 30, 3), dtype=np.uint8)
        image[:, 10:20] = 255
        result = fit_background(image, 10, 10)
        self.assertEqual(result.shape, (10, 10, 3))
        self.assertTrue(np.all(result == 255))

    def test_invalid_smoothing_rejected(self):
        for value in ("-0.1", "1", "nan"):
            with self.subTest(value=value), patch("sys.argv", ["main.py", "--smoothing", value]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args()

    def test_video_pipeline_and_measurement(self):
        # Реальный инференс и видеодекодер, без камеры и окна.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(15):
                    writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            finally:
                writer.release()
            image = root / "background.png"
            cv2.imwrite(str(image), np.full((20, 30, 3), 128, dtype=np.uint8))
            for mode in ("color", "blur", "image"):
                report = root / f"{mode}.json"
                command = ["main.py", "--video", str(video), "--headless", "--mode", mode,
                           "--width", "320", "--height", "240", "--frames", "3",
                           "--report", str(report)]
                if mode == "image":
                    command += ["--background", str(image)]
                with self.subTest(mode=mode), patch("sys.argv", command):
                    with contextlib.redirect_stdout(io.StringIO()):
                        run(parse_args())
                    result = json.loads(report.read_text())
                    self.assertEqual(result["measured_frames"], 3)
                    self.assertEqual(result["warmup_frames"], 10)
                    self.assertEqual(result["input_resolution"], [320, 240])
                    self.assertAlmostEqual(result["processing_fps"], 3 / result["processing_seconds"])
                    self.assertGreater(result["loop_seconds"], result["processing_seconds"])


if __name__ == "__main__":
    unittest.main()
