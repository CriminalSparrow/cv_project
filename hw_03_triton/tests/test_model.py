"""Unit checks for metrics cleanup and request isolation; no running Triton required."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class Metric:
    def __init__(self):
        self.value = 0
        self.history = []

    def increment(self, delta):
        self.value += delta
        self.history.append(self.value)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.pb = SimpleNamespace(
            get_input_tensor_by_name=lambda request, name: request,
            Tensor=lambda name, data: SimpleNamespace(name=name, data=data),
            TritonError=lambda message: message,
            InferenceResponse=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        path = Path(__file__).resolve().parents[1] / "model_repository/selfie_segmentation/1/model.py"
        spec = importlib.util.spec_from_file_location("hw03_model", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"triton_python_backend_utils": self.pb,
                                     "mediapipe": MagicMock(), "cv2": MagicMock()}):
            spec.loader.exec_module(module)
        self.model = module.TritonPythonModel()
        self.model.active = Metric()
        self.model.processing_time = Metric()
        self.model.segmenter = MagicMock()

    @staticmethod
    def request(image=None):
        if image is None:
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        return SimpleNamespace(as_numpy=lambda: image)

    def test_success_counts_only_active_processing(self):
        def segment(_):
            self.assertEqual(self.model.active.value, 1)
            return SimpleNamespace(confidence_masks=[SimpleNamespace(
                numpy_view=lambda: np.ones((256, 256), np.float32))])
        self.model.segmenter.segment.side_effect = segment
        responses = self.model.execute([self.request(), self.request()])
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0].output_tensors[0].data.dtype, np.float32)
        self.assertEqual(self.model.active.history, [1, 0, 1, 0])
        self.assertGreater(self.model.processing_time.value, 0)

    def test_failure_does_not_leak_gauge_or_stop_later_request(self):
        mask = SimpleNamespace(confidence_masks=[SimpleNamespace(
            numpy_view=lambda: np.zeros((256, 256), np.float32))])
        self.model.segmenter.segment.side_effect = [RuntimeError("inference failed"), mask]
        responses = self.model.execute([self.request(), self.request()])
        self.assertEqual(responses[0].error, "inference failed")
        self.assertTrue(hasattr(responses[1], "output_tensors"))
        self.assertEqual(self.model.active.value, 0)
        self.assertEqual(len(self.model.processing_time.history), 2)
        self.assertGreater(self.model.processing_time.history[1], self.model.processing_time.history[0])

    def test_invalid_input_is_rejected(self):
        for request in [None, self.request(np.zeros((1, 2), np.uint8)),
                        self.request(np.zeros((256, 256, 3), np.float32))]:
            with self.subTest(request=request):
                result = self.model.execute([request])[0]
                self.assertTrue(hasattr(result, "error"))
                self.assertEqual(self.model.active.value, 0)
        self.model.segmenter.segment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
