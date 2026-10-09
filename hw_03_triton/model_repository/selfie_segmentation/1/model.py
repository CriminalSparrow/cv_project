"""Stateless CPU segmentation; each request is one RGB image."""
from pathlib import Path
from time import perf_counter

import cv2
import mediapipe as mp
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        cv2.setNumThreads(1)
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(Path(__file__).with_name("selfie_segmenter.tflite")),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=False,
            output_confidence_masks=True,
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)
        labels = {"model": args["model_name"], "version": args["model_version"],
                  "instance": args["model_instance_name"]}
        self.time_family = pb_utils.MetricFamily(
            name="segmentation_processing_seconds_total",
            description="Cumulative time inside request processing, including failed requests",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.active_family = pb_utils.MetricFamily(
            name="segmentation_requests_in_flight",
            description="Requests currently executing in the Python model (excludes Triton queue)",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.processing_time = self.time_family.Metric(labels=labels)
        self.active = self.active_family.Metric(labels=labels)

    def execute(self, requests):
        responses = []
        for request in requests:
            self.active.increment(1)
            started = perf_counter()
            try:
                tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE")
                if tensor is None:
                    raise ValueError("IMAGE is required")
                image = tensor.as_numpy()
                if image.dtype != np.uint8 or image.shape != (256, 256, 3):
                    raise ValueError("IMAGE must be UINT8 RGB [256,256,3]")
                result = self.segmenter.segment(mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)
                ))
                mask = result.confidence_masks[0].numpy_view().copy().astype(np.float32)
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("MASK", mask)]
                ))
            except Exception as exc:
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc))))
            finally:
                self.processing_time.increment(perf_counter() - started)
                self.active.increment(-1)
        return responses

    def finalize(self):
        self.segmenter.close()
        # Metrics must be released before their families (Python backend API).
        del self.processing_time, self.active
        del self.time_family, self.active_family
