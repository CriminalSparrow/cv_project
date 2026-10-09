"""HTTP/gRPC request, mask validation, optional comparison with direct MediaPipe."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB").resize((256, 256)), dtype=np.uint8)


def infer(image, url, protocol="http"):
    if protocol == "grpc":
        import tritonclient.grpc as client
    else:
        import tritonclient.http as client
    with client.InferenceServerClient(url=url) as server:
        if not server.is_model_ready("selfie_segmentation"):
            raise RuntimeError("selfie_segmentation is not ready")
        tensor = client.InferInput("IMAGE", image.shape, "UINT8")
        tensor.set_data_from_numpy(image)
        result = server.infer("selfie_segmentation", [tensor],
                              outputs=[client.InferRequestedOutput("MASK")])
        mask = result.as_numpy("MASK")
    if mask.shape != (256, 256) or mask.dtype != np.float32:
        raise AssertionError(f"Unexpected mask: {mask.shape}, {mask.dtype}")
    if not np.isfinite(mask).all() or mask.min() < -1e-6 or mask.max() > 1 + 1e-6:
        raise AssertionError("Mask values must be finite probabilities")
    return mask


def reference(image):
    import mediapipe as mp
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(ROOT / "model_repository/selfie_segmentation/1/selfie_segmenter.tflite"),
            delegate=mp.tasks.BaseOptions.Delegate.CPU),
        output_category_mask=False, output_confidence_masks=True)
    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
        return segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                                 ).confidence_masks[0].numpy_view().copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["http", "grpc"], default="http")
    parser.add_argument("--url", help="Default: localhost:8000 (HTTP), localhost:8001 (gRPC)")
    parser.add_argument("--image", type=Path, default=ROOT / "examples/person.jpg")
    parser.add_argument("--output", type=Path, default=ROOT / "results/local_client")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--check-reference", action="store_true")
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be positive")
    url = args.url or ("localhost:8001" if args.protocol == "grpc" else "localhost:8000")
    image = load_image(args.image)
    ref = reference(image) if args.check_reference else None
    summaries = []
    for _ in range(args.requests):
        mask = infer(image, url, args.protocol)
        item = {"min": float(mask.min()), "max": float(mask.max()), "mean": float(mask.mean())}
        if ref is not None:
            np.testing.assert_allclose(mask, ref, atol=1e-5, rtol=1e-5)
            item["reference_max_abs_error"] = float(np.max(np.abs(mask - ref)))
        summaries.append(item)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "mask.npy", mask)
    Image.fromarray((mask * 255).clip(0, 255).astype(np.uint8)).save(args.output / "mask.png")
    alpha = np.clip((mask[..., None] - 0.25) / 0.5, 0, 1)
    merged = image * alpha + np.array([80, 160, 80]) * (1 - alpha)
    comparison = np.concatenate([image, merged.clip(0, 255).astype(np.uint8)], axis=1)
    Image.fromarray(comparison).save(args.output / "comparison.png")
    report = {"protocol": args.protocol, "url": url, "requests": summaries}
    (args.output / "requests.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
