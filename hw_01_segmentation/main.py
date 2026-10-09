import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from segmentation import BackgroundRemover


WINDOW = "Background removal | Q / Esc: exit"


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return value


def rgb_color(value):
    try:
        color = tuple(int(part) for part in value.split(","))
    except ValueError:
        color = ()
    if len(color) != 3 or any(part < 0 or part > 255 for part in color):
        raise argparse.ArgumentTypeError("цвет задаётся как R,G,B, от 0 до 255")
    return color[::-1]  # OpenCV использует BGR.


def parse_args():
    parser = argparse.ArgumentParser(description="Удаление фона на CPU (MediaPipe)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="индекс камеры, по умолчанию 0")
    source.add_argument("--video", type=Path, help="видеофайл вместо камеры")
    parser.add_argument("--width", type=positive_int, default=640)
    parser.add_argument("--height", type=positive_int, default=480)
    parser.add_argument("--mode", choices=("color", "blur", "image"), default="color")
    parser.add_argument("--color", type=rgb_color, default=rgb_color("80,160,80"))
    parser.add_argument("--background", type=Path, help="картинка для режима image")
    parser.add_argument("--smoothing", type=float, default=0.25, help="доля прошлой маски: [0, 1)")
    parser.add_argument("--compare", action="store_true", help="исходный кадр слева, результат справа")
    parser.add_argument("--headless", action="store_true", help="замер без окна и задержек воспроизведения")
    parser.add_argument("--frames", type=positive_int, help="число измеряемых кадров после прогрева")
    parser.add_argument("--report", type=Path, default=Path("results/local.json"))
    args = parser.parse_args()
    if not 0 <= args.smoothing < 1:
        parser.error("--smoothing должен находиться в диапазоне [0, 1)")
    if args.camera < 0:
        parser.error("индекс камеры не может быть отрицательным")
    if args.mode == "image" and args.background is None:
        parser.error("для --mode image нужен --background")
    if args.background is not None and args.mode != "image":
        parser.error("--background используется только с --mode image")
    return args


def cpu_name():
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def draw_text(frame, text, y):
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def run(args):
    cv2.setNumThreads(1)
    background = None
    if args.background is not None:
        background = cv2.imread(str(args.background))
        if background is None:
            raise ValueError(f"Не удалось прочитать фон: {args.background}")
    if args.video is not None and not args.video.is_file():
        raise ValueError(f"Видеофайл не найден: {args.video}")

    source = str(args.video) if args.video is not None else args.camera
    cap = cv2.VideoCapture(source)
    remover = None
    latencies = []
    recent_processing = deque(maxlen=30)
    recent_loop = deque(maxlen=30)
    elapsed = 0.0
    frames = 0
    warmup = 10
    input_size = None
    try:
        if not cap.isOpened():
            raise RuntimeError("Не удалось открыть источник. Проверьте путь, индекс камеры и доступ к ней.")
        if args.video is None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap.set(cv2.CAP_PROP_FPS, 30)
        file_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_period = 1 / file_fps if np.isfinite(file_fps) and file_fps > 0 else 1 / 30
        remover = BackgroundRemover(args.width, args.height, args.mode, args.color,
                                    background, args.smoothing)
        while True:
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if args.video is None or frames == 0:
                    raise RuntimeError("Источник не вернул кадр. Проверьте камеру или содержимое файла.")
                break
            input_size = [frame.shape[1], frame.shape[0]]
            processing_start = time.perf_counter()
            frame = cv2.resize(frame, (args.width, args.height))
            if args.video is None:
                frame = cv2.flip(frame, 1)
            output = remover.process(frame)
            processing_time = time.perf_counter() - processing_start
            frames += 1
            measured = frames > warmup
            if measured:
                latencies.append(processing_time)
                recent_processing.append(processing_time)

            key = -1
            if not args.headless:
                display = np.hstack((frame, output)) if args.compare else output
                if args.compare:
                    draw_text(display, "Original", 26)
                    cv2.putText(display, "Result", (args.width + 12, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                text = "Warmup..."
                if recent_processing:
                    fps = len(recent_processing) / sum(recent_processing)
                    loop_fps = len(recent_loop) / sum(recent_loop) if recent_loop else 0
                    text = f"CPU | process: {fps:.1f} FPS | loop: {loop_fps:.1f} FPS"
                draw_text(display, text, args.height - 18)
                cv2.imshow(WINDOW, display)
                # Файл показываем с его исходной частотой, замер headless идёт без пауз.
                delay = max(1, round(1000 * (frame_period - (time.perf_counter() - loop_start)))) \
                    if args.video is not None else 1
                key = cv2.waitKey(delay) & 0xFF
            if measured:
                loop_time = time.perf_counter() - loop_start
                elapsed += loop_time
                recent_loop.append(loop_time)
            if key in (ord("q"), 27) or (args.frames and len(latencies) >= args.frames):
                break
            if not args.headless and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
    finally:
        cap.release()
        if remover is not None:
            remover.close()
        if not args.headless:
            cv2.destroyAllWindows()

    if not latencies:
        raise RuntimeError("Недостаточно кадров для замера: первые 10 кадров отведены на прогрев.")
    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": cpu_name(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mediapipe": mp.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "delegate": "CPU",
        "source": str(source),
        "input_resolution": input_size,
        "processing_resolution": [args.width, args.height],
        "mode": args.mode,
        "smoothing": args.smoothing,
        "headless": args.headless,
        "compare": args.compare,
        "warmup_frames": warmup,
        "measured_frames": len(latencies),
        "processing_seconds": sum(latencies),
        "loop_seconds": elapsed,
        "processing_fps": len(latencies) / sum(latencies),
        "loop_fps": len(latencies) / elapsed,
        "mean_latency_ms": 1000 * float(np.mean(latencies)),
        "p95_latency_ms": 1000 * float(np.percentile(latencies, 95)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"CPU: {report['cpu']}; {args.width}x{args.height}; {len(latencies)} кадров")
    print(f"Обработка: {report['processing_fps']:.1f} FPS; "
          f"средняя задержка: {report['mean_latency_ms']:.2f} мс; "
          f"p95: {report['p95_latency_ms']:.2f} мс")
    print(f"Полный цикл: {report['loop_fps']:.1f} FPS. Отчёт: {args.report}")


if __name__ == "__main__":
    try:
        run(parse_args())
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)
