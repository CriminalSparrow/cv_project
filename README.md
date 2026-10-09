# ДЗ №3: инференс модели в Triton

Код и отчёт: [hw_03_triton/README.md](hw_03_triton/README.md).
Selfie Segmentation из ДЗ 1, Python backend, CPU, HTTP/gRPC и кастомные метрики.
Запуск (Docker Compose v2+, доступ к NGC/PyPI):

```bash
cd hw_03_triton
docker compose up --build -d --wait server
```

Замеры на Apple M3 Pro, linux/amd64 под эмуляцией, Triton 2.43.0:

| Конфигурация | Запросов/с | Задержка |
| --- | ---: | --- |
| 1 экземпляр, concurrency 1 (Performance Analyzer) | 128.157 | p95 8.635 мс |
| 1 экземпляр, concurrency 4 (Model Analyzer) | 138.1 | p99 32.4 мс |
| 4 экземпляра, concurrency 8 (Model Analyzer) | 494.0 | p99 20.1 мс |
| 4 экземпляра, concurrency 4 (Model Analyzer) | 475.4 | p99 9.9 мс |

Выбраны 4 экземпляра — ускорение в 3.58 раза. Для меньшей задержки подходит
concurrency 4: 475.4 запросов/с при p99 9.9 мс. Анализ и CSV — в папке задания.
Из-за ответа NGC 403 замеры выполнены на локальном образе Triton 2.43.0;
стандартная сборка через NGC 23.12 целиком не проверена.

---

# ДЗ №1: удаление фона в реальном времени

Ветка `hw_01_segmentation`, код и полный отчёт — в
[hw_01_segmentation/README.md](hw_01_segmentation/README.md).

Используется [MediaPipe Selfie Segmentation](https://chuoling.github.io/mediapipe/solutions/selfie_segmentation.html):
лёгкая CNN на основе MobileNetV3 с входом 256×256. Готовая модель включена в репозиторий,
инференс выполняется на CPU через TensorFlow Lite / XNNPACK. Она подходит для обработки
человека рядом с веб-камерой благодаря небольшому размеру и высокой скорости.
Маска строится на каждом кадре, сглаживается и используется для замены либо размытия фона.

Быстрый запуск из корня репозитория, **Python 3.11**, macOS/Linux:

```bash
cd hw_01_segmentation
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --camera 0 --width 640 --height 480 --mode color --compare
```

Windows PowerShell: создайте окружение через `py -3.11 -m venv .venv`,
активируйте `.\.venv\Scripts\Activate.ps1`; установка зависимостей и запуск те же.
Режимы фона: `--mode color`, `--mode blur`,
`--mode image --background background.jpg`. Вместо камеры: `--video input.mp4`.
Выход — Q/Esc, отчёт сохраняется в `results/local.json`.

Замеры на **Apple M3 Pro**, macOS 27.0.1, Python 3.11.9, CPU, обработка **640×480**:

| Фон | FPS обработки | Средняя latency | p95 latency |
| --- | ---: | ---: | ---: |
| Цвет | 115.8 | 8.64 мс | 10.12 мс |
| Размытие | 108.5 | 9.22 мс | 10.16 мс |

Проверено 300 кадров после 10 кадров прогрева на `vtest.avi` из OpenCV (вход 768×576),
без окна и ограничения скорости. FPS = число завершённых кадров / время обработки,
включая сегментацию, сглаживание маски и композитинг. Это скорость обработки файла,
не частота веб-камеры. Условия, команды повторного замера и JSON находятся в полном отчёте.

На сложном тестовом изображении люди выделяются, но остаются ошибки на границах
и ложное включение части постороннего объекта. Для видео предусмотрено сглаживание
маски. При проверке с веб-камерой получены 93 FPS обработки и 28 FPS полного цикла.
Качество приемлемое, но заметны артефакты вокруг головы.

**Демо:** [видео на Google Drive](https://drive.google.com/file/d/1WieQmzRXcnuXWvDFZhwfewDLNvrdlmpk/view?usp=sharing).

---

# Автоматический подбор музыки для визуального контента

Проект посвящён разработке системы автоматического подбора музыкального сопровождения для коротких видео с учётом пользовательского текстового запроса.

Модель получает на вход:

- видеофайл;
- название или текстовое описание видео;
- пользовательский запрос к музыке, например:  
  `"спокойная меланхоличная музыка"`,  
  `"энергичная музыка для спортивного видео"`,  
  `"напряжённый cinematic soundtrack"`.

На выходе система возвращает top-K музыкальных треков из фиксированного каталога, отсортированных по score релевантности.
Каталог аудио доступен по [ссылке](https://drive.google.com/file/d/11s8irMyChZc04JjCsVsdDC6QBI9x0tEq/view?usp=sharing)



Основные блоки модели:

- **Text block** — объединяет эмбеддинги `user_query` и `video_title`;
- **Video block** — проецирует видеоэмбеддинг в общее пространство;
- **Duration block** — кодирует длительность видео;
- **Fusion block** — объединяет текстовые, визуальные и числовые признаки;
- **Multilabel output head** — выдаёт logits по всем `track_id`.

Для обучения используется `BCEWithLogitsLoss`.


# Установка и активация виртуального окружения:
python -m venv venv
venv/Scripts/activate

Установка зависимостей:
pip install -r requirements.txt

Работы велись на Python 3.14

# Запуск инференса
Для ручного инференса используется скрипт: manual_inference.py

Он считает эмбеддинги с нуля:

кодирует user_query;
кодирует video_title;
кодирует видео;
подаёт признаки в обученную multilabel-модель;
возвращает top-K track_id.

Запуск в одну строку:

python manual_inference.py --video_path "OpenLAV/videos/408_Felix_Jumps_From_The_Stratosphere_-_Earth_Lab.mp4" --video_title "Прыжок из стратосферы" --user_query "Мне нужна напряженная музыка" --checkpoint_path "model_weights/best_target_binary_multilabel_model_inference.pt" --video_model_name "MCG-NJU/videomae-base-finetuned-kinetics" --top_k 10 --output_csv manual_inference_multilabel_predictions.csv
