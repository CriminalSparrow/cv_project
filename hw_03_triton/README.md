# ДЗ 3. Инференс модели

MediaPipe Selfie Segmentation из ДЗ 1 запущена в Triton через Python backend на CPU.
Вход `IMAGE` — RGB, UINT8, [256,256,3]; выход `MASK` — FP32, [256,256], вероятность
переднего плана. Веса включены в репозиторий. Запросы независимы, без сглаживания
по предыдущему кадру. Batching отключён: выбранный MediaPipe API обрабатывает один кадр.

## Запуск

Нужны Docker, Compose v2+ и доступ к NGC/PyPI для сборки. Основная платформа —
Linux x86_64; на Apple Silicon используется эмуляция. GPU не требуется.
Образ `23.12-py3` выбран для совместимости с NumPy 1.26.4.

Из корня репозитория:

```bash
cd hw_03_triton
docker compose up --build -d --wait server
```

HTTP: `localhost:8000`, gRPC: `localhost:8001`, метрики: `localhost:8002/metrics`.
Контейнеры соединены сетью Compose. Проверка готовности и остановка:

```bash
curl --fail http://localhost:8000/v2/health/ready
docker compose down
```

У успешного health-запроса код 200 и пустое тело.

## Запросы и метрики

```bash
docker compose run --rm client clients/infer.py --url server:8000 --check-reference
docker compose run --rm client clients/infer.py --protocol grpc --url server:8001 --check-reference
docker compose run --rm client clients/check_metrics.py --url server:8001 --metrics-url http://server:8002/metrics
```

Клиент отправляет по три запроса с `examples/person.jpg`, проверяет размер и диапазон
маски. `--check-reference` сравнивает ответ с прямым MediaPipe. Маска, изображение
с заменённым фоном и JSON сохраняются в `results/local_client/`.

В Python backend добавлены две метрики:

- `segmentation_processing_seconds_total` (COUNTER) — суммарное время обработки
  внутри модели, без сети и очереди Triton. Измеряется через `perf_counter`.
- `segmentation_requests_in_flight` (GAUGE) — число исполняемых запросов.
  Обновляется перед обработкой и в `finally`, в том числе при ошибке.

У каждого экземпляра своя серия с меткой `instance`; для значения по модели
серии суммируются. Проверку метрик запускают без других клиентов.

## Профилирование

Performance Analyzer запускается в отдельном SDK-контейнере:

```bash
docker compose run --rm perf
```

HTTP, batch size 1, concurrency 1–8, нулевой UINT8-тензор 256×256×3.
Окно измерения 3000 мс, допуск стабильности 10%, критерий — p95.
Новый результат сохраняется в `results/local_perf.csv`, лог — в `results/local_perf.log`.
Чтение JPEG и изменение размера на клиенте в замер не входят.

Для Model Analyzer основной сервер останавливается:

```bash
docker compose stop server
docker compose run --rm analyzer
```

В `profiling/config.yaml` проверяются 1, 2 и 4 CPU-экземпляра при concurrency
1, 2, 4 и 8. Цель — максимальный throughput при p99 ≤ 50 мс. Это выбранный
предел задержки для интерактивной обработки, не требование задания.

Analyzer установлен в серверный образ и запускает Triton в режиме `local`,
поэтому доступны зависимости MediaPipe. Для нагрузки создаётся JSON с нулевым
тензором: сокращение `input-data=zero` вызывает ошибку в Model Analyzer 1.36.
Отчёты и checkpoint сохраняются в `results/local_analyzer/`; перед независимым
повторным замером этот каталог нужно перенести или удалить.

## Результаты

Замеры 09.10.2026: Apple M3 Pro, macOS 27.0.1, Docker 29.1.3, linux/amd64
под эмуляцией. Docker VM: 12 CPU и 8.22 ГБ RAM. Triton 2.43.0, Python 3.10.12,
MediaPipe 0.10.21, NumPy 1.26.4, Performance Analyzer 2.42.0, Model Analyzer 1.36.0.

NGC вернул 403 при загрузке образов 23.12, поэтому замеры выполнены на имевшемся
локальном образе Triton 2.43.0. Performance Analyzer запускался в отдельном
контейнере из пакета NVIDIA Model Analyzer. 
Повторить локальный вариант:

```bash
docker compose build --build-arg TRITON_IMAGE=triton-load-rig:deps server
docker compose up -d --wait server
PERF_IMAGE=hw03-triton:local docker compose run --rm perf
```

Этот локальный тег не доступен для скачивания; обычный запуск требует доступа к NGC.

HTTP и gRPC: все 6 ответов совпали с прямым MediaPipe.
На 1280 запросах от четырёх клиентов COUNTER вырос на 9.119 с, GAUGE прошёл
0 → 1 → 0. Для выбранного профиля с четырьмя экземплярами также проверены
3 точных ответа и 4477 запросов под нагрузкой: GAUGE 0 → 4 → 0,
COUNTER вырос на 34.942 с. Сумма времени параллельных обработчиков может
превышать время по часам.

### Performance Analyzer

Один экземпляр модели. Исходные данные — [results/perf.csv](results/perf.csv).
В CSV задержки указаны в микросекундах, в таблице — в миллисекундах.

| Concurrency | Запросов/с | p95, мс | p99, мс |
| ---: | ---: | ---: | ---: |
| 1 | 128.157 | 8.635 | 9.355 |
| 2 | 134.410 | 16.036 | 16.951 |
| 3 | 135.563 | 23.845 | 24.839 |
| 4 | 135.197 | 31.679 | 33.215 |
| 5 | 135.106 | 39.639 | 41.199 |
| 6 | 133.697 | 47.704 | 52.278 |
| 7 | 136.334 | 54.668 | 57.045 |
| 8 | 134.276 | 64.227 | 68.928 |

При concurrency от 2 скорость почти не растёт: один экземпляр уже загружен.
Compute infer остаётся около 7.2–7.4 мс, а средняя очередь растёт с 0.026
до 51.665 мс. Повышение concurrency увеличивает задержку, почти не давая throughput.

### Model Analyzer

Проверены 16 точек: три варианта числа экземпляров и исходный конфиг,
который также содержит один экземпляр. Полные данные —
[results/model_analyzer.csv](results/model_analyzer.csv), задержки в нём в мс.
Лучшие точки с p99 ≤ 50 мс:

| Экземпляры CPU | Concurrency | Запросов/с | p99, мс |
| ---: | ---: | ---: | ---: |
| 1 | 4 | 138.1 | 32.4 |
| 2 | 4 | 265.5 | 16.6 |
| 4 | 8 | 494.0 | 20.1 |
| 4 | 4 | 475.4 | 9.9 |

Выбраны **4 экземпляра**: максимальная скорость в 3.58 раза выше, чем у одного.
Для меньшей задержки предпочтительна concurrency 4: переход к 8 даёт лишь 3.9%
throughput, но примерно удваивает p99. При concurrency 1 дополнительные
экземпляры не помогают — нужен поток одновременных запросов.

Для последней точки Analyzer увеличил окно до 4000 мс для стабильности.
Сбор CPU-метрик тоже вносит накладные расходы. На машине с меньшим числом CPU
выбор конфигурации следует повторить.

Запуск выбранной конфигурации из `profiling/best_config.pbtxt`:

```bash
docker compose -f compose.yaml -f compose.optimized.yaml up --build -d --wait server
```

При обращении через сервис `client` также указываются оба `-f`.
Обычный `compose.yaml` запускает один экземпляр для повторения базового замера.

## Источники

- [Модель Selfie Segmenter](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite).
- [Тестовое изображение MediaPipe](https://storage.googleapis.com/mediapipe-assets/segmentation_input_rotation0.jpg).
- [Документация Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/contents.html).
