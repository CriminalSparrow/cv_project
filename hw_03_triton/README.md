# ДЗ №3. Инференс модели в Triton

В Triton перенесена **MediaPipe Selfie Segmentation из ДЗ 1**. Модель выделяет
человека на изображении. Веса включены в папку; обучение и загрузка весов при
запуске не нужны. Решение автономно: папку `hw_03_triton` можно скопировать отдельно.

## Архитектура

Выбран Python backend: MediaPipe уже предоставляет CPU-инференс TFLite/XNNPACK,
а `triton_python_backend_utils` позволяет добавить метрики прямо в модель.
BLS-прокси не требуется. Один экземпляр модели создаёт один ImageSegmenter
при загрузке и переиспользует его для запросов.

Контракт API: `IMAGE`, `UINT8`, RGB, **[256,256,3]** → `MASK`, `FP32`, **[256,256]**,
вероятность переднего плана от 0 до 1. У запроса нет batch-измерения:
`max_batch_size: 0`. Клиент изменяет размер изображения перед отправкой.
Маску можно использовать для замены фона; клиент сохраняет пример результата.

В отличие от видеоприложения ДЗ 1, сервер не хранит предыдущую маску:
каждый запрос независим. Сглаживание во времени смешивало бы кадры разных клиентов.
Модель не поддерживает нативную пакетную обработку в выбранном MediaPipe API,
поэтому dynamic batching не включён. Исследуется параллелизм через CPU instances.

## Запуск

Нужны Docker и Docker Compose v2+, доступ к NGC (`nvcr.io`) и PyPI во время сборки.
Основная целевая платформа — **Linux x86_64**, CPU; NVIDIA GPU не требуется.
На Apple Silicon используется `linux/amd64` и эмуляция Docker Desktop:
производительность нельзя переносить на обычный Linux-сервер.
Образы NVIDIA большие: предусмотрите свободное место для базового образа и SDK.

Из корня репозитория:

```bash
cd hw_03_triton
docker compose up --build -d --wait server
```

Это одна команда сборки Dockerfile и запуска сервера. В Dockerfile по умолчанию
`nvcr.io/nvidia/tritonserver:23.12-py3`: NumPy остаётся в ветке 1.x.
Порты опубликованы только на localhost: HTTP **8000**, gRPC **8001**, метрики **8002**.
Контейнеры общаются через отдельную сеть Compose; `--net host` не требуется.

```bash
curl --fail http://localhost:8000/v2/health/ready
curl --fail http://localhost:8000/v2/models/selfie_segmentation
docker compose logs server
```

HTTP 200 у health endpoint имеет пустое тело — это нормальный результат.
Если порты заняты, освободите их или измените только левую часть port mapping
в `compose.yaml`; адреса между контейнерами останутся прежними.

Остановка созданных контейнеров:

```bash
docker compose down
```

## Тестовые запросы

Используются зависимости из собранного образа, установка Python на хост не нужна:

```bash
docker compose run --rm client clients/infer.py --url server:8000 --requests 3 --check-reference --output results/http
docker compose run --rm client clients/infer.py --protocol grpc --url server:8001 --requests 3 --check-reference --output results/grpc
```

Оба вызова проверяют форму, тип, конечность и диапазон маски. `--check-reference`
дополнительно сравнивает маску с прямым вызовом MediaPipe с теми же весами
(`atol=rtol=1e-5`). Результаты сохраняются на хосте: `requests.json`, `mask.npy`,
`mask.png`, `comparison.png` (слева вход, справа заменённый фон).

Для своего изображения с хоста (Python 3.10/3.11):

```bash
python -m pip install -r clients/requirements.txt
python clients/infer.py --image /path/to/photo.jpg --protocol http
python clients/infer.py --image /path/to/photo.jpg --protocol grpc
```

`--check-reference` требует дополнительно MediaPipe; внутри Docker он уже установлен.
Источник примера и весов указан в [examples/SOURCE.md](examples/SOURCE.md).

## Кастомные метрики

```bash
curl --fail http://localhost:8002/metrics
docker compose run --rm client clients/check_metrics.py --url server:8001 --metrics-url http://server:8002/metrics --output results/metrics.json
```

Проверку запускают без других клиентов. Скрипт отправляет реальные параллельные
gRPC-запросы и опрашивает `/metrics`: COUNTER должен вырасти без уменьшений,
GAUGE — стать положительным во время обработки и вернуться к нулю.
Полный ряд наблюдений сохраняется в JSON.

| Метрика | Тип | Семантика |
| --- | --- | --- |
| `segmentation_processing_seconds_total` | COUNTER | Накопленное время обработки запросов внутри Python backend, секунды |
| `segmentation_requests_in_flight` | GAUGE | Число запросов, исполняемых сейчас внутри Python backend |

Метки `model`, `version`, `instance` различают экземпляры. Для значения по модели
нужно суммировать серии: `sum(segmentation_requests_in_flight{model="selfie_segmentation"})`.
Каждый синхронный экземпляр даёт GAUGE 0 или 1; при четырёх экземплярах сумма до 4.
Ожидание в очереди Triton не считается обработкой. COUNTER использует монотонный
`perf_counter`, включает преобразование тензора, сегментацию и построение ответа,
но не сетевую передачу и очередь. `finally` обновляет время и уменьшает GAUGE
даже при ошибке обработки. Отклонённые самим Triton запросы до `execute` сюда не входят.
При перезагрузке модели серии сбрасываются; в Prometheus для скорости используют `rate()`.

## Performance Analyzer

Сначала запустите `server`, затем отдельный SDK-контейнер в той же сети:

```bash
docker compose run --rm perf
```

Команда внутри контейнера находится в [scripts/perf.sh](scripts/perf.sh):
`perf_analyzer -m selfie_segmentation`, HTTP, concurrency **1–8**, batch size 1,
окна по 3000 мс, допуск стабильности 10%, максимум 10 попыток, критерий latency p95.
CSV и полный stdout: `results/perf/baseline.csv` и `baseline.log`.
`set -o pipefail` не позволяет выдать неудачный прогон за успешный.

Для нагрузки выбран фиксированный нулевой UINT8-тензор. Это измерение инференса
и передачи тензоров, без чтения JPEG, изменения размера и композитинга на клиенте.
Корректность на настоящем изображении проверяется отдельными запросами выше.
Throughput — завершённые инференсы/сек; latency клиента включает передачу данных,
очередь и исполнение. Это другие величины, чем FPS полного приложения из ДЗ 1.

## Model Analyzer

Остановите основной сервер, чтобы он не конкурировал за CPU:

```bash
docker compose stop server
docker compose run --rm analyzer
```

В [profiling/config.yaml](profiling/config.yaml) задан конечный поиск:
**1, 2, 4 экземпляра CPU × concurrency 1, 2, 4, 8**. Batch size всегда 1.
Цель — максимальный throughput при **p99 ≤ 50 мс**. Это выбранный для поиска
практический предел задержки, а не требование задания. Он предотвращает выбор
конфигурации, достигающей скорости ценой слишком длинной очереди. Расход памяти
также нужно учитывать при выборе. Автоматический широкий поиск отключён, потому что
batching для этого интерфейса не применим, а число доступных CPU ограничено.

`triton-model-analyzer` установлен в серверный образ. `triton_launch_mode: local`
запускает дочерний `tritonserver` внутри этого контейнера с теми же зависимостями
MediaPipe. Docker socket и запуск чистого образа без зависимостей не используются.
`cpu_only: true` отключает необходимость GPU для профилируемой модели.
Перед запуском скрипт создаёт `results/input-data-0.json` с тем же нулевым
UINT8-изображением, что в отдельном Performance Analyzer. Явный JSON нужен из-за
ошибки парсера Model Analyzer 1.36 при передаче сокращения `input-data: zero`.

Результаты: `results/model_analyzer/` (CSV, отчёты и логи), `results/configs/`
(варианты конфигов), `results/checkpoints/` (возобновление). Повторный запуск
может использовать checkpoint. Для нового независимого замера переместите эти
три каталога в архив перед запуском. Исходный `config.pbtxt` не изменяется.

## Результаты и ограничения проверки

### Среда фактического прогона, 09.10.2026

Apple M3 Pro, macOS 27.0.1, Docker Engine 29.1.3; Linux VM: 12 CPU,
8 217 165 824 байт RAM. Контейнер **linux/amd64 под эмуляцией**, CPU, без GPU.
Triton **2.43.0**, Python **3.10.12**, NumPy **1.26.4**, MediaPipe **0.10.21**,
Model Analyzer **1.36.0**, Performance Analyzer **2.42.0**.
Это обычный рабочий компьютер, а не изолированный стенд; результаты не являются
оценкой производительности native x86_64 или GPU.

**Ограничение воспроизводимости в текущей сети:** загрузка обоих образов NGC
`23.12-py3` и `23.12-py3-sdk` вернула HTTP **403 Forbidden**. Поэтому фактическая
сборка использовала уже имевшийся локальный образ `triton-load-rig:deps` с Triton
2.43.0 и NumPy 1.26.4. Performance Analyzer запускался в **отдельном контейнере**
`hw03-triton:local`, из wheel NVIDIA `triton-model-analyzer`. Стандартный путь
с образами NGC 23.12 сохранён в Dockerfile/Compose, но **целиком здесь не проверен**.
Для повторения именно локального варианта нужны те же заранее загруженные образы:

```bash
docker compose build --build-arg TRITON_IMAGE=triton-load-rig:deps server
docker compose up -d --wait server
PERF_IMAGE=hw03-triton:local docker compose run --rm perf
docker compose stop server
docker compose run --rm analyzer
```

`triton-load-rig:deps` — локальный тег этого компьютера, не публичный образ для
проверяющего. В другой среде используется стандартная команда из раздела запуска
с доступом к NGC. Подмена тегов NGC локальным образом не производилась.
Точные условия: [environment.json](results/environment.json),
[пакеты среды](results/environment_packages.json),
[метаданные работающего сервера](results/server_metadata.json).

### Корректность и метрики

- HTTP: 3 из 3 успешных ответов; gRPC: 3 из 3. Максимальная абсолютная разница
  с прямым MediaPipe **0.0** во всех шести запросах.
- Маска FP32 256×256, диапазон **[0,1]**, среднее **0.46766889**.
  Исходные ответы: [HTTP](results/http/requests.json), [gRPC](results/grpc/requests.json).
- Проверка метрик: **1280** успешных запросов от 4 клиентов за около 10 секунд.
  COUNTER: **0.075456 → 9.194851 с**, прирост **9.119394 с**.
  GAUGE: **0 → 1 → 0**; это ожидаемо для одного экземпляра, остальные запросы
  ждут в очереди. Проверены монотонность COUNTER и отсутствие отрицательного GAUGE.
  Полный ряд: [metrics.json](results/metrics.json).
- 3 unit-теста прошли: успешные запросы, ошибка инференса с продолжением обработки,
  неверный вход. Отдельно прошли проверка синтаксиса Python, Bash и конфигурации Compose.

![Исходное изображение и замена фона](results/http/comparison.png)

Визуально люди выделяются; часть лошади ошибочно остаётся в переднем плане.
Края волос и предметы около человека также могут сегментироваться неточно.
Это ограничение готовой модели Selfie Segmentation, а не расхождение серверного
инференса с исходной моделью.

### Performance Analyzer: один CPU-экземпляр

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

Источник: [baseline.csv](results/perf/baseline.csv),
[полный лог](results/perf/baseline.log). В CSV времена указаны в **микросекундах**;
таблица выше переводит их в миллисекунды. CSV отсортирован самим инструментом
по throughput, таблица — по concurrency.

Один экземпляр насыщается уже при concurrency около 2: дальнейшее увеличение
даёт примерно ту же скорость **134–136 запросов/с**. Среднее время compute infer
остаётся около **7.2–7.4 мс**, зато среднее ожидание в очереди растёт с **0.026 мс**
при concurrency 1 до **51.665 мс** при concurrency 8. Поэтому p95 растёт более
чем в 7 раз. Для одного экземпляра высокая concurrency ухудшает отзывчивость
почти без выигрыша в throughput; увеличить реальную параллельность можно
дополнительными экземплярами модели.

### Model Analyzer: сравнение конфигураций

Прогон завершён успешно: **4 конфигурации × 4 concurrency = 16 точек**.
Инструмент дополнительно проверяет исходный `config_default`; `config_0` тоже
имеет один CPU-экземпляр, так что уникальных вариантов числа экземпляров три.
В таблице — лучшая точка каждого варианта, удовлетворяющая p99 ≤ 50 мс:

| Экземпляры CPU | Конфигурация | Concurrency | Запросов/с | p99, мс |
| ---: | --- | ---: | ---: | ---: |
| 1 | `config_0` | 4 | 138.1 | 32.4 |
| 2 | `config_1` | 4 | 265.5 | 16.6 |
| **4** | **`config_2`** | **8** | **494.0** | **20.1** |
| 4, вариант с меньшей задержкой | `config_2` | 4 | 475.4 | 9.9 |

Источники: [все 16 точек CSV](results/model_analyzer/results/metrics-model-inference.csv),
[лог Model Analyzer](results/model_analyzer/run.log),
[полные выводы perf_analyzer](results/model_analyzer/perf.log),
[сводный PDF](results/model_analyzer/reports/summaries/selfie_segmentation/result_summary.pdf),
[подробный отчёт лучшего конфига](results/model_analyzer/reports/detailed/selfie_segmentation_config_2/detailed_report.pdf).
В CSV Model Analyzer задержка уже указана в **миллисекундах**.

Выбраны **4 CPU-экземпляра**: по лучшим допустимым точкам выигрыш относительно
одного составляет **3.58×**, относительно двух — **1.86×**. У каждого экземпляра
свой процесс Python и своя модель; это позволяет действительно исполнять запросы
параллельно. При concurrency 1 дополнительные экземпляры ускорения не дают:
121.1 запросов/с для четырёх против 129.9 для одного. Выигрыш появляется именно
при одновременной работе нескольких клиентов.

Для низкой задержки предпочтительна concurrency около **4**: переход к 8
добавляет всего **3.9%** throughput, но примерно удваивает p99 (**9.9 → 20.1 мс**).
Приоритет максимальной пропускной способности выбирает concurrency 8, и она
укладывается в заданный предел 50 мс. Это характеристика клиентской нагрузки,
а не настройка, которую сервер навязывает всем клиентам.

Для точки 4 экземпляра / concurrency 8 Model Analyzer увеличил измерительное
окно с 3000 до **4000 мс** для достижения стабильности. CPU-мониторинг в нём
включён и сам вносит накладные расходы; поэтому небольшое отличие от отдельного
Performance Analyzer ожидаемо. Показатели памяти стандартного отчёта не следует
принимать за полный расход всей группы процессов Python backend; выбор здесь
обоснован измеренными throughput и latency. На машине с меньшим числом CPU
следует повторить поиск: больше экземпляров не гарантирует ускорение.

Готовый выбранный конфиг: [profiling/best_config.pbtxt](profiling/best_config.pbtxt).
Запуск четырёх экземпляров одной командой:

```bash
docker compose -f compose.yaml -f compose.optimized.yaml up --build -d --wait server
```

Для уже собранного образа можно опустить `--build`. Проверка этого профиля:

```bash
docker compose -f compose.yaml -f compose.optimized.yaml run --rm client clients/infer.py --url server:8000 --check-reference
```

При работе с выбранным профилем сохраняйте оба `-f` и в командах клиентов.
Обычный `compose.yaml` намеренно оставляет один экземпляр, чтобы команда
Performance Analyzer из предыдущего раздела воспроизводила базовый замер.

Выбранный профиль дополнительно запущен через `compose.optimized.yaml`:
3 gRPC-ответа снова точно совпали с прямым MediaPipe
([результат](results/optimized/requests.json)). Проверка 8 параллельными клиентами
дала **4477** успешных запросов за около 10 секунд; сумма GAUGE достигла **4**
и вернулась к **0**, COUNTER вырос с **0 до 34.941570 с**
([полные метрики](results/metrics_optimized.json)). Сумма времени четырёх
параллельных обработчиков закономерно может превышать время по часам.
Эта функциональная проверка не подменяет замеры Performance Analyzer.

## Проверка кода

```bash
docker compose run --rm --no-deps client -m unittest discover -s tests -v
```

Тесты проверяют учёт времени, сброс GAUGE при ошибке, продолжение обработки
следующего запроса и отклонение неверных данных. Это дополнение к реальным
HTTP/gRPC-запросам, а не замена проверки запущенного Triton.

## Структура

- `Dockerfile`, `compose.yaml`, `requirements.txt` — среда и запуск.
- `model_repository/selfie_segmentation/` — конфигурация, Python backend и веса.
- `clients/` — HTTP/gRPC-примеры и проверка метрик под нагрузкой.
- `scripts/`, `profiling/` — запуск профилировщиков и пространство поиска.
- `examples/` — входное изображение и его источник.
- `results/` — фактические замеры и артефакты.
- `tests/` — проверки поведения backend при успехе и ошибках.

## Документация

- [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/contents.html).
- [Python backend и Custom Metrics, r23.12](https://github.com/triton-inference-server/python_backend/tree/r23.12#custom-metrics).
- [Настройки Model Analyzer, r23.12](https://github.com/triton-inference-server/model_analyzer/blob/r23.12/docs/config.md).
