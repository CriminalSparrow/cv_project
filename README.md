# Подбор музыки к видео и текстовому запросу

Исходная multilabel-модель из [отчёта](Отчет%202%20семестр%20CV.pdf):
E5 кодирует пользовательский запрос и название видео, VideoMAE кодирует видео,
fusion-блок учитывает эти признаки и длительность, а выходная голова ранжирует
фиксированный каталог треков. Обучение использует BCEWithLogitsLoss.

## Основные файлы

- `stage_1_data_preparation_and_EDA.ipynb` — загрузка исходных данных,
  формирование пар и текстовых запросов, EDA.
- `final_dataset_with_queries/` — зафиксированные train/val/test, шаблоны запросов
  и матрица совместимости. Сохранены точные исходные CSV.
- `train/` — архитектура, подготовка признаков, загрузчики, метрики и ноутбук обучения.
- `inference/`, `common_utils/`, `manual_inference.py` — загрузка и запуск модели.
- `model_weights/best_target_binary_multilabel_model_inference.pt` — финальный
  обученный checkpoint со словарём треков; сохранён в Git.
- `SHA256SUMS` — контрольные суммы данных, checkpoint и единственного отчёта.

## Восстановить на Mac

```bash
git clone https://github.com/CriminalSparrow/cv_project.git
cd cv_project
shasum -a 256 -c SHA256SUMS
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos-py312.txt
```

Прямые зависимости Mac зафиксированы по удалённому локальному окружению.
Полная переустановка и повторное обучение при очистке не запускались.
`requirements.txt` сохраняет исходные версии обучения, включая CUDA-сборку PyTorch;
для неё нужен соответствующий индекс пакетов из [инструкции PyTorch](https://pytorch.org/get-started/locally/).

## Предобученные модели

Веса энкодеров загружаются отдельно:

- [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3).
- [MCG-NJU/videomae-base-finetuned-kinetics](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics/tree/488eb9a0565f257b32866000305c8178965eb9f6).

Это ревизии, найденные в локальном кеше перед очисткой. Исходный код по имени
модели обращается к текущей версии; для зафиксированных весов передайте локальные
пути после загрузки конкретных ревизий:

```bash
source .venv/bin/activate
TEXT_MODEL="$(hf download intfloat/multilingual-e5-small --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3)"
VIDEO_MODEL="$(hf download MCG-NJU/videomae-base-finetuned-kinetics --revision 488eb9a0565f257b32866000305c8178965eb9f6)"
python manual_inference.py \
  --video_path /path/to/video.mp4 \
  --video_title "Название видео" \
  --user_query "Мне нужна спокойная музыка" \
  --checkpoint_path model_weights/best_target_binary_multilabel_model_inference.pt \
  --project_root . \
  --text_model_name "$TEXT_MODEL" \
  --video_model_name "$VIDEO_MODEL" \
  --top_k 10
```

## Исходные данные и повторная подготовка

- [OpenLAV, официальный источник и video_database.csv](https://www.psycharchives.org/en/item/18779e98-c04b-4299-8311-dc442dc89bcd).
  Для первого ноутбука сохраните исходную таблицу как `OpenLAV/video_database.csv`.
  Он скачивает видео по `source_URL` и обрезает их по `length_s`; нужны `yt-dlp`
  и FFmpeg в PATH.
- [MTG-Jamendo, источник из ноутбука](https://huggingface.co/datasets/rkstgr/mtg-jamendo).
  Ноутбук использует metadata split `valid` и архивы `data/val/`.
- [Сохранённый каталог аудио](https://drive.google.com/file/d/11s8irMyChZc04JjCsVsdDC6QBI9x0tEq/view).

Готовая разметка в `final_dataset_with_queries/` сохраняет исходное разбиение.
При воспроизведении используйте её, чтобы не менять протокол оценки. Ссылки
исходных видеороликов могут со временем перестать работать.

Предвычисленные эмбеддинги и их JSON-индексы создаёт
`train/precompute_embeddings.py`; порядок обучения сохранён в
`train/stage_2_training_models.ipynb`. Графики EDA пересоздаются первым ноутбуком.
Ноутбуки содержат исторические локальные пути: задайте пути к восстановленным
данным и запускайте подготовку из корня репозитория.
