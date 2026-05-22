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