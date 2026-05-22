from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def _normalize_rel_path(path: object) -> str:
    """Нормализует путь для сравнения строк из CSV."""
    return str(path).replace("\\", "/").strip().lower()


def load_existing_train_val_test_csv(
    train_csv_path: str,
    val_csv_path: str,
    test_csv_path: str,
    query_col: str = "user_query",
    title_col: str = "video_title",
    video_path_col: str = "video_path",
    duration_col: str = "duration_s",
    label_col: str = "track_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Читает уже готовые train / val / test CSV.
    Никакого нового split здесь не делается.

    Добавляет колонку track_label: числовой id класса для track_id.
    """

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    required_cols = [
        query_col,
        title_col,
        video_path_col,
        duration_col,
        label_col,
    ]

    for name, df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"{name}.csv missing columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

    for df in [train_df, val_df, test_df]:
        df[query_col] = df[query_col].fillna("").astype(str)
        df[title_col] = df[title_col].fillna("").astype(str)
        df[video_path_col] = df[video_path_col].astype(str)
        df[duration_col] = df[duration_col].astype(float)
        df[label_col] = df[label_col].astype(str)

    # Берём union по train/val/test, чтобы в val/test не было неизвестных индексов.
    all_track_ids = sorted(
        set(train_df[label_col].unique())
        | set(val_df[label_col].unique())
        | set(test_df[label_col].unique())
    )

    track2idx = {track_id: idx for idx, track_id in enumerate(all_track_ids)}
    idx2track = {idx: track_id for track_id, idx in track2idx.items()}

    for df in [train_df, val_df, test_df]:
        df["track_label"] = df[label_col].map(track2idx).astype(int)

    train_tracks = set(train_df[label_col].unique())
    val_tracks = set(val_df[label_col].unique())
    test_tracks = set(test_df[label_col].unique())

    unseen_val = sorted(val_tracks - train_tracks)
    unseen_test = sorted(test_tracks - train_tracks)

    if unseen_val:
        print(f"WARNING: {len(unseen_val)} track_id from val are absent in train.")

    if unseen_test:
        print(f"WARNING: {len(unseen_test)} track_id from test are absent in train.")

    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Test size:", len(test_df))
    print("Number of track classes:", len(track2idx))

    return train_df, val_df, test_df, track2idx, idx2track


def remove_videos_by_path(
    df: pd.DataFrame,
    bad_video_paths: Iterable[str],
    video_path_col: str = "video_path",
) -> pd.DataFrame:
    """
    Удаляет строки с конкретными video_path.
    Полезно для единичных битых файлов.
    """
    bad_set = {_normalize_rel_path(p) for p in bad_video_paths}
    mask = ~df[video_path_col].map(_normalize_rel_path).isin(bad_set)

    removed = int((~mask).sum())
    clean_df = df.loc[mask].reset_index(drop=True)

    print(f"Removed rows: {removed}")
    print(f"Remaining rows: {len(clean_df)}")

    return clean_df


def is_readable_video(
    video_path: Path,
    min_size_bytes: int = 1024,
) -> bool:
    """
    Быстрая проверка, что видео существует, не слишком маленькое и читается OpenCV.
    cv2 импортируется только внутри функции, чтобы не ломать обычную загрузку CSV.
    """
    if not video_path.exists():
        return False

    if video_path.stat().st_size < min_size_bytes:
        return False

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Для проверки видео установи opencv-python: pip install opencv-python"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return False

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    success, frame = cap.read()
    cap.release()

    return bool(frame_count > 0 and success and frame is not None)


def remove_unreadable_videos(
    df: pd.DataFrame,
    video_root: str,
    video_path_col: str = "video_path",
    min_size_bytes: int = 1024,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Удаляет из датафрейма строки с отсутствующими или нечитаемыми видео.
    Возвращает очищенный df и список плохих путей.
    """
    video_root_path = Path(video_root)
    good_indices: List[int] = []
    bad_paths: List[str] = []

    # Проверяем уникальные пути один раз.
    path_status: Dict[str, bool] = {}

    for rel_path in df[video_path_col].astype(str).unique():
        path = Path(rel_path.replace("\\", "/"))
        full_path = path if path.is_absolute() else video_root_path / path
        path_status[rel_path] = is_readable_video(full_path, min_size_bytes=min_size_bytes)

        if not path_status[rel_path]:
            bad_paths.append(rel_path)

    for idx, row in df.iterrows():
        rel_path = str(row[video_path_col])
        if path_status.get(rel_path, False):
            good_indices.append(idx)

    clean_df = df.loc[good_indices].reset_index(drop=True)

    print("Original rows:", len(df))
    print("Clean rows:", len(clean_df))
    print("Removed rows:", len(df) - len(clean_df))

    if bad_paths:
        print("Bad videos:")
        for path in bad_paths:
            print(" -", path)

    return clean_df, bad_paths
