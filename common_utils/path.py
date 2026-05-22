from pathlib import Path
from typing import Optional, Union


def resolve_path(
    path: Union[str, Path],
    root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Разрешает путь.

    Если path абсолютный — возвращает его.
    Если относительный и root задан — возвращает root / path.
    """
    path = Path(str(path).replace("\\", "/"))

    if path.is_absolute():
        return path

    if root is not None:
        return Path(root) / path

    return path.resolve()