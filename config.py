import json
import os
from pathlib import Path

from platformdirs import user_config_dir

DEFAULT_MODEL = "hotchpotch/bekko-static-embedding-v1"
MODEL_ENV = "ARXIMEDES_MODEL"
CONFIG_FILE = Path(user_config_dir("arximedes")) / "config.json"


def _load() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text())
    except FileNotFoundError:
        return {}


def _save(cfg: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2) + "\n")


def get_model() -> tuple[str, str]:
    """The embedding model to use and where the choice came from: env var > config file > default."""
    if os.environ.get(MODEL_ENV):
        return os.environ[MODEL_ENV], f"${MODEL_ENV}"
    saved = _load().get("model")
    if saved:
        return saved, str(CONFIG_FILE)
    return DEFAULT_MODEL, "default"


def set_model(name: str | None) -> None:
    """Save the model to the config file; None removes it so the default applies."""
    cfg = _load()
    if name:
        cfg["model"] = name
    else:
        cfg.pop("model", None)
    _save(cfg)
