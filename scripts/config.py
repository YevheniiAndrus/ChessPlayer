#!/usr/bin/env python3
"""
config.py

Loads config.yaml (at the project root) into a plain nested dict, with
every path in its `paths` section resolved to an absolute path (or, for
glob patterns, an absolute-prefixed pattern string) relative to
config.yaml's own location -- so every script gets consistent paths no
matter which directory it's actually run from.

Requires:
    pip install pyyaml

Usage:
    from config import load_config

    cfg = load_config()
    cfg["data"]["seq_len"]
    cfg["paths"]["vocab"]              # absolute pathlib.Path
    cfg["paths"]["train_tfrecord_pattern"]   # absolute-prefixed glob string
    cfg["model_defaults"]["d_model"]
    cfg["tuner_search_space"]["learning_rate"]
"""

from pathlib import Path
from typing import Any, Dict, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _looks_like_a_glob_pattern(value: str) -> bool:
    return any(ch in value for ch in "*?[]")


def load_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = config_path.resolve().parent
    paths = cfg.get("paths", {})
    for key, value in paths.items():
        if _looks_like_a_glob_pattern(value):
            # Glob patterns (e.g. "shard-0000[0-8]*.tfrecord") need to stay
            # strings usable by tf.data.Dataset.list_files -- just make the
            # directory portion absolute.
            paths[key] = str(base_dir / value)
        else:
            paths[key] = base_dir / value

    return cfg


def sample_hyperparameter(hp, search_space: Dict[str, Any], name: str):
    """Sample one hyperparameter from a keras_tuner HyperParameters object
    (`hp`) according to its entry in config.yaml's tuner_search_space, so
    hypermodel.py doesn't need to know the mechanics of each range type --
    it just asks for a hyperparameter by name."""
    spec = search_space[name]
    kind = spec["type"]

    if kind == "choice":
        return hp.Choice(name, spec["values"], default=spec.get("default"))
    if kind == "int":
        return hp.Int(
            name,
            min_value=spec["min"],
            max_value=spec["max"],
            step=spec.get("step", 1),
            default=spec.get("default"),
        )
    if kind == "float":
        return hp.Float(
            name,
            min_value=spec["min"],
            max_value=spec["max"],
            step=spec.get("step"),
            sampling=spec.get("sampling", "linear"),
            default=spec.get("default"),
        )
    if kind == "boolean":
        return hp.Boolean(name, default=spec.get("default", False))

    raise ValueError(f"Unknown hyperparameter type {kind!r} for {name!r} in tuner_search_space.")


if __name__ == "__main__":
    # Quick manual check: python config.py [path/to/config.yaml]
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    loaded = load_config(path)
    print(json.dumps(loaded, indent=2, default=str))
