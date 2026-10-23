"""Config loading: configs/base.yaml deep-merged with an optional model file, then --set key.sub=value overrides."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs" / "base.yaml"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(item: str) -> dict:
    """'train.lr=3e-4' -> {'train': {'lr': 0.0003}}. Values are parsed as YAML (lists, numbers, booleans)."""
    key, sep, raw = item.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"--set expects key.sub=value, got {item!r}")
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:  # PyYAML reads '3e-4' as a string
            value = float(value)
        except ValueError:
            pass
    for part in reversed(key.strip().split(".")):
        value = {part: value}
    return value


def resolve_file(path: str | Path) -> Path:
    """A path given on the command line: as given if it exists, else relative to the project root."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p.resolve()
    return (ROOT / p).resolve()


def load_config(model_config: str | Path | None = None, overrides=(), base: str | Path = BASE_CONFIG) -> dict:
    cfg = yaml.safe_load(Path(base).read_text())
    if model_config:
        path = resolve_file(model_config)
        cfg = deep_merge(cfg, yaml.safe_load(path.read_text()) or {})
        try:
            cfg["config_file"] = path.relative_to(ROOT).as_posix()
        except ValueError:
            cfg["config_file"] = str(path)
    for item in overrides:
        cfg = deep_merge(cfg, parse_override(item))
    return cfg


def config_hash(obj, n: int = 10) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:n]


def project_path(cfg: dict, key: str) -> Path:
    """cfg['paths'][key] resolved against the project root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def add_config_args(parser: argparse.ArgumentParser, model_config: bool = True) -> None:
    if model_config:
        parser.add_argument("--config", help="model config, e.g. configs/models/tcn.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set train.lr=3e-4 (repeatable)")


def config_from_args(args: argparse.Namespace) -> dict:
    return load_config(getattr(args, "config", None), args.set)
