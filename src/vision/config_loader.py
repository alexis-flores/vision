"""
config_loader.py
FR-001: ingest camera configuration from a configuration file.

Supports JSON natively and YAML if PyYAML is installed. The feature mask
is given as a list of feature names so the file stays human-editable:

    "features": ["GAIN", "EXPOSURE", "FRAME_RATE", "RESOLUTION"]

A missing or malformed file raises ConfigError (logged at ERROR per 5.5).
"""

from __future__ import annotations

import json
import logging
import os
from functools import reduce
from operator import or_
from typing import Any, Dict, List

from .camera_types import CameraConfig, CameraFeature, PixelFormat

log = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration file missing, unreadable, or invalid."""


def _parse_features(names: List[str]) -> CameraFeature:
    if not names:
        return CameraFeature.NONE
    flags = []
    for n in names:
        try:
            flags.append(CameraFeature[n.upper()])
        except KeyError:
            raise ConfigError(f"Unknown camera feature {n!r}") from None
    return reduce(or_, flags, CameraFeature.NONE)


def _load_raw(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise ConfigError(f"Configuration file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            if ext in (".yaml", ".yml"):
                try:
                    import yaml
                except ImportError as e:
                    raise ConfigError(
                        "PyYAML required for YAML config files") from e
                try:
                    return yaml.safe_load(fh) or {}
                except yaml.YAMLError as e:  # NOT a ValueError subclass
                    raise ConfigError(
                        f"Malformed configuration file {path}: {e}") from e
            return json.load(fh)
    except (json.JSONDecodeError, ValueError) as e:
        raise ConfigError(f"Malformed configuration file {path}: {e}") from e


def load_camera_config(path: str) -> CameraConfig:
    """Build a single CameraConfig from a config file (top-level object)."""
    raw = _load_raw(path)
    return _config_from_dict(raw)


def load_camera_configs(path: str) -> List[CameraConfig]:
    """
    Load one or more camera configs. Accepts either a single config object
    or {"cameras": [ {...}, {...} ]} for multi-camera rigs.

    Camera names must be unique: the service keys cameras by name and rejects
    a duplicate at registration, so we catch it here (as a ConfigError that
    callers already handle) rather than letting it surface as a raw ValueError
    mid-registration with the service half-populated.
    """
    raw = _load_raw(path)
    if isinstance(raw, dict) and "cameras" in raw:
        configs = [_config_from_dict(c) for c in raw["cameras"]]
    else:
        configs = [_config_from_dict(raw)]
    seen: set[str] = set()
    for cfg in configs:
        if cfg.name in seen:
            raise ConfigError(
                f"Duplicate camera name {cfg.name!r} in {path}; "
                f"each camera must have a unique 'name'")
        seen.add(cfg.name)
    return configs


def _config_from_dict(d: Dict[str, Any]) -> CameraConfig:
    data = dict(d) # shallow copy; we pop keys that need transforming
    try:
        features = _parse_features(data.pop("features", []))
        pixfmt_name = data.pop("output_pixel_format", None)
        output_pixel_format = (PixelFormat(pixfmt_name) if pixfmt_name
                               else PixelFormat.BGR8)
        for tup_key in ("max_resolution", "resolution"):
            if tup_key in data and data[tup_key] is not None:
                data[tup_key] = tuple(data[tup_key])
        cfg = CameraConfig(features=features,
                           output_pixel_format=output_pixel_format, **data)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"Invalid configuration field: {e}") from e
    log.info("Loaded camera config %r (%s)", cfg.name, cfg.model)
    return cfg
