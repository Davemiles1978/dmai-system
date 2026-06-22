"""Runtime component loader using importlib, directories, and zip archives."""

from __future__ import annotations

import importlib
import io
import json
import logging
import os
import zipfile
from typing import Any, Optional, Type

from dmai.registry.component_base import BaseComponent

logger = logging.getLogger("dmai.loader")

COMPONENTS_DIR = os.path.join(os.getcwd(), "components")


class ComponentLoader:
    """Loads :class:`BaseComponent` subclasses from various sources."""

    def load_from_entry_point(self, entry_point_str: str) -> Type[BaseComponent]:
        """Import a component class from a ``module.path:ClassName`` string."""
        if ":" not in entry_point_str:
            raise ValueError(f"Invalid entry point '{entry_point_str}' (expected module:Class)")
        module_path, class_name = entry_point_str.split(":", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        if not (isinstance(cls, type) and issubclass(cls, BaseComponent)):
            raise TypeError(f"{entry_point_str} is not a BaseComponent subclass")
        return cls

    def load_from_dir(self, path: str) -> tuple[dict[str, Any], Type[BaseComponent]]:
        """Load a component from a directory containing ``component.json``."""
        manifest_path = os.path.join(path, "component.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"No component.json in {path}")
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        entry_point = manifest.get("entry_point")
        if not entry_point:
            raise ValueError(f"component.json in {path} missing 'entry_point'")
        cls = self.load_from_entry_point(entry_point)
        return manifest, cls

    def load_from_zip(self, zip_bytes: bytes, target_dir: Optional[str] = None) -> tuple[dict[str, Any], Type[BaseComponent]]:
        """Extract a zipped component into ``components/`` and load it."""
        target_dir = target_dir or COMPONENTS_DIR
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            root = names[0].split("/", 1)[0] if names else "component"
            zf.extractall(target_dir)
        extracted = os.path.join(target_dir, root)
        return self.load_from_dir(extracted)

    def discover(self, components_dir: Optional[str] = None) -> list[dict[str, Any]]:
        """Scan ``components/`` for ``component.json`` manifests."""
        components_dir = components_dir or COMPONENTS_DIR
        discovered: list[dict[str, Any]] = []
        if not os.path.isdir(components_dir):
            return discovered
        for entry in os.listdir(components_dir):
            full = os.path.join(components_dir, entry)
            manifest_path = os.path.join(full, "component.json")
            if os.path.isfile(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as fh:
                        discovered.append(json.load(fh))
                except Exception as exc:  # pragma: no cover - skip bad manifests
                    logger.warning("Skipping bad manifest %s: %s", manifest_path, exc)
        return discovered
