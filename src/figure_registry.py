"""
src/figure_registry.py — Domain Figure Registry & Model Lifecycle Management
=============================================================================
Centralizes all origami figure catalog specifications and verifier dispatch:
- Loads dynamic figure configurations from configs/figures/*.json with in-memory caching
- Resolves relative model checkpoint paths against project root and environment variables
- Instantiates and caches figure-specific StepVerifier adapters
- Implements dictionary-like access for zero-friction caller integration
"""

import glob
import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple


class FigureRegistry:
    """
    Deep domain module governing origami figure specifications and verifier dispatch.
    
    Hides all filesystem JSON reads, path resolution, schema normalization, and
    verifier lifecycle caching behind a small, queryable interface.
    """

    def __init__(self, configs_dir: Optional[str] = None, base_dir: Optional[str] = None):
        if base_dir is None:
            self.base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
        else:
            self.base_dir = os.path.normpath(base_dir)

        if configs_dir is None:
            self.configs_dir = os.path.join(self.base_dir, "configs", "figures")
        else:
            self.configs_dir = os.path.normpath(configs_dir)

        self._registry: Dict[str, Dict[str, Any]] = {}
        self._verifiers: Dict[str, Any] = {}

    def load(self, force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
        """Load all figure JSON definitions from disk with in-memory caching."""
        if self._registry and not force_reload:
            return self._registry

        registry = {}
        if os.path.exists(self.configs_dir):
            pattern = os.path.join(self.configs_dir, "*.json")
            for filepath in sorted(glob.glob(pattern)):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        fig_id = data.get("id")
                        if fig_id:
                            registry[fig_id] = data
                except Exception as e:
                    print(f"[WARN] Failed to load figure config {filepath}: {e}")

        self._registry = registry
        if force_reload:
            self.clear_verifiers()

        return self._registry

    def list_summaries(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """
        Return catalog summary cards for the selection gallery.
        Sorted available figures first, then by difficulty level.
        """
        registry = self.load(force_reload=force_reload)
        summaries = []
        for fig_id, fig in registry.items():
            steps = fig.get("steps", [])
            summaries.append({
                "id": fig.get("id", fig_id),
                "title_th": fig.get("title_th", ""),
                "title_en": fig.get("title_en", ""),
                "subtitle": fig.get("subtitle", ""),
                "description_th": fig.get("description_th", ""),
                "description_en": fig.get("description_en", ""),
                "status": fig.get("status", "coming_soon"),
                "difficulty": fig.get("difficulty", "ระดับง่าย"),
                "difficulty_level": fig.get("difficulty_level") or 1,
                "estimated_minutes": fig.get("estimated_minutes", 5),
                "thumbnail": fig.get("thumbnail", "/static/references/step-1.png"),
                "icon": fig.get("icon", "origami"),
                "total_steps": fig.get("total_steps", len(steps)),
            })

        summaries.sort(key=lambda x: (0 if x["status"] == "available" else 1, x.get("difficulty_level") or 1))
        return summaries

    def get(self, figure_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the complete specification dictionary for a given figure ID."""
        if not self._registry:
            self.load()
        return self._registry.get(figure_id)

    def get_classes(self, figure_id: str) -> List[str]:
        """Return the ordered list of step class names for a figure (e.g. ['Chk1', 'Chk2', ...])."""
        fig = self.get(figure_id)
        if not fig:
            return []
        steps = fig.get("steps", [])
        return [step["class_name"] for step in steps if "class_name" in step]

    def get_step_names(self, figure_id: str) -> Dict[str, str]:
        """Return a mapping of class_name to human-readable English name."""
        fig = self.get(figure_id)
        if not fig:
            return {}
        names = {"Chk0_NonOrigami": "Non-Origami / Background"}
        for step in fig.get("steps", []):
            c_name = step.get("class_name")
            if c_name:
                names[c_name] = step.get("name_en", c_name)
        return names

    def resolve_model_path(self, figure_id: str) -> Optional[str]:
        """Resolve the model checkpoint path for a figure against project root and env."""
        fig = self.get(figure_id)
        if not fig:
            return None

        # Check environment variable override for crane
        if figure_id == "crane":
            env_path = os.environ.get("GAMISENPAI_MODEL_PATH")
            if env_path:
                return env_path if os.path.isabs(env_path) else os.path.normpath(os.path.join(self.base_dir, env_path))

        model_path = fig.get("model_path")
        if not model_path:
            return None

        if os.path.isabs(model_path):
            return model_path

        # Resolve relative to base_dir
        candidate = os.path.normpath(os.path.join(self.base_dir, model_path))
        if os.path.exists(candidate):
            return candidate

        # Fallback to weights/base/ or weights/checkpoints/ if moved
        filename = os.path.basename(model_path)
        base_candidate = os.path.join(self.base_dir, "weights", "base", filename)
        if os.path.exists(base_candidate):
            return base_candidate

        chk_candidate = os.path.join(self.base_dir, "weights", "checkpoints", filename)
        if os.path.exists(chk_candidate):
            return chk_candidate

        return candidate

    def clear_verifiers(self) -> None:
        """Clear all cached verifier instances."""
        self._verifiers.clear()

    def get_verifier(self, figure_id: str, device: Optional[str] = None) -> Optional[Any]:
        """
        Get or initialize the cached StepVerifier for the requested figure.
        Returns None if the figure is unknown or has no model available.
        """
        if figure_id in self._verifiers:
            return self._verifiers[figure_id]

        fig = self.get(figure_id)
        if not fig or fig.get("status") != "available":
            return None

        model_path = self.resolve_model_path(figure_id)
        if not model_path or not os.path.exists(model_path):
            print(f"[WARN] Figure '{figure_id}' model checkpoint not found at: {model_path}")
            return None

        classes = tuple(self.get_classes(figure_id))
        conf_threshold = float(fig.get("conf_threshold", 0.55))
        imgsz = int(fig.get("imgsz", 640))
        try:
            if model_path.endswith(".pt"):
                from yolo_verifier import YoloStepVerifier
                verifier = YoloStepVerifier(
                    model_path=model_path,
                    classes=classes,
                    conf_threshold=conf_threshold,
                    imgsz=imgsz,
                    device=device,
                )
                self._verifiers[figure_id] = verifier
                return verifier

            print(f"[WARN] Figure '{figure_id}' model_path has unsupported format (expected .pt): {model_path}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to initialize verifier for '{figure_id}' from '{model_path}': {e}")
            return None

    # Mapping protocol methods for transparent dict compatibility
    def __contains__(self, item: str) -> bool:
        if not self._registry:
            self.load()
        return item in self._registry

    def __getitem__(self, item: str) -> Dict[str, Any]:
        if not self._registry:
            self.load()
        return self._registry[item]

    def keys(self):
        if not self._registry:
            self.load()
        return self._registry.keys()

    def items(self):
        if not self._registry:
            self.load()
        return self._registry.items()

    def values(self):
        if not self._registry:
            self.load()
        return self._registry.values()

    def __len__(self) -> int:
        if not self._registry:
            self.load()
        return len(self._registry)


# Global default instance
registry = FigureRegistry()
