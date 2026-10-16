"""
tests/test_figure_registry.py
Unit tests for the deep FigureRegistry domain module.
"""
import json
import os
import pytest
from src.figure_registry import FigureRegistry


def test_figure_registry_loads_default_configs():
    registry = FigureRegistry()
    registry.load()
    assert "crane" in registry
    assert "frog" in registry
    crane = registry.get("crane")
    assert crane is not None
    assert crane["id"] == "crane"
    assert crane["total_steps"] == 6


def test_figure_registry_list_summaries():
    registry = FigureRegistry()
    summaries = registry.list_summaries()
    assert isinstance(summaries, list)
    assert len(summaries) >= 2
    
    # Available figures must be sorted before coming_soon
    first_fig = summaries[0]
    assert first_fig["status"] == "available"
    assert first_fig["id"] == "crane"


def test_figure_registry_get_classes():
    registry = FigureRegistry()
    classes = registry.get_classes("crane")
    assert classes == ["Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"]


def test_figure_registry_get_nonexistent():
    registry = FigureRegistry()
    assert registry.get("unknown_origami") is None
    assert registry.get_classes("unknown_origami") == []


def test_figure_registry_caching_and_reload(tmp_path):
    # Test with an isolated custom directory
    fig1 = {
        "id": "lotus",
        "title_en": "Lotus",
        "title_th": "ดอกบัว",
        "status": "available",
        "difficulty_level": 1,
        "steps": [{"step_number": 1, "class_name": "Chk1"}],
    }
    with open(tmp_path / "lotus.json", "w", encoding="utf-8") as f:
        json.dump(fig1, f)

    reg = FigureRegistry(configs_dir=str(tmp_path))
    assert "lotus" in reg
    assert len(reg.list_summaries()) == 1

    # Add a second figure without force_reload
    fig2 = {
        "id": "butterfly",
        "title_en": "Butterfly",
        "title_th": "ผีเสื้อ",
        "status": "coming_soon",
        "difficulty_level": 2,
        "steps": [],
    }
    with open(tmp_path / "butterfly.json", "w", encoding="utf-8") as f:
        json.dump(fig2, f)

    # Cache should still return 1 item
    assert len(reg.list_summaries(force_reload=False)) == 1

    # Reload should now see 2 items
    assert len(reg.list_summaries(force_reload=True)) == 2
    assert "butterfly" in reg


def test_figure_registry_get_verifier_handles_exception(monkeypatch):
    reg = FigureRegistry()
    reg.load()

    def mock_init(*args, **kwargs):
        raise RuntimeError("Corrupted weights or CUDA OOM simulation")

    import yolo_verifier
    monkeypatch.setattr(yolo_verifier, "YoloStepVerifier", mock_init)

    # Must return None gracefully instead of raising unhandled exception
    verifier = reg.get_verifier("crane")
    assert verifier is None


def test_figure_registry_list_summaries_with_none_difficulty_level(tmp_path):
    """CORE-11: Verify that difficulty_level=None (null in JSON) does not raise TypeError on sort."""
    fig1 = {
        "id": "lotus",
        "title_en": "Lotus",
        "title_th": "ดอกบัว",
        "status": "available",
        "difficulty_level": None,  # null in JSON
        "steps": [{"step_number": 1, "class_name": "Chk1"}],
    }
    fig2 = {
        "id": "rose",
        "title_en": "Rose",
        "title_th": "ดอกกุหลาบ",
        "status": "available",
        "difficulty_level": 2,
        "steps": [{"step_number": 1, "class_name": "Chk1"}],
    }
    with open(tmp_path / "lotus.json", "w", encoding="utf-8") as f:
        json.dump(fig1, f)
    with open(tmp_path / "rose.json", "w", encoding="utf-8") as f:
        json.dump(fig2, f)

    reg = FigureRegistry(configs_dir=str(tmp_path))
    summaries = reg.list_summaries()
    assert len(summaries) == 2
    # Verify sorting succeeds without TypeError and lotus (fallback 1) sorts before rose (2)
    assert summaries[0]["id"] == "lotus"
    assert summaries[0]["difficulty_level"] == 1
    assert summaries[1]["id"] == "rose"
    assert summaries[1]["difficulty_level"] == 2


def test_figure_registry_resolve_model_path_env_override(monkeypatch, tmp_path):
    reg = FigureRegistry()
    reg.load()

    # Test cross-platform absolute path override
    custom_weights_file = tmp_path / "best.pt"
    custom_weights_path = str(custom_weights_file)
    monkeypatch.setenv("GAMISENPAI_MODEL_PATH", custom_weights_path)

    resolved = reg.resolve_model_path("crane")
    assert resolved == custom_weights_path

    # Test relative path override (resolved relative to reg.base_dir)
    rel_weights = "weights/custom/best.pt"
    monkeypatch.setenv("GAMISENPAI_MODEL_PATH", rel_weights)
    resolved_rel = reg.resolve_model_path("crane")
    assert resolved_rel == os.path.normpath(os.path.join(reg.base_dir, rel_weights))


def test_figure_registry_resolve_model_path_nonexistent():
    reg = FigureRegistry()
    assert reg.resolve_model_path("nonexistent_figure") is None


def test_figure_registry_clear_verifiers():
    reg = FigureRegistry()
    reg._verifiers["fake_figure"] = "dummy_verifier"
    assert "fake_figure" in reg._verifiers
    reg.clear_verifiers()
    assert len(reg._verifiers) == 0

