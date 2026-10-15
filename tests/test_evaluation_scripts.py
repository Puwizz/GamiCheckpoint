from pathlib import Path
from unittest.mock import MagicMock
import pytest

import scripts.evaluation.audit_dataset_leakage as audit_module
import scripts.evaluation.eval_and_benchmark as eval_bench_module
import scripts.evaluation.eval_and_benchmark_run10 as eval_run10_module


def test_audit_dataset_leakage_main_exit_success(monkeypatch):
    mock_args = MagicMock(dir="data/dummy_dir")
    monkeypatch.setattr(audit_module, "parse_args", lambda: mock_args)
    monkeypatch.setattr(audit_module, "audit_dataset", lambda target: True)

    with pytest.raises(SystemExit) as exc_info:
        audit_module.main()
    assert exc_info.value.code == 0


def test_audit_dataset_leakage_main_exit_failure(monkeypatch):
    mock_args = MagicMock(dir="data/dummy_dir")
    monkeypatch.setattr(audit_module, "parse_args", lambda: mock_args)
    monkeypatch.setattr(audit_module, "audit_dataset", lambda target: False)

    with pytest.raises(SystemExit) as exc_info:
        audit_module.main()
    assert exc_info.value.code == 1


def test_eval_and_benchmark_main_model_not_found_exit_failure(monkeypatch):
    mock_args = MagicMock(name="nonexistent_run", model=None, data="dummy.yaml", imgsz=640, device="cpu")
    monkeypatch.setattr(eval_bench_module, "parse_args", lambda: mock_args)
    monkeypatch.setattr(Path, "exists", lambda self: False)

    with pytest.raises(SystemExit) as exc_info:
        eval_bench_module.main()
    assert exc_info.value.code == 1


def test_eval_and_benchmark_run10_main_model_not_found_exit_failure(monkeypatch):
    mock_args = MagicMock(model=None, data="dummy.yaml", imgsz=640, device="cpu")
    monkeypatch.setattr(eval_run10_module, "parse_args", lambda: mock_args)
    monkeypatch.setattr(Path, "exists", lambda self: False)

    with pytest.raises(SystemExit) as exc_info:
        eval_run10_module.main()
    assert exc_info.value.code == 1
