"""
Unit tests for src/step_config.py (CORE-13)
"""

import pytest
from src.step_config import step_number, STEP_CLASSES, ORIGAMI_STEPS


def test_step_number_canonical_classes():
    assert step_number("Chk0_NonOrigami") == 0
    assert step_number("Chk1") == 1
    assert step_number("Chk2") == 2
    assert step_number("Chk3") == 3
    assert step_number("Chk4") == 4
    assert step_number("Chk5") == 5
    assert step_number("Chk6") == 6


def test_step_number_generalized_multi_figure_steps():
    """CORE-13: Verifies that step numbers beyond Crane's 6 steps are supported."""
    assert step_number("Chk7") == 7
    assert step_number("Chk8") == 8
    assert step_number("Chk12") == 12
    assert step_number("Chk25") == 25


@pytest.mark.parametrize("invalid_input", [
    "",
    "Chk",
    "ChkABC",
    "Step1",
    "NonOrigami",
    "Chk_1",
    None,
    123,
])
def test_step_number_invalid_inputs(invalid_input):
    with pytest.raises(ValueError):
        step_number(invalid_input)


def test_step_classes_and_origami_steps_constants():
    assert len(STEP_CLASSES) == 7
    assert STEP_CLASSES[0] == "Chk0_NonOrigami"
    assert len(ORIGAMI_STEPS) == 6
    assert ORIGAMI_STEPS == STEP_CLASSES[1:]
