"""Canonical definitions for the seven-class crane classifier with negative background."""

STEP_CLASSES = (
    "Chk0_NonOrigami",
    "Chk1",
    "Chk2",
    "Chk3",
    "Chk4",
    "Chk5",
    "Chk6",
)

ORIGAMI_STEPS = (
    "Chk1",
    "Chk2",
    "Chk3",
    "Chk4",
    "Chk5",
    "Chk6",
)

STEP_NAMES = {
    "Chk0_NonOrigami": "Non-Origami / Background",
    "Chk1": "Square Base",
    "Chk2": "Kite Folds",
    "Chk3": "Bird Base",
    "Chk4": "Neck & Tail Setup",
    "Chk5": "Head Fold",
    "Chk6": "Completed Crane",
}


def step_number(class_name):
    """Return the numeric step for a validated class name (0 for Non-Origami, 1-6 for steps)."""
    if class_name not in STEP_CLASSES:
        raise ValueError(f"Unsupported step class: {class_name!r}")
    if class_name == "Chk0_NonOrigami":
        return 0
    return int(class_name.removeprefix("Chk"))
