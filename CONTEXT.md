# GamiSenpai

Real-time guided origami crane checkpoint verification system.

## Language

**Checkpoint**:
A recognized milestone fold shape in the six-step origami crane sequence (`Chk1` through `Chk6`).
_Avoid_: stage, phase, level

**Square Base**:
The initial checkpoint (`Chk1`), formed by neat layered square collapse with the open end visible.
_Avoid_: base fold, step 1 shape

**Kite Folds**:
The second checkpoint (`Chk2`), where side edges fold cleanly toward the central crease to form a long diamond.
_Avoid_: diamond fold, step 2

**Bird Base**:
The third checkpoint (`Chk3`), a narrow elongated base with two lower points ready to become neck and tail.
_Avoid_: petal base, step 3

**Neck & Tail Setup**:
The fourth checkpoint (`Chk4`), where both lower points are raised upward while the neck tip remains unfolded.
_Avoid_: raised points, step 4

**Head Fold**:
The fifth checkpoint (`Chk5`), where the neck tip is folded downward to form the crane head.
_Avoid_: beak fold, step 5

**Completed Crane**:
The final checkpoint (`Chk6`), where both wings are spread outward and all crane features are fully formed.
_Avoid_: finished bird, crane model

**Non-Origami**:
A negative background class (`Chk0_NonOrigami`) representing hands, clutter, empty desks, or non-crane objects.
_Avoid_: background, noise, unclassified

**Guided Flow**:
The sequential user progression state machine requiring consecutive checkpoint confirmations before unlocking subsequent steps.
_Avoid_: wizard, onboarding, flow engine

**StepSessionEngine**:
The in-process domain module that governs guided flow state transitions, leaky-bucket confirmation streak accumulation, and lock guards.
_Avoid_: flow controller, session manager

**Multi-Origami Hub**:
The landing gallery catalog where users discover, preview, and select origami figures before entering the craft studio workspace.
_Avoid_: figure picker, level selector, menu

**Figure Registry**:
The backend-driven dynamic domain module loading figure specifications from `configs/figures/*.json`, resolving model artifact paths, and managing StepVerifier lifecycle with in-memory caching.
_Avoid_: figure manager, model database, config loader

**Detector Adapter**:
An internal adapter plugged into the StepVerifier at its internal seam to execute model predictions (`YoloDetectorAdapter`, or `SyntheticDetectorAdapter` for tests).
_Avoid_: model service, detector client, model wrapper
