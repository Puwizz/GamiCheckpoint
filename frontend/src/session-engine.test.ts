import { describe, it, expect, beforeEach } from 'vitest';
import { StepSessionEngine } from './session-engine';
import { VerificationResult } from './types';

describe('StepSessionEngine (Deep Module Contract)', () => {
  let engine: StepSessionEngine;

  beforeEach(() => {
    engine = new StepSessionEngine(1);
  });

  it('starts at Step 1 with 0 streak', () => {
    const state = engine.getState();
    expect(state.currentStep).toBe(1);
    expect(state.stepStreak).toBe(0);
    expect(state.completedSteps.size).toBe(0);
  });

  it('accumulates streak with Leaky Bucket on matching predictions', () => {
    const matchingPrediction: VerificationResult = {
      recognized: true,
      step: 'Chk1',
      step_number: 1,
      confidence: 85,
      top_gap: 20,
    };

    const action1 = engine.evaluatePrediction(matchingPrediction);
    expect(action1.type).toBe('CONFIRMING');
    if (action1.type === 'CONFIRMING') {
      expect(action1.streak).toBe(1);
    }

    const action2 = engine.evaluatePrediction(matchingPrediction);
    expect(action2.type).toBe('CONFIRMING');
    if (action2.type === 'CONFIRMING') {
      expect(action2.streak).toBe(2);
    }

    const action3 = engine.evaluatePrediction(matchingPrediction);
    expect(action3.type).toBe('MATCH_SUCCESS');
    if (action3.type === 'MATCH_SUCCESS') {
      expect(action3.finishedStep).toBe(1);
      expect(action3.nextStep).toBe(2);
    }
  });

  it('decays streak by 1 on single missed frame (Leaky Bucket grace frame)', () => {
    const match: VerificationResult = {
      recognized: true,
      step: 'Chk1',
      step_number: 1,
      confidence: 85,
      top_gap: 20,
    };

    engine.evaluatePrediction(match); // streak = 1
    expect(engine.getState().stepStreak).toBe(1);

    // 1 frame of Unrecognized noise
    const noise: VerificationResult = {
      recognized: false,
      prediction: 'Chk0_NonOrigami',
      recognition_reason: 'non_origami',
    };
    const noiseAction = engine.evaluatePrediction(noise);
    expect(noiseAction.type).toBe('UNRECOGNIZED');
    expect(engine.getState().stepStreak).toBe(0); // Decayed by 1

    // Next frame recovers
    const recoveryAction = engine.evaluatePrediction(match);
    expect(recoveryAction.type).toBe('CONFIRMING');
    expect(engine.getState().stepStreak).toBe(1);
  });

  it('suppresses red LOCK on Step 6 when Step 5 is momentarily detected', () => {
    const engineStep6 = new StepSessionEngine(6);

    const step5Prediction: VerificationResult = {
      recognized: true,
      step: 'Chk5',
      step_number: 5,
      confidence: 80,
      top_gap: 15,
    };

    const action = engineStep6.evaluatePrediction(step5Prediction);
    expect(action.type).toBe('LOCKED_MISMATCH');
    if (action.type === 'LOCKED_MISMATCH') {
      expect(action.status.kind).toBe('waiting'); // Not 'incorrect' (no red lock)
      expect(action.status.icon).toBe('WING');
      expect(action.status.title).toContain('กางปีกออกให้กว้าง');
    }
  });

  it('handles completion through Step 6', () => {
    const engineStep6 = new StepSessionEngine(6);

    const step6Prediction: VerificationResult = {
      recognized: true,
      step: 'Chk6',
      step_number: 6,
      confidence: 85,
      top_gap: 20,
    };

    engineStep6.evaluatePrediction(step6Prediction); // 1
    engineStep6.evaluatePrediction(step6Prediction); // 2
    const finalAction = engineStep6.evaluatePrediction(step6Prediction); // 3

    expect(finalAction.type).toBe('ALL_COMPLETE');
    expect(engineStep6.getState().completedSteps.has(6)).toBe(true);
    expect(engineStep6.getState().currentStep).toBe(7);
  });

  it('supports dynamic figure steps and uses configured step names in mismatch messages', () => {
    const customSteps = [
      { name: 'Waterbomb Base', tip: 'พับฐานลูกบอลน้ำ', reference: 'พับฐานลูกบอลน้ำ' },
      { name: 'Front Legs Fold', tip: 'พับขากบด้านหน้า', reference: 'พับขากบด้านหน้า' },
    ];

    engine.reconfigure(2, customSteps);
    expect(engine.getState().currentStep).toBe(1);

    // Mismatch when detecting Step 2 while waiting for Step 1
    const step2Prediction: VerificationResult = {
      recognized: true,
      step: 'Chk2',
      step_number: 2,
      confidence: 80,
      top_gap: 15,
    };

    const action = engine.evaluatePrediction(step2Prediction);
    expect(action.type).toBe('LOCKED_MISMATCH');
    if (action.type === 'LOCKED_MISMATCH') {
      // Must use custom step name 'Waterbomb Base', not crane's 'Square Base'
      expect(action.status.description).toContain('Waterbomb Base');
      expect(action.status.description).not.toContain('Square Base');
    }
  });

  it('clamps and updates step state on setStep() and resets streak', () => {
    // Accumulate streak first
    const match: VerificationResult = {
      recognized: true,
      step: 'Chk1',
      step_number: 1,
      confidence: 85,
    };
    engine.evaluatePrediction(match);
    expect(engine.getState().stepStreak).toBe(1);

    // Set step explicitly to 3 with completed steps [1, 2]
    engine.setStep(3, [1, 2]);
    const state = engine.getState();
    expect(state.currentStep).toBe(3);
    expect(state.stepStreak).toBe(0);
    expect(state.completedSteps.has(1)).toBe(true);
    expect(state.completedSteps.has(2)).toBe(true);

    // Test out of range clamping
    engine.setStep(-5);
    expect(engine.getState().currentStep).toBe(1);

    engine.setStep(99);
    expect(engine.getState().currentStep).toBe(7); // totalSteps (6) + 1
  });

  it('handles frame_ready false gracefully by decaying streak and returning FRAME_NOT_READY', () => {
    const match: VerificationResult = {
      recognized: true,
      step: 'Chk1',
      step_number: 1,
      confidence: 85,
    };
    engine.evaluatePrediction(match);
    expect(engine.getState().stepStreak).toBe(1);

    const unreadyFrame: VerificationResult = {
      frame_ready: false,
      recognized: false,
      recognition_reason: 'camera_placeholder_or_too_dark',
    };

    const action = engine.evaluatePrediction(unreadyFrame);
    expect(action.type).toBe('FRAME_NOT_READY');
    expect(engine.getState().stepStreak).toBe(0); // Decayed by 1
    if (action.type === 'FRAME_NOT_READY') {
      expect(action.status.kind).toBe('waiting');
      expect(action.status.title).toBe('Waiting for a real camera frame');
    }
  });

  it('includes competing step names in ambiguous_step status and prefers coaching_hint', () => {
    const engineStep3 = new StepSessionEngine(3);

    const ambiguousData: VerificationResult = {
      recognized: false,
      recognition_reason: 'ambiguous_step',
      prediction: 'Chk2',
      step: 'Chk2',
      step_number: 2,
      confidence: 54,
      top_gap: 3.5,
      coaching_hint: 'Your fold is halfway between Kite Folds and Bird Base. Crease the edges firmly to match Step 3 (Bird Base).',
      top_predictions: [
        { class_name: 'Chk2', step_number: 2, step_name: 'Kite Folds', confidence: 54 },
        { class_name: 'Chk3', step_number: 3, step_name: 'Bird Base', confidence: 50.5 },
      ],
    };

    const action = engineStep3.evaluatePrediction(ambiguousData);
    expect(action.type).toBe('UNRECOGNIZED');
    if (action.type === 'UNRECOGNIZED') {
      expect(action.status.description).toContain('Kite Folds');
      expect(action.status.description).toContain('Bird Base');
      expect(action.tip).toBe(ambiguousData.coaching_hint);
    }
  });

  it('falls back cleanly when ambiguous_step has fewer than 2 top_predictions', () => {
    const engineStep2 = new StepSessionEngine(2);
    const singleCandidateAmbiguous: VerificationResult = {
      recognized: false,
      recognition_reason: 'ambiguous_step',
      top_gap: 4.0,
      top_predictions: [{ class_name: 'Chk2', step_number: 2, step_name: '', confidence: 50 }],
    };

    const action = engineStep2.evaluatePrediction(singleCandidateAmbiguous);
    expect(action.type).toBe('UNRECOGNIZED');
    if (action.type === 'UNRECOGNIZED') {
      expect(action.status.description).toContain('top-gap 4.0%');
      expect(action.status.description).not.toContain('undefined');
    }
  });
});


