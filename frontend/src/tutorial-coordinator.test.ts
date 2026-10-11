import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  TutorialCoordinator,
} from './tutorial-coordinator';

describe('TutorialCoordinator', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.clear();
    }
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('initializes with default state in Watch mode', () => {
    const coordinator = new TutorialCoordinator();
    expect(coordinator.getMode()).toBe('watch');
    expect(coordinator.getCurrentStep()).toBe(1);
    expect(coordinator.getPlaybackSpeed()).toBe(1.0);
    expect(coordinator.getIsLooping()).toBe(true);
    expect(coordinator.getIsCelebrating()).toBe(false);
    expect(coordinator.getIsEscapeHatchActive()).toBe(false);
    expect(coordinator.getCurrentGuide().stepNumber).toBe(1);
  });

  it('switches between watch and verify modes', () => {
    const modeChangeSpy = vi.fn();
    const coordinator = new TutorialCoordinator({ onModeChange: modeChangeSpy });

    coordinator.enterVerifyMode();
    expect(coordinator.getMode()).toBe('verify');
    expect(modeChangeSpy).toHaveBeenCalledWith('verify');

    coordinator.enterWatchMode();
    expect(coordinator.getMode()).toBe('watch');
    expect(modeChangeSpy).toHaveBeenCalledWith('watch');
  });

  it('triggers escape hatch callback after configured timeout in verify mode', () => {
    const escapeSpy = vi.fn();
    const coordinator = new TutorialCoordinator({
      escapeHatchMs: 3000,
      onEscapeHatch: escapeSpy,
    });

    coordinator.enterVerifyMode();
    expect(coordinator.getIsEscapeHatchActive()).toBe(false);

    // Advance timer past 3000ms
    vi.advanceTimersByTime(3001);
    expect(coordinator.getIsEscapeHatchActive()).toBe(true);
    expect(escapeSpy).toHaveBeenCalledWith(true);

    // Switching back to watch clears the escape hatch
    coordinator.enterWatchMode();
    expect(coordinator.getIsEscapeHatchActive()).toBe(false);
    expect(escapeSpy).toHaveBeenCalledWith(false);
  });

  it('handles step verified: marks completed, celebrates, and advances to next step', () => {
    const celebrationStartSpy = vi.fn();
    const celebrationEndSpy = vi.fn();
    const stepChangeSpy = vi.fn();

    const coordinator = new TutorialCoordinator({
      celebrationMs: 1000,
      onCelebrationStart: celebrationStartSpy,
      onCelebrationEnd: celebrationEndSpy,
      onStepChange: stepChangeSpy,
    });

    coordinator.enterVerifyMode();
    coordinator.handleStepVerified(1);

    expect(coordinator.isStepCompleted(1)).toBe(true);
    expect(coordinator.getIsCelebrating()).toBe(true);
    expect(celebrationStartSpy).toHaveBeenCalledWith(1);

    // Advance past celebration duration
    vi.advanceTimersByTime(1001);

    expect(coordinator.getIsCelebrating()).toBe(false);
    expect(celebrationEndSpy).toHaveBeenCalledWith(2);
    expect(coordinator.getCurrentStep()).toBe(2);
    expect(coordinator.getMode()).toBe('watch'); // Resets to watch mode for next step
  });

  it('allows manual skipping via escape hatch', () => {
    const coordinator = new TutorialCoordinator();
    coordinator.setStep(3);
    coordinator.enterVerifyMode();

    coordinator.skipCurrentStep();
    expect(coordinator.isStepCompleted(3)).toBe(true);
    expect(coordinator.getCurrentStep()).toBe(4);
    expect(coordinator.getMode()).toBe('watch');
  });

  it('saves and restores progress from localStorage', () => {
    const coordinator1 = new TutorialCoordinator({ celebrationMs: 500 });
    coordinator1.setStep(4);
    coordinator1.handleStepVerified(4);
    vi.advanceTimersByTime(501); // Wait for celebration to finish and advance step

    // Initialize second coordinator instance
    const coordinator2 = new TutorialCoordinator();
    expect(coordinator2.getCurrentStep()).toBe(5); // Auto-advanced to 5
    expect(coordinator2.isStepCompleted(4)).toBe(true);
  });

  it('controls playback speed and loop toggles', () => {
    const speedSpy = vi.fn();
    const loopSpy = vi.fn();
    const coordinator = new TutorialCoordinator({
      onSpeedChange: speedSpy,
      onLoopChange: loopSpy,
    });

    coordinator.setPlaybackSpeed(0.5);
    expect(coordinator.getPlaybackSpeed()).toBe(0.5);
    expect(speedSpy).toHaveBeenCalledWith(0.5);

    const isLooping = coordinator.toggleLoop();
    expect(isLooping).toBe(false);
    expect(loopSpy).toHaveBeenCalledWith(false);
  });

  it('handles final step skip: marks final step completed, advances step count, and triggers celebration/completion callbacks (FE-02)', () => {
    const celebrationStartSpy = vi.fn();
    const celebrationEndSpy = vi.fn();
    const escapeHatchSpy = vi.fn();

    const coordinator = new TutorialCoordinator({
      totalSteps: 6,
      onCelebrationStart: celebrationStartSpy,
      onCelebrationEnd: celebrationEndSpy,
      onEscapeHatch: escapeHatchSpy,
    });

    coordinator.setStep(6);
    coordinator.enterVerifyMode();

    coordinator.skipCurrentStep();

    expect(coordinator.isStepCompleted(6)).toBe(true);
    expect(coordinator.getCurrentStep()).toBe(7); // totalSteps + 1 indicates completed state
    expect(celebrationStartSpy).toHaveBeenCalledWith(6);
    expect(celebrationEndSpy).toHaveBeenCalledWith(null);
    expect(coordinator.getIsEscapeHatchActive()).toBe(false);
  });

  it('handles final step verification: advances step to totalSteps + 1 upon completion', () => {
    const celebrationEndSpy = vi.fn();
    const coordinator = new TutorialCoordinator({
      totalSteps: 6,
      celebrationMs: 500,
      onCelebrationEnd: celebrationEndSpy,
    });

    coordinator.setStep(6);
    coordinator.enterVerifyMode();
    coordinator.handleStepVerified(6);

    vi.advanceTimersByTime(501);

    expect(coordinator.isStepCompleted(6)).toBe(true);
    expect(coordinator.getCurrentStep()).toBe(7);
    expect(celebrationEndSpy).toHaveBeenCalledWith(null);
  });

  it('restores progress respecting dynamic totalSteps for figures with fewer steps (FE-03)', () => {
    const map = new Map<string, string>();
    const storage: StorageProvider = {
      getItem: (k) => map.get(k) ?? null,
      setItem: (k, v) => { map.set(k, v); },
      removeItem: (k) => { map.delete(k); },
    };

    storage.setItem(
      'gamisenpai_tutorial_progress',
      JSON.stringify({ currentStep: 5, completedSteps: [1, 2, 3, 4, 5] })
    );

    const coordinator = new TutorialCoordinator({ totalSteps: 4, storageProvider: storage });
    expect(coordinator.getCurrentStep()).toBe(1);
    expect(coordinator.getCompletedSteps()).toEqual([1, 2, 3, 4]);

    storage.setItem(
      'gamisenpai_tutorial_progress',
      JSON.stringify({ currentStep: 3, completedSteps: [1, 2] })
    );

    const coordinatorValid = new TutorialCoordinator({ totalSteps: 4, storageProvider: storage });
    expect(coordinatorValid.getCurrentStep()).toBe(3);
    expect(coordinatorValid.getCompletedSteps()).toEqual([1, 2]);
  });

  it('restores progress respecting dynamic totalSteps for figures with more steps (FE-03)', () => {
    const map = new Map<string, string>();
    const storage: StorageProvider = {
      getItem: (k) => map.get(k) ?? null,
      setItem: (k, v) => { map.set(k, v); },
      removeItem: (k) => { map.delete(k); },
    };

    storage.setItem(
      'gamisenpai_tutorial_progress',
      JSON.stringify({ currentStep: 7, completedSteps: [1, 2, 3, 4, 5, 6] })
    );

    const coordinator = new TutorialCoordinator({ totalSteps: 8, storageProvider: storage });
    expect(coordinator.getCurrentStep()).toBe(7);
    expect(coordinator.getCompletedSteps()).toEqual([1, 2, 3, 4, 5, 6]);
  });
});

