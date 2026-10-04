/**
 * TutorialCoordinator
 * 
 * Deep module that orchestrates the Interactive Video Tutorial flow:
 * - Alternates between 'watch' (Watch & Fold) and 'verify' (Show & Verify)
 * - Controls video player settings (playback speed, loop, fallback detection)
 * - Coordinates the 15-second Escape Hatch timeout
 * - Persists and restores step progress via window.localStorage
 * - Produces celebratory chimes using Web Audio API (zero external asset deps)
 */

import {
  TutorMode,
  PlaybackSpeed,
  TutorialProgress,
  StepFoldGuide,
  STEP_FOLD_GUIDES,
  TOTAL_STEPS,
} from './types';

export const PROGRESS_STORAGE_KEY = 'gamisenpai_tutorial_progress';
export const ESCAPE_HATCH_TIMEOUT_MS = 15000;
export const CELEBRATION_DURATION_MS = 2500;

export interface StorageProvider {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

class MemoryStorage implements StorageProvider {
  private map = new Map<string, string>();
  getItem(key: string): string | null {
    return this.map.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.map.set(key, value);
  }
  removeItem(key: string): void {
    this.map.delete(key);
  }
}

// Global in-memory storage fallback for Node/non-browser test runs
const fallbackMemoryStorage = new MemoryStorage();

export interface TutorialCoordinatorOptions {
  storageProvider?: StorageProvider;
  storageKey?: string;
  totalSteps?: number;
  guides?: StepFoldGuide[];
  escapeHatchMs?: number;
  celebrationMs?: number;
  initialStep?: number;
  onModeChange?: (mode: TutorMode) => void;
  onStepChange?: (step: number, guide: StepFoldGuide) => void;
  onSpeedChange?: (speed: PlaybackSpeed) => void;
  onLoopChange?: (loop: boolean) => void;
  onEscapeHatch?: (active: boolean) => void;
  onCelebrationStart?: (step: number) => void;
  onCelebrationEnd?: (nextStep: number | null) => void;
}

export class TutorialCoordinator {
  private mode: TutorMode = 'watch';
  private currentStep: number = 1;
  private totalSteps: number = TOTAL_STEPS;
  private guides: StepFoldGuide[] = STEP_FOLD_GUIDES;
  private playbackSpeed: PlaybackSpeed = 1.0;
  private isLooping: boolean = true;
  private completedSteps: Set<number> = new Set();
  private isCelebrating: boolean = false;
  private escapeHatchActive: boolean = false;
  private escapeHatchTimer: any = null;
  private celebrationTimer: any = null;
  private audioCtx: AudioContext | null = null;

  private storageKey: string;
  private readonly storage: StorageProvider;
  private readonly escapeHatchMs: number;
  private readonly celebrationMs: number;

  public onModeChange?: (mode: TutorMode) => void;
  public onStepChange?: (step: number, guide: StepFoldGuide) => void;
  public onSpeedChange?: (speed: PlaybackSpeed) => void;
  public onLoopChange?: (loop: boolean) => void;
  public onEscapeHatch?: (active: boolean) => void;
  public onCelebrationStart?: (step: number) => void;
  public onCelebrationEnd?: (nextStep: number | null) => void;

  constructor(options: TutorialCoordinatorOptions = {}) {
    this.storageKey = options.storageKey || PROGRESS_STORAGE_KEY;
    this.storage =
      options.storageProvider ||
      (typeof window !== 'undefined' && window.localStorage ? window.localStorage : fallbackMemoryStorage);
    this.escapeHatchMs = options.escapeHatchMs || ESCAPE_HATCH_TIMEOUT_MS;
    this.celebrationMs = options.celebrationMs || CELEBRATION_DURATION_MS;
    if (options.totalSteps) this.totalSteps = options.totalSteps;
    if (options.guides) this.guides = options.guides;

    this.onModeChange = options.onModeChange;
    this.onStepChange = options.onStepChange;
    this.onSpeedChange = options.onSpeedChange;
    this.onLoopChange = options.onLoopChange;
    this.onEscapeHatch = options.onEscapeHatch;
    this.onCelebrationStart = options.onCelebrationStart;
    this.onCelebrationEnd = options.onCelebrationEnd;

    this.restoreProgress(options.initialStep);
  }

  // --- Reconfigure for different origami figures ---
  public reconfigure(totalSteps: number, guides: StepFoldGuide[], storageKey?: string): void {
    this.totalSteps = totalSteps;
    this.guides = guides;
    if (storageKey) {
      this.storageKey = storageKey;
    }
    this.completedSteps.clear();
    this.currentStep = 1;
    this.clearEscapeHatchTimer();
    this.restoreProgress();
    this.onStepChange?.(this.currentStep, this.getCurrentGuide());
  }

  // --- Getters ---
  public getMode(): TutorMode {
    return this.mode;
  }

  public getCurrentStep(): number {
    return this.currentStep;
  }

  public getTotalSteps(): number {
    return this.totalSteps;
  }

  public getGuides(): StepFoldGuide[] {
    return this.guides;
  }

  public getPlaybackSpeed(): PlaybackSpeed {
    return this.playbackSpeed;
  }

  public getIsLooping(): boolean {
    return this.isLooping;
  }

  public getCompletedSteps(): number[] {
    return Array.from(this.completedSteps).sort((a, b) => a - b);
  }

  public isStepCompleted(step: number): boolean {
    return this.completedSteps.has(step);
  }

  public getIsCelebrating(): boolean {
    return this.isCelebrating;
  }

  public getIsEscapeHatchActive(): boolean {
    return this.escapeHatchActive;
  }

  public getCurrentGuide(): StepFoldGuide {
    const guide = this.guides.find((g) => g.stepNumber === this.currentStep);
    return guide || this.guides[0] || STEP_FOLD_GUIDES[0];
  }

  // --- Mode Switching ---
  public setMode(mode: TutorMode): void {
    if (this.mode === mode) return;
    this.mode = mode;

    if (mode === 'verify') {
      this.startEscapeHatchTimer();
    } else {
      this.clearEscapeHatchTimer();
    }

    this.onModeChange?.(this.mode);
  }

  public enterWatchMode(): void {
    this.setMode('watch');
  }

  public enterVerifyMode(): void {
    this.setMode('verify');
  }

  // --- Video Controls ---
  public setPlaybackSpeed(speed: PlaybackSpeed): void {
    this.playbackSpeed = speed;
    this.onSpeedChange?.(this.playbackSpeed);
  }

  public toggleLoop(): boolean {
    this.isLooping = !this.isLooping;
    this.onLoopChange?.(this.isLooping);
    return this.isLooping;
  }

  public setLoop(loop: boolean): void {
    this.isLooping = loop;
    this.onLoopChange?.(this.isLooping);
  }

  // --- Step Navigation & Progression ---
  public setStep(step: number, forceWatchMode: boolean = true): void {
    const clamped = Math.max(1, Math.min(this.totalSteps, step));
    this.currentStep = clamped;
    this.clearEscapeHatchTimer();

    if (forceWatchMode) {
      this.mode = 'watch';
      this.onModeChange?.(this.mode);
    }

    this.saveProgress();
    this.onStepChange?.(this.currentStep, this.getCurrentGuide());
  }

  public handleStepVerified(step: number): void {
    if (this.isCelebrating) return;
    if (step !== this.currentStep) return;

    this.completedSteps.add(step);
    this.saveProgress();
    this.clearEscapeHatchTimer();

    this.isCelebrating = true;
    this.onCelebrationStart?.(step);
    this.playCelebrationChime();

    // Auto-advance after celebration duration
    if (this.celebrationTimer) {
      clearTimeout(this.celebrationTimer);
    }

    this.celebrationTimer = setTimeout(() => {
      this.isCelebrating = false;
      if (this.currentStep < this.totalSteps) {
        const nextStep = this.currentStep + 1;
        this.setStep(nextStep, true);
        this.onCelebrationEnd?.(nextStep);
      } else {
        // Final completion!
        this.onCelebrationEnd?.(null);
      }
    }, this.celebrationMs);
  }

  public skipCurrentStep(): void {
    this.clearEscapeHatchTimer();
    this.completedSteps.add(this.currentStep);
    this.saveProgress();

    if (this.currentStep < this.totalSteps) {
      this.setStep(this.currentStep + 1, true);
    } else {
      this.onCelebrationEnd?.(null);
    }
  }

  // --- Escape Hatch Timer ---
  private startEscapeHatchTimer(): void {
    this.clearEscapeHatchTimer();
    this.escapeHatchTimer = setTimeout(() => {
      if (this.mode === 'verify' && !this.isCelebrating) {
        this.escapeHatchActive = true;
        this.onEscapeHatch?.(true);
      }
    }, this.escapeHatchMs);
  }

  private clearEscapeHatchTimer(): void {
    if (this.escapeHatchTimer) {
      clearTimeout(this.escapeHatchTimer);
      this.escapeHatchTimer = null;
    }
    if (this.escapeHatchActive) {
      this.escapeHatchActive = false;
      this.onEscapeHatch?.(false);
    }
  }

  // --- LocalStorage Persistence ---
  private saveProgress(): void {
    try {
      const data: TutorialProgress = {
        currentStep: this.currentStep,
        completedSteps: Array.from(this.completedSteps),
        lastActiveTime: new Date().toISOString(),
      };
      this.storage.setItem(this.storageKey, JSON.stringify(data));
    } catch {
      // Ignore storage errors (quotas)
    }
  }

  private restoreProgress(forcedStep?: number): void {
    if (forcedStep && forcedStep >= 1 && forcedStep <= TOTAL_STEPS) {
      this.currentStep = forcedStep;
      return;
    }

    try {
      const raw = this.storage.getItem(this.storageKey);
      if (!raw) return;
      const data: TutorialProgress = JSON.parse(raw);
      if (data.currentStep && data.currentStep >= 1 && data.currentStep <= TOTAL_STEPS) {
        this.currentStep = data.currentStep;
      }
      if (Array.isArray(data.completedSteps)) {
        this.completedSteps = new Set(data.completedSteps);
      }
    } catch {
      // Fall back to Step 1 on parse error
      this.currentStep = 1;
    }
  }

  public resetProgress(): void {
    this.completedSteps.clear();
    this.currentStep = 1;
    this.mode = 'watch';
    this.clearEscapeHatchTimer();
    try {
      this.storage.removeItem(this.storageKey);
    } catch {
      // Ignore
    }
    this.onModeChange?.(this.mode);
    this.onStepChange?.(this.currentStep, this.getCurrentGuide());
  }

  // --- Web Audio Chime Synthesizer ---
  public playCelebrationChime(): void {
    try {
      const AudioContextClass =
        (window as any).AudioContext || (window as any).webkitAudioContext;
      if (!AudioContextClass) return;

      if (!this.audioCtx) {
        this.audioCtx = new AudioContextClass();
      }

      if (this.audioCtx && this.audioCtx.state === 'suspended') {
        this.audioCtx.resume();
      }

      const now = this.audioCtx ? this.audioCtx.currentTime : 0;
      // Synthesize 3 harmonic bell tones (C5: 523.25, E5: 659.25, G5: 783.99)
      const frequencies = [523.25, 659.25, 783.99];

      frequencies.forEach((freq, idx) => {
        if (!this.audioCtx) return;
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();

        osc.type = 'triangle';
        osc.frequency.setValueAtTime(freq, now + idx * 0.08);

        gain.gain.setValueAtTime(0, now + idx * 0.08);
        gain.gain.linearRampToValueAtTime(0.18, now + idx * 0.08 + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.08 + 0.6);

        osc.connect(gain);
        gain.connect(this.audioCtx.destination);

        osc.start(now + idx * 0.08);
        osc.stop(now + idx * 0.08 + 0.65);
      });
    } catch {
      // Graceful silence if audio is restricted
    }
  }

  public destroy(): void {
    this.clearEscapeHatchTimer();
    if (this.celebrationTimer) {
      clearTimeout(this.celebrationTimer);
      this.celebrationTimer = null;
    }
    if (this.audioCtx && this.audioCtx.state !== 'closed') {
      try {
        this.audioCtx.close();
      } catch {
        // Ignore
      }
    }
    this.audioCtx = null;
  }
}
