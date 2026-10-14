import { describe, it, expect, vi } from 'vitest';
import { VideoStepLoader, VideoElementTarget } from './video-loader';

function createMockVideoElement(): VideoElementTarget & {
  listeners: Record<string, EventListenerOrEventListenerObject[]>;
  emit: (event: string) => void;
} {
  const listeners: Record<string, EventListenerOrEventListenerObject[]> = {};

  return {
    src: '',
    playbackRate: 1.0,
    loop: false,
    listeners,
    load: vi.fn(),
    play: vi.fn().mockResolvedValue(undefined),
    addEventListener: vi.fn((event: string, listener: EventListenerOrEventListenerObject) => {
      if (!listeners[event]) listeners[event] = [];
      listeners[event].push(listener);
    }),
    removeEventListener: vi.fn((event: string, listener: EventListenerOrEventListenerObject) => {
      if (!listeners[event]) return;
      listeners[event] = listeners[event].filter((l) => l !== listener);
    }),
    emit: (event: string) => {
      const callbacks = listeners[event] || [];
      callbacks.forEach((cb) => {
        if (typeof cb === 'function') cb(new Event(event));
      });
    },
  };
}

describe('VideoStepLoader', () => {
  it('loads video successfully and triggers onSuccess on canplay', () => {
    const loader = new VideoStepLoader();
    const mockVideo = createMockVideoElement();
    const onStart = vi.fn();
    const onSuccess = vi.fn();
    const onFallback = vi.fn();

    loader.load(mockVideo, '/videos/step-1.mp4', '/images/fallback.png', 1.0, true, {
      onStart,
      onSuccess,
      onFallback,
    });

    expect(onStart).toHaveBeenCalled();
    expect(mockVideo.src).toBe('/videos/step-1.mp4');
    expect(mockVideo.playbackRate).toBe(1.0);
    expect(mockVideo.loop).toBe(true);
    expect(mockVideo.load).toHaveBeenCalled();

    mockVideo.emit('canplay');
    expect(onSuccess).toHaveBeenCalled();
    expect(mockVideo.play).toHaveBeenCalled();
    expect(onFallback).not.toHaveBeenCalled();
  });

  it('triggers onFallback immediately when videoUrl is empty', () => {
    const loader = new VideoStepLoader();
    const mockVideo = createMockVideoElement();
    const onFallback = vi.fn();

    loader.load(mockVideo, '', '/images/fallback.png', 1.0, true, {
      onFallback,
    });

    expect(onFallback).toHaveBeenCalledWith('/images/fallback.png');
    expect(mockVideo.load).not.toHaveBeenCalled();
  });

  it('triggers onFallback when video emits error event', () => {
    const loader = new VideoStepLoader();
    const mockVideo = createMockVideoElement();
    const onFallback = vi.fn();

    loader.load(mockVideo, '/videos/step-1.mp4', '/images/fallback.png', 1.0, true, {
      onFallback,
    });

    mockVideo.emit('error');
    expect(onFallback).toHaveBeenCalledWith('/images/fallback.png');
  });

  it('ignores aborted error from superseded video load when steps change rapidly (FE-04)', () => {
    const loader = new VideoStepLoader();
    const mockVideo = createMockVideoElement();
    const onFallbackStep1 = vi.fn();
    const onSuccessStep1 = vi.fn();
    const onFallbackStep2 = vi.fn();
    const onSuccessStep2 = vi.fn();

    let capturedStep1Error: ((e: Event) => void) | null = null;
    const originalAdd = mockVideo.addEventListener;
    mockVideo.addEventListener = vi.fn((event: string, listener: EventListenerOrEventListenerObject) => {
      if (event === 'error' && !capturedStep1Error && typeof listener === 'function') {
        capturedStep1Error = listener as (e: Event) => void;
      }
      originalAdd(event, listener);
    });

    loader.load(mockVideo, '/videos/step-1.mp4', '/images/step-1.png', 1.0, true, {
      onSuccess: onSuccessStep1,
      onFallback: onFallbackStep1,
    });

    loader.load(mockVideo, '/videos/step-2.mp4', '/images/step-2.png', 1.0, true, {
      onSuccess: onSuccessStep2,
      onFallback: onFallbackStep2,
    });

    expect(mockVideo.src).toBe('/videos/step-2.mp4');

    if (capturedStep1Error) {
      (capturedStep1Error as (e: Event) => void)(new Event('error'));
    }
    expect(onFallbackStep1).not.toHaveBeenCalled();
    expect(onFallbackStep2).not.toHaveBeenCalled();

    mockVideo.emit('canplay');
    expect(onSuccessStep2).toHaveBeenCalled();
    expect(onFallbackStep2).not.toHaveBeenCalled();
  });

  it('cancels pending load on cancel', () => {
    const loader = new VideoStepLoader();
    const mockVideo = createMockVideoElement();
    const onSuccess = vi.fn();
    const onFallback = vi.fn();

    loader.load(mockVideo, '/videos/step-1.mp4', '/images/fallback.png', 1.0, true, {
      onSuccess,
      onFallback,
    });

    loader.cancel();
    mockVideo.emit('canplay');
    mockVideo.emit('error');

    expect(onSuccess).not.toHaveBeenCalled();
    expect(onFallback).not.toHaveBeenCalled();
  });
});
