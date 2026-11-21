import { describe, it, expect, vi } from 'vitest';
import { VideoStepLoader, VideoElementTarget, isVideoFullscreen, toggleVideoFullscreen, exitVideoFullscreenIfActive } from './video-loader';

function createMockVideoElement(): VideoElementTarget & {
  listeners: Record<string, EventListenerOrEventListenerObject[]>;
  emit: (event: string) => void;
} {
  const listeners: Record<string, EventListenerOrEventListenerObject[]> = {};

  return {
    src: '',
    playbackRate: 1.0,
    loop: false,
    muted: false,
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
    }, true);

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

  it('does not autoplay when autoPlay is false', () => {
    const loader = new VideoStepLoader();
    const mockVideo = createMockVideoElement();
    const onSuccess = vi.fn();

    loader.load(mockVideo, '/videos/step-1.mp4', '/images/fallback.png', 1.0, false, {
      onSuccess,
    }, false);

    mockVideo.emit('canplay');
    expect(onSuccess).toHaveBeenCalled();
    expect(mockVideo.play).not.toHaveBeenCalled();
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

  it('reuses existing video source when videoUrl is identical and readyState >= 2 (avoids reload stall)', () => {
    const loader = new VideoStepLoader();
    const mockVideo = createMockVideoElement();
    mockVideo.src = 'http://localhost:5173/videos/step-1.mp4';
    mockVideo.readyState = 3; // HAVE_FUTURE_DATA

    const onStart = vi.fn();
    const onSuccess = vi.fn();
    const onFallback = vi.fn();

    loader.load(mockVideo, '/videos/step-1.mp4', '/images/fallback.png', 1.5, true, {
      onStart,
      onSuccess,
      onFallback,
    });

    expect(onStart).toHaveBeenCalled();
    expect(mockVideo.load).not.toHaveBeenCalled();
    expect(mockVideo.playbackRate).toBe(1.5);
    expect(mockVideo.loop).toBe(true);
    expect(onSuccess).toHaveBeenCalled();
    expect(onFallback).not.toHaveBeenCalled();
  });

  it('does not call video.load() a second time when an identical videoUrl is already loading in-flight', () => {
    const loader = new VideoStepLoader();
    const mockVideo = createMockVideoElement();
    const onSuccess1 = vi.fn();
    const onSuccess2 = vi.fn();

    loader.load(mockVideo, '/videos/step-1.mp4', '/images/fallback.png', 1.0, false, {
      onSuccess: onSuccess1,
    });
    expect(mockVideo.load).toHaveBeenCalledTimes(1);

    mockVideo.readyState = 0;
    loader.load(mockVideo, '/videos/step-1.mp4', '/images/fallback.png', 1.25, true, {
      onSuccess: onSuccess2,
    });

    expect(mockVideo.load).toHaveBeenCalledTimes(1);
    expect(mockVideo.playbackRate).toBe(1.25);
    expect(mockVideo.loop).toBe(true);

    mockVideo.emit('canplay');
    expect(onSuccess1).not.toHaveBeenCalled();
    expect(onSuccess2).toHaveBeenCalledTimes(1);
  });

  it('unmutes video by default (muted = false) and respects explicit isMuted state', () => {
    const loader = new VideoStepLoader();
    const mockVideo = createMockVideoElement();
    mockVideo.muted = true;

    loader.load(mockVideo, '/videos/step-1.mp4', '/images/fallback.png', 1.0, false, {}, false, false);
    expect(mockVideo.muted).toBe(false);

    // Cached ready reuse with isMuted = true
    mockVideo.src = 'http://localhost:5173/videos/step-1.mp4';
    mockVideo.readyState = 3;
    loader.load(mockVideo, '/videos/step-1.mp4', '/images/fallback.png', 1.0, false, {}, false, true);
    expect(mockVideo.muted).toBe(true);
  });
});

describe('Video Fullscreen Helpers', () => {
  it('toggles container fullscreen using standard requestFullscreen', async () => {
    const classSet = new Set<string>();
    const container = {
      classList: {
        contains: (cls: string) => classSet.has(cls),
        add: (cls: string) => classSet.add(cls),
        remove: (cls: string) => classSet.delete(cls),
      },
      requestFullscreen: vi.fn().mockResolvedValue(undefined),
    };
    const video = {};
    const mockDoc = {
      fullscreenElement: null as any,
      exitFullscreen: vi.fn().mockResolvedValue(undefined),
    };

    const res1 = await toggleVideoFullscreen(container, video, mockDoc);
    expect(res1).toBe(true);
    expect(container.requestFullscreen).toHaveBeenCalled();

    mockDoc.fullscreenElement = container;
    expect(isVideoFullscreen(container, video, mockDoc)).toBe(true);

    const res2 = await toggleVideoFullscreen(container, video, mockDoc);
    expect(res2).toBe(false);
    expect(mockDoc.exitFullscreen).toHaveBeenCalled();
  });

  it('falls back to video.webkitEnterFullscreen on iOS when container fullscreen fails', async () => {
    const classSet = new Set<string>();
    const container = {
      classList: {
        contains: (cls: string) => classSet.has(cls),
        add: (cls: string) => classSet.add(cls),
        remove: (cls: string) => classSet.delete(cls),
      },
      requestFullscreen: vi.fn().mockRejectedValue(new Error('Not supported')),
    };
    const video = {
      webkitEnterFullscreen: vi.fn(),
      webkitExitFullscreen: vi.fn(),
      webkitDisplayingFullscreen: false,
    };
    const mockDoc = { fullscreenElement: null };

    const entered = await toggleVideoFullscreen(container, video, mockDoc);
    expect(entered).toBe(true);
    expect(video.webkitEnterFullscreen).toHaveBeenCalled();

    video.webkitDisplayingFullscreen = true;
    expect(isVideoFullscreen(container, video, mockDoc)).toBe(true);

    await exitVideoFullscreenIfActive(container, video, mockDoc);
    expect(video.webkitExitFullscreen).toHaveBeenCalled();
  });

  it('falls back to .is-css-fullscreen class when native fullscreen APIs are missing', async () => {
    const classSet = new Set<string>();
    const container = {
      classList: {
        contains: (cls: string) => classSet.has(cls),
        add: (cls: string) => classSet.add(cls),
        remove: (cls: string) => classSet.delete(cls),
      },
    };
    const video = {};
    const mockDoc = {};

    expect(isVideoFullscreen(container, video, mockDoc)).toBe(false);
    const entered = await toggleVideoFullscreen(container, video, mockDoc);
    expect(entered).toBe(true);
    expect(container.classList.contains('is-css-fullscreen')).toBe(true);
    expect(isVideoFullscreen(container, video, mockDoc)).toBe(true);

    await exitVideoFullscreenIfActive(container, video, mockDoc);
    expect(container.classList.contains('is-css-fullscreen')).toBe(false);
    expect(isVideoFullscreen(container, video, mockDoc)).toBe(false);
  });

  it('returns false from isVideoFullscreen when container is null or undefined', async () => {
    const exitFullscreen = vi.fn();
    expect(isVideoFullscreen(null, {}, { fullscreenElement: null, exitFullscreen })).toBe(false);
    expect(isVideoFullscreen(undefined, {}, {})).toBe(false);

    const toggled = await toggleVideoFullscreen(null, {}, { fullscreenElement: null, exitFullscreen });
    expect(toggled).toBe(false);
    expect(exitFullscreen).not.toHaveBeenCalled();
  });
});
