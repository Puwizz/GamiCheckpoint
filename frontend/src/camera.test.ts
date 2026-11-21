import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { CameraStreamAdapter } from './camera';

interface MockTrack {
  label: string;
  stop: ReturnType<typeof vi.fn>;
  getSettings: () => { deviceId?: string };
}

function createMockStream(label = 'USB Camera', deviceId = 'cam-1'): {
  stream: MediaStream;
  track: MockTrack;
} {
  const track: MockTrack = {
    label,
    stop: vi.fn(),
    getSettings: () => ({ deviceId }),
  };
  const stream = {
    getTracks: () => [track],
    getVideoTracks: () => [track],
  } as unknown as MediaStream;
  return { stream, track };
}

function createMockVideoElement(): HTMLVideoElement {
  return {
    srcObject: null,
    videoWidth: 1280,
    videoHeight: 720,
    pause: vi.fn(),
    removeAttribute: vi.fn(),
    load: vi.fn(),
  } as unknown as HTMLVideoElement;
}

describe('CameraStreamAdapter', () => {
  const originalNavigator = globalThis.navigator;

  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    Object.defineProperty(globalThis, 'navigator', {
      value: originalNavigator,
      configurable: true,
      writable: true,
    });
  });

  function mockGetUserMedia(impl: (constraints: MediaStreamConstraints) => Promise<MediaStream>) {
    const getUserMediaMock = vi.fn(impl);
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        mediaDevices: {
          getUserMedia: getUserMediaMock,
          enumerateDevices: vi.fn().mockResolvedValue([]),
        },
      },
      configurable: true,
      writable: true,
    });
    return getUserMediaMock;
  }

  it('requests exact deviceId constraint first when a valid deviceId is selected', async () => {
    const { stream } = createMockStream('External Webcam', 'device-ext-99');
    const getUserMediaMock = mockGetUserMedia(async () => stream);
    const videoEl = createMockVideoElement();
    const adapter = new CameraStreamAdapter(videoEl);

    adapter.setSelectedDeviceId('device-ext-99');
    const result = await adapter.start();

    expect(result).toBe(stream);
    expect(getUserMediaMock).toHaveBeenCalledTimes(1);
    expect(getUserMediaMock).toHaveBeenCalledWith({
      video: {
        deviceId: { exact: 'device-ext-99' },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    expect(adapter.getActiveTrackLabel()).toBe('External Webcam');
    expect(adapter.getActiveDeviceId()).toBe('device-ext-99');
    expect(adapter.hasActiveStream()).toBe(true);
    expect(adapter.isStreaming()).toBe(true);
  });

  it('falls back through ideal deviceId, facingMode environment, and bare video: true when constraints fail', async () => {
    const { stream } = createMockStream('Fallback Integrated Camera', 'device-integrated-1');
    const calls: MediaStreamConstraints[] = [];

    const getUserMediaMock = mockGetUserMedia(async (constraints) => {
      calls.push(constraints);
      if (calls.length < 4) {
        const err = new Error('Constraint failed');
        err.name = 'OverconstrainedError';
        throw err;
      }
      return stream;
    });

    const videoEl = createMockVideoElement();
    const adapter = new CameraStreamAdapter(videoEl);
    adapter.setSelectedDeviceId('busy-cam-id');

    const result = await adapter.start();

    expect(result).toBe(stream);
    expect(getUserMediaMock).toHaveBeenCalledTimes(4);
    expect(calls[0]).toEqual({
      video: { deviceId: { exact: 'busy-cam-id' }, width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    expect(calls[1]).toEqual({
      video: { deviceId: { ideal: 'busy-cam-id' }, width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    expect(calls[2]).toEqual({
      video: { facingMode: { ideal: 'environment' } },
    });
    expect(calls[3]).toEqual({
      video: true,
    });
    expect(adapter.getActiveDeviceId()).toBe('device-integrated-1');
  });

  it('stops all active tracks and resets video element state on stop() and before re-starting', async () => {
    const first = createMockStream('Cam 1', 'cam-1');
    const second = createMockStream('Cam 2', 'cam-2');
    let count = 0;
    mockGetUserMedia(async () => {
      count += 1;
      return count === 1 ? first.stream : second.stream;
    });

    const videoEl = createMockVideoElement();
    const adapter = new CameraStreamAdapter(videoEl);

    adapter.setSelectedDeviceId('cam-1');
    await adapter.start();
    expect(first.track.stop).not.toHaveBeenCalled();

    // Switching cameras calls start() which must stop the previous stream first
    adapter.setSelectedDeviceId('cam-2');
    await adapter.start();
    expect(first.track.stop).toHaveBeenCalledTimes(1);
    expect(videoEl.srcObject).toBe(second.stream);

    adapter.stop();
    expect(second.track.stop).toHaveBeenCalledTimes(1);
    expect(videoEl.srcObject).toBeNull();
    expect(videoEl.pause).toHaveBeenCalled();
    expect(videoEl.removeAttribute).toHaveBeenCalledWith('src');
    expect(videoEl.load).toHaveBeenCalled();
    expect(adapter.hasActiveStream()).toBe(false);
  });

  it('stops superseded MediaStream tracks when concurrent start() calls overlap', async () => {
    const first = createMockStream('Slow Cam 1', 'cam-1');
    const second = createMockStream('Fast Cam 2', 'cam-2');

    let resolveFirst!: (s: MediaStream) => void;
    const firstPromise = new Promise<MediaStream>((resolve) => {
      resolveFirst = resolve;
    });

    let callIndex = 0;
    mockGetUserMedia(() => {
      callIndex += 1;
      if (callIndex === 1) {
        return firstPromise;
      }
      return Promise.resolve(second.stream);
    });

    const videoEl = createMockVideoElement();
    const adapter = new CameraStreamAdapter(videoEl);

    adapter.setSelectedDeviceId('cam-1');
    const p1 = adapter.start();

    // Before p1's getUserMedia resolves, user switches to cam-2 and calls start()
    adapter.setSelectedDeviceId('cam-2');
    const p2 = adapter.start();

    const activeStream = await p2;
    expect(activeStream).toBe(second.stream);

    // Now the stale first getUserMedia resolves
    resolveFirst(first.stream);
    await expect(p1).rejects.toMatchObject({ name: 'AbortError' });

    // The stale stream's hardware track MUST be stopped immediately and not overwrite cam-2
    expect(first.track.stop).toHaveBeenCalledTimes(1);
    expect(second.track.stop).not.toHaveBeenCalled();
    expect(adapter.getActiveDeviceId()).toBe('cam-2');
    expect(videoEl.srcObject).toBe(second.stream);
  });

  it('downscales frame to maxDimension (640px) preserving aspect ratio on captureFrameBlob', async () => {
    const videoEl = createMockVideoElement();
    (videoEl as any).videoWidth = 1920;
    (videoEl as any).videoHeight = 1080;

    let capturedWidth = 0;
    let capturedHeight = 0;
    let toBlobType = '';
    let toBlobQuality = 0;

    const mockCtx = {
      drawImage: vi.fn((_video: any, _sx: number, _sy: number, dw: number, dh: number) => {
        capturedWidth = dw;
        capturedHeight = dh;
      }),
      getImageData: vi.fn((_sx: number, _sy: number, w: number, h: number) => ({
        data: new Uint8ClampedArray(w * h * 4).map((_, i) => (i < (w * h * 2) ? 60 : 200)),
      })),
    };

    const mockCanvas = {
      width: 0,
      height: 0,
      getContext: vi.fn().mockReturnValue(mockCtx),
      toBlob: vi.fn((cb: (b: Blob) => void, type?: string, quality?: any) => {
        toBlobType = type || '';
        toBlobQuality = quality;
        cb(new Blob(['fake-jpeg'], { type: 'image/jpeg' }));
      }),
    };

    const originalDocument = (globalThis as any).document;
    (globalThis as any).document = {
      createElement: vi.fn().mockReturnValue(mockCanvas),
    };

    try {
      const adapter = new CameraStreamAdapter(videoEl);
      const { blob, quality, captureWidth, captureHeight } = await adapter.captureFrameBlob(640, 0.7);

      expect(blob).toBeInstanceOf(Blob);
      expect(quality.ready).toBe(true);
      expect(capturedWidth).toBe(640);
      expect(capturedHeight).toBe(360);
      expect(captureWidth).toBe(640);
      expect(captureHeight).toBe(360);
      expect(toBlobType).toBe('image/jpeg');
      expect(toBlobQuality).toBe(0.7);
    } finally {
      (globalThis as any).document = originalDocument;
    }
  });

  it('downscales portrait mobile frame (720x1280) so max dimension is 640px', async () => {
    const videoEl = createMockVideoElement();
    (videoEl as any).videoWidth = 720;
    (videoEl as any).videoHeight = 1280;

    let capturedWidth = 0;
    let capturedHeight = 0;

    const mockCtx = {
      drawImage: vi.fn((_video: any, _sx: number, _sy: number, dw: number, dh: number) => {
        capturedWidth = dw;
        capturedHeight = dh;
      }),
      getImageData: vi.fn((_sx: number, _sy: number, w: number, h: number) => ({
        data: new Uint8ClampedArray(w * h * 4).map((_, i) => (i < (w * h * 2) ? 60 : 200)),
      })),
    };

    const mockCanvas = {
      width: 0,
      height: 0,
      getContext: vi.fn().mockReturnValue(mockCtx),
      toBlob: vi.fn((cb: (b: Blob) => void) => {
        cb(new Blob(['fake-jpeg'], { type: 'image/jpeg' }));
      }),
    };

    const originalDocument = (globalThis as any).document;
    (globalThis as any).document = {
      createElement: vi.fn().mockReturnValue(mockCanvas),
    };

    try {
      const adapter = new CameraStreamAdapter(videoEl);
      const { blob, quality, captureWidth, captureHeight } = await adapter.captureFrameBlob(640, 0.7);

      expect(blob).toBeInstanceOf(Blob);
      expect(quality.ready).toBe(true);
      expect(capturedWidth).toBe(360);
      expect(capturedHeight).toBe(640);
      expect(captureWidth).toBe(360);
      expect(captureHeight).toBe(640);
    } finally {
      (globalThis as any).document = originalDocument;
    }
  });


  it('defaults to facingMode ideal environment when no deviceId is selected', async () => {
    const { stream } = createMockStream('Rear Camera', 'cam-rear');
    const calls: MediaStreamConstraints[] = [];

    mockGetUserMedia(async (constraints) => {
      calls.push(constraints);
      return stream;
    });

    const videoEl = createMockVideoElement();
    const adapter = new CameraStreamAdapter(videoEl);

    await adapter.start();

    expect(calls[0]).toEqual({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
    });
  });

  it('identifies camera_placeholder_or_too_dark when frame is black', () => {
    const videoEl = createMockVideoElement();
    const adapter = new CameraStreamAdapter(videoEl);
    const data = new Uint8ClampedArray(640 * 480 * 4); // all 0 (black)
    const mockCtx = {
      getImageData: vi.fn().mockReturnValue({ data }),
    } as unknown as CanvasRenderingContext2D;

    const quality = (adapter as any).analyzeCanvasQuality(mockCtx, 640, 480);
    expect(quality.ready).toBe(false);
    expect(quality.reason).toBe('camera_placeholder_or_too_dark');
  });

  it('identifies frame_has_too_little_detail when frame is flat gray with no contrast', () => {
    const videoEl = createMockVideoElement();
    const adapter = new CameraStreamAdapter(videoEl);
    const data = new Uint8ClampedArray(640 * 480 * 4);
    data.fill(128); // flat gray
    const mockCtx = {
      getImageData: vi.fn().mockReturnValue({ data }),
    } as unknown as CanvasRenderingContext2D;

    const quality = (adapter as any).analyzeCanvasQuality(mockCtx, 640, 480);
    expect(quality.ready).toBe(false);
    expect(quality.reason).toBe('frame_has_too_little_detail');
  });

  it('identifies ok when frame has good brightness and contrast', () => {
    const videoEl = createMockVideoElement();
    const adapter = new CameraStreamAdapter(videoEl);
    const data = new Uint8ClampedArray(640 * 480 * 4);
    for (let i = 0; i < data.length; i += 8) {
      data[i] = 200; data[i + 1] = 200; data[i + 2] = 200;
      data[i + 4] = 50; data[i + 5] = 50; data[i + 6] = 50;
    }
    const mockCtx = {
      getImageData: vi.fn().mockReturnValue({ data }),
    } as unknown as CanvasRenderingContext2D;

    const quality = (adapter as any).analyzeCanvasQuality(mockCtx, 640, 480);
    expect(quality.ready).toBe(true);
    expect(quality.reason).toBe('ok');
  });

  it('returns empty_frame from captureFrameBlob when video dimensions are 0x0', async () => {
    const videoEl = createMockVideoElement();
    (videoEl as any).videoWidth = 0;
    (videoEl as any).videoHeight = 0;
    const adapter = new CameraStreamAdapter(videoEl);

    const res = await adapter.captureFrameBlob(640, 0.7);
    expect(res.blob).toBeNull();
    expect(res.quality.ready).toBe(false);
    expect(res.quality.reason).toBe('empty_frame');
    expect(res.captureWidth).toBe(0);
    expect(res.captureHeight).toBe(0);
  });

  it('returns empty_frame from captureFrameBlob when canvas 2d context is null', async () => {
    const videoEl = createMockVideoElement();
    const mockCanvas = {
      width: 0,
      height: 0,
      getContext: vi.fn().mockReturnValue(null),
      toBlob: vi.fn(),
    };
    const originalDocument = (globalThis as any).document;
    (globalThis as any).document = { createElement: vi.fn().mockReturnValue(mockCanvas) };

    try {
      const adapter = new CameraStreamAdapter(videoEl);
      const res = await adapter.captureFrameBlob(640, 0.7);
      expect(mockCanvas.getContext).toHaveBeenCalledWith('2d', { willReadFrequently: true });
      expect(res.blob).toBeNull();
      expect(res.quality.ready).toBe(false);
      expect(res.quality.reason).toBe('empty_frame');
      expect(mockCanvas.toBlob).not.toHaveBeenCalled();
    } finally {
      (globalThis as any).document = originalDocument;
    }
  });

  it('rejects dark frame in captureFrameBlob with camera_placeholder_or_too_dark without calling toBlob', async () => {
    const videoEl = createMockVideoElement();
    const mockCtx = {
      drawImage: vi.fn(),
      getImageData: vi.fn((_sx: number, _sy: number, w: number, h: number) => ({
        data: new Uint8ClampedArray(w * h * 4),
      })),
    };
    const mockCanvas = {
      width: 0,
      height: 0,
      getContext: vi.fn().mockReturnValue(mockCtx),
      toBlob: vi.fn(),
    };
    const originalDocument = (globalThis as any).document;
    (globalThis as any).document = { createElement: vi.fn().mockReturnValue(mockCanvas) };

    try {
      const adapter = new CameraStreamAdapter(videoEl);
      const res = await adapter.captureFrameBlob(640, 0.7);
      expect(res.blob).toBeNull();
      expect(res.quality.ready).toBe(false);
      expect(res.quality.reason).toBe('camera_placeholder_or_too_dark');
      expect(mockCanvas.toBlob).not.toHaveBeenCalled();
    } finally {
      (globalThis as any).document = originalDocument;
    }
  });

  it('rejects low-contrast frame in captureFrameBlob with frame_has_too_little_detail without calling toBlob', async () => {
    const videoEl = createMockVideoElement();
    const mockCtx = {
      drawImage: vi.fn(),
      getImageData: vi.fn((_sx: number, _sy: number, w: number, h: number) => {
        const data = new Uint8ClampedArray(w * h * 4);
        data.fill(128);
        return { data };
      }),
    };
    const mockCanvas = {
      width: 0,
      height: 0,
      getContext: vi.fn().mockReturnValue(mockCtx),
      toBlob: vi.fn(),
    };
    const originalDocument = (globalThis as any).document;
    (globalThis as any).document = { createElement: vi.fn().mockReturnValue(mockCanvas) };

    try {
      const adapter = new CameraStreamAdapter(videoEl);
      const res = await adapter.captureFrameBlob(640, 0.7);
      expect(res.blob).toBeNull();
      expect(res.quality.ready).toBe(false);
      expect(res.quality.reason).toBe('frame_has_too_little_detail');
      expect(mockCanvas.toBlob).not.toHaveBeenCalled();
    } finally {
      (globalThis as any).document = originalDocument;
    }
  });

  it('returns empty_frame when getImageData buffer is empty', async () => {
    const videoEl = createMockVideoElement();
    const mockCtx = {
      drawImage: vi.fn(),
      getImageData: vi.fn().mockReturnValue({ data: new Uint8ClampedArray(0) }),
    };
    const mockCanvas = {
      width: 0,
      height: 0,
      getContext: vi.fn().mockReturnValue(mockCtx),
      toBlob: vi.fn(),
    };
    const originalDocument = (globalThis as any).document;
    (globalThis as any).document = { createElement: vi.fn().mockReturnValue(mockCanvas) };

    try {
      const adapter = new CameraStreamAdapter(videoEl);
      const res = await adapter.captureFrameBlob(640, 0.7);
      expect(res.blob).toBeNull();
      expect(res.quality.ready).toBe(false);
      expect(res.quality.reason).toBe('empty_frame');
    } finally {
      (globalThis as any).document = originalDocument;
    }
  });

  it('uses Rec.601 luma weights matching OpenCV BGR2GRAY', () => {
    const videoEl = createMockVideoElement();
    const adapter = new CameraStreamAdapter(videoEl);
    const data = new Uint8ClampedArray(100 * 100 * 4);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = 0;
      data[i + 1] = 0;
      data[i + 2] = 255;
      data[i + 3] = 255;
    }
    const mockCtx = {
      getImageData: vi.fn().mockReturnValue({ data }),
    } as unknown as CanvasRenderingContext2D;

    const quality = (adapter as any).analyzeCanvasQuality(mockCtx, 100, 100);
    expect(quality.meanLuma).toBeCloseTo(29.07, 1);
    expect(quality.ready).toBe(false);
    expect(quality.reason).toBe('frame_has_too_little_detail');
  });
});