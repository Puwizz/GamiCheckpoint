import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { DetectionOverlayRenderer, formatRoiMethod } from './hud-overlay';
import { ROIData } from './types';

function createMockCanvas(): HTMLCanvasElement {
  const ctx = {
    setTransform: vi.fn(),
    clearRect: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    fillRect: vi.fn(),
    strokeRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    arc: vi.fn(),
    arcTo: vi.fn(),
    closePath: vi.fn(),
    fill: vi.fn(),
    stroke: vi.fn(),
    fillText: vi.fn(),
    measureText: vi.fn().mockReturnValue({ width: 80 }),
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    globalAlpha: 1,
    font: '',
    textBaseline: 'middle',
    textAlign: 'left',
  } as unknown as CanvasRenderingContext2D;

  return {
    width: 640,
    height: 480,
    getContext: vi.fn().mockReturnValue(ctx),
    getBoundingClientRect: vi.fn().mockReturnValue({ width: 640, height: 480 }),
  } as unknown as HTMLCanvasElement;
}

function createMockVideo(): HTMLVideoElement {
  return {
    videoWidth: 1280,
    videoHeight: 720,
  } as unknown as HTMLVideoElement;
}

describe('DetectionOverlayRenderer', () => {
  let originalResizeObserver: any;
  let originalWindow: any;
  let mockWindow: any;

  beforeEach(() => {
    originalResizeObserver = (globalThis as any).ResizeObserver;
    originalWindow = (globalThis as any).window;

    mockWindow = {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      devicePixelRatio: 1,
    };
    (globalThis as any).window = mockWindow;
  });

  afterEach(() => {
    (globalThis as any).ResizeObserver = originalResizeObserver;
    (globalThis as any).window = originalWindow;
    vi.restoreAllMocks();
  });

  it('formats roi method labels correctly', () => {
    expect(formatRoiMethod('white_paper')).toBe('White Paper');
    expect(formatRoiMethod('edges')).toBe('Edge Detection');
    expect(formatRoiMethod('grabcut')).toBe('GrabCut');
    expect(formatRoiMethod('center')).toBe('Center Fallback');
    expect(formatRoiMethod('custom_method')).toBe('custom_method');
    expect(formatRoiMethod(undefined)).toBe('');
  });

  it('attaches and disconnects ResizeObserver when ResizeObserver is available', () => {
    const observeFn = vi.fn();
    const disconnectFn = vi.fn();

    class MockResizeObserver {
      constructor(_callback: () => void) {}
      observe = observeFn;
      disconnect = disconnectFn;
    }

    mockWindow.ResizeObserver = MockResizeObserver;
    (globalThis as any).ResizeObserver = MockResizeObserver;

    const canvas = createMockCanvas();
    const video = createMockVideo();
    const renderer = new DetectionOverlayRenderer(canvas, video);

    expect(observeFn).toHaveBeenCalledWith(canvas);

    renderer.destroy();
    expect(disconnectFn).toHaveBeenCalledTimes(1);
    expect(canvas.getContext('2d')?.clearRect).toHaveBeenCalled();

    expect(() => renderer.destroy()).not.toThrow();
    expect(disconnectFn).toHaveBeenCalledTimes(1);
  });

  it('attaches window resize listener as fallback and removes it on destroy', () => {
    mockWindow.ResizeObserver = undefined;
    (globalThis as any).ResizeObserver = undefined;

    const canvas = createMockCanvas();
    const video = createMockVideo();
    const renderer = new DetectionOverlayRenderer(canvas, video);

    expect(mockWindow.addEventListener).toHaveBeenCalledWith('resize', expect.any(Function));

    renderer.destroy();
    expect(mockWindow.removeEventListener).toHaveBeenCalledWith('resize', expect.any(Function));

    expect(() => renderer.destroy()).not.toThrow();
  });

  it('handles resize updates properly', () => {
    mockWindow.devicePixelRatio = 2;

    const canvas = createMockCanvas();
    const video = createMockVideo();
    const renderer = new DetectionOverlayRenderer(canvas, video);

    (canvas.getBoundingClientRect as any).mockReturnValue({ width: 800, height: 600 });
    renderer.resize();

    expect(canvas.width).toBe(1600);
    expect(canvas.height).toBe(1200);
    expect(canvas.getContext('2d')?.setTransform).toHaveBeenCalledWith(2, 0, 0, 2, 0, 0);

    renderer.destroy();
  });

  it('clears target opacity on clear and reset', () => {
    const canvas = createMockCanvas();
    const video = createMockVideo();
    const renderer = new DetectionOverlayRenderer(canvas, video);

    const roi: ROIData = {
      applied: true,
      bbox: [100, 100, 300, 300],
      method: 'white_paper',
    };

    renderer.updateOverlay({ roi, state: 'tracking', stepNumber: 1, stepName: 'Fold 1' });
    renderer.clear();
    renderer.reset();

    expect(canvas.getContext('2d')?.clearRect).toHaveBeenCalled();
    renderer.destroy();
  });

  it('ignores invalid roi data and disables rendering when setEnabled is false', () => {
    const canvas = createMockCanvas();
    const video = createMockVideo();
    const renderer = new DetectionOverlayRenderer(canvas, video);

    renderer.updateOverlay({
      roi: { applied: false, bbox: [0, 0, 0, 0] },
      state: 'tracking',
    });

    renderer.setEnabled(false);
    expect(canvas.getContext('2d')?.clearRect).toHaveBeenCalled();

    renderer.destroy();
  });
});
