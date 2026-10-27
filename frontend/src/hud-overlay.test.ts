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

  it('maps valid polygon points to canvas coordinates and rejects degenerate polygons', () => {
    const canvas = createMockCanvas();
    const video = createMockVideo(); // 1280x720 on 640x480 canvas
    const renderer = new DetectionOverlayRenderer(canvas, video);

    const mapped = renderer.mapPolygonToCanvas([
      [320, 180],
      [960, 180],
      [640, 540],
    ]);
    expect(mapped).not.toBeNull();
    expect(mapped).toHaveLength(3);
    expect(mapped![0].x).toBeGreaterThanOrEqual(0);
    expect(mapped![0].y).toBeGreaterThanOrEqual(0);

    expect(renderer.mapPolygonToCanvas([[10, 10], [20, 20]])).toBeNull();
    expect(renderer.mapPolygonToCanvas([[NaN, 10], [20, 20], [30, 30]])).toBeNull();

    renderer.destroy();
  });

  it('draws polygon contour path on canvas when roi.polygon has at least 3 vertices', () => {
    const canvas = createMockCanvas();
    const video = createMockVideo();
    const renderer = new DetectionOverlayRenderer(canvas, video);
    const ctx = canvas.getContext('2d') as any;

    renderer.updateOverlay({
      roi: {
        applied: true,
        bbox: [200, 100, 600, 400],
        polygon: [
          [200, 100],
          [600, 100],
          [400, 400],
        ],
        method: 'yolo_detection',
      },
      state: 'tracking',
      stepNumber: 2,
      stepName: 'Kite Folds',
      confidence: 88,
    });

    (renderer as any).render();
    expect(ctx.closePath).toHaveBeenCalled();
    expect(ctx.fill).toHaveBeenCalled();
    expect(ctx.stroke).toHaveBeenCalled();
    // Option 1: Solid rectangular bounding box must NOT be drawn when polygon is present
    expect(ctx.strokeRect).not.toHaveBeenCalled();

    // Default friendly label does not include raw technical keywords like "yolo_detection" or "88%"
    const labelCalls = ctx.fillText.mock.calls.map((c: any[]) => c[0]);
    expect(labelCalls.some((txt: string) => txt.includes('ขั้นตอนที่ 2: Kite Folds'))).toBe(true);
    expect(labelCalls.some((txt: string) => txt.includes('yolo_detection'))).toBe(false);

    renderer.destroy();
  });

  it('renders debug info only when showDebugInfo is explicitly enabled', () => {
    const canvas = createMockCanvas();
    const video = createMockVideo();
    const renderer = new DetectionOverlayRenderer(canvas, video);
    const ctx = canvas.getContext('2d') as any;

    renderer.updateOverlay({
      roi: {
        applied: true,
        bbox: [200, 100, 600, 400],
        polygon: [
          [200, 100],
          [600, 100],
          [400, 400],
        ],
        method: 'white_paper',
      },
      state: 'tracking',
      stepNumber: 3,
      stepName: 'Bird Base',
      confidence: 95,
    });

    renderer.setShowDebugInfo(true);
    expect(renderer.isDebugInfoEnabled()).toBe(true);

    (renderer as any).render();
    const debugCalls = ctx.fillText.mock.calls.map((c: any[]) => c[0]);
    expect(debugCalls.some((txt: string) => txt.includes('95%') && txt.includes('White Paper'))).toBe(true);

    renderer.destroy();
  });

  it('renders viewfinder corners when polygon is absent (fallback bbox mode)', () => {
    const canvas = createMockCanvas();
    const video = createMockVideo();
    const renderer = new DetectionOverlayRenderer(canvas, video);
    const ctx = canvas.getContext('2d') as any;

    renderer.updateOverlay({
      roi: {
        applied: true,
        bbox: [150, 100, 500, 400],
        method: 'edges',
      },
      state: 'tracking',
      stepNumber: 1,
      stepName: 'Square Base',
    });

    (renderer as any).render();
    expect(ctx.fillRect).toHaveBeenCalled();
    expect(ctx.stroke).toHaveBeenCalled();
    // In fallback mode, only corner brackets are stroked; full strokeRect is omitted
    expect(ctx.strokeRect).not.toHaveBeenCalled();

    renderer.destroy();
  });

  it('correctly maps bbox and polygon when roi is computed on downscaled source frame (640x360)', () => {
    const canvas = createMockCanvas(); // 640x360
    (canvas.getBoundingClientRect as any).mockReturnValue({ width: 640, height: 360 });
    const video = { videoWidth: 1920, videoHeight: 1080 } as HTMLVideoElement;
    const renderer = new DetectionOverlayRenderer(canvas, video);

    const roi: ROIData = {
      applied: true,
      bbox: [100, 50, 500, 300],
      polygon: [
        [100, 50],
        [500, 50],
        [500, 300],
        [100, 300],
      ],
      frame_size: [640, 360],
      method: 'yolo_detection',
    };

    renderer.updateOverlay({ roi, state: 'tracking', stepNumber: 1 });

    const targetBox = (renderer as any).targetBox;
    expect(targetBox).not.toBeNull();
    expect(targetBox.x).toBeCloseTo(100, 1);
    expect(targetBox.y).toBeCloseTo(50, 1);
    expect(targetBox.width).toBeCloseTo(400, 1);
    expect(targetBox.height).toBeCloseTo(250, 1);

    const targetPoly = (renderer as any).currentPolygon;
    expect(targetPoly).not.toBeNull();
    expect(targetPoly[0].x).toBeCloseTo(100, 1);
    expect(targetPoly[0].y).toBeCloseTo(50, 1);

    renderer.destroy();
  });

  it('correctly maps coordinates using sourceDimensions parameter when roi.frame_size is absent', () => {
    const canvas = createMockCanvas(); // 640x360
    (canvas.getBoundingClientRect as any).mockReturnValue({ width: 640, height: 360 });
    const video = { videoWidth: 1920, videoHeight: 1080 } as HTMLVideoElement;
    const renderer = new DetectionOverlayRenderer(canvas, video);

    const roi: ROIData = {
      applied: true,
      bbox: [100, 50, 500, 300],
      method: 'yolo_detection',
    };

    renderer.updateOverlay({
      roi,
      state: 'tracking',
      stepNumber: 1,
      sourceDimensions: { width: 640, height: 360 },
    });

    const targetBox = (renderer as any).targetBox;
    expect(targetBox).not.toBeNull();
    expect(targetBox.x).toBeCloseTo(100, 1);
    expect(targetBox.y).toBeCloseTo(50, 1);
    expect(targetBox.width).toBeCloseTo(400, 1);
    expect(targetBox.height).toBeCloseTo(250, 1);

    renderer.destroy();
  });


  it('lazily resizes canvas on updateOverlay when initial dimensions were 0x0 due to hidden container', () => {
    const canvas = createMockCanvas();
    (canvas.getBoundingClientRect as any).mockReturnValue({ width: 0, height: 0 });
    const video = { videoWidth: 1280, videoHeight: 720 } as HTMLVideoElement;
    const renderer = new DetectionOverlayRenderer(canvas, video);

    // Initially 0x0 because container was display:none
    expect((renderer as any).width).toBe(0);
    expect((renderer as any).height).toBe(0);

    // Container becomes visible before updateOverlay
    (canvas.getBoundingClientRect as any).mockReturnValue({ width: 640, height: 360 });

    renderer.updateOverlay({
      roi: {
        applied: true,
        bbox: [100, 50, 400, 250],
        frame_size: [640, 360],
        method: 'yolo_detection',
      },
      state: 'tracking',
      stepNumber: 1,
    });

    expect((renderer as any).width).toBe(640);
    expect((renderer as any).height).toBe(360);
    expect((renderer as any).targetBox).not.toBeNull();

    renderer.destroy();
  });
});
