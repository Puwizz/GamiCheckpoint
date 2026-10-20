import { ROIData, STEPS } from './types';

export interface OverlayMeta {
  stepNumber: number | null;
  stepName: string;
  streakText: string;
  confidence: number;
  method: string;
  fusionMode: string;
  state: 'tracking' | 'matched' | 'locked' | 'unrecognized' | 'idle';
}

export interface BoxCoords {
  x: number;
  y: number;
  width: number;
  height: number;
}

function lerp(start: number, end: number, factor: number): number {
  return start + (end - start) * factor;
}

const ROI_METHOD_LABELS: Record<string, string> = {
  white_paper: 'White Paper',
  edges: 'Edge Detection',
  grabcut: 'GrabCut',
  center: 'Center Fallback',
};

export function formatRoiMethod(methodKey?: string): string {
  if (!methodKey) return '';
  return ROI_METHOD_LABELS[methodKey] || methodKey;
}

export class DetectionOverlayRenderer {
  private canvas: HTMLCanvasElement | null;
  private ctx: CanvasRenderingContext2D | null;
  private video: HTMLVideoElement | null;
  private enabled: boolean = true;
  private showDebugInfo: boolean = false;

  private targetBox: BoxCoords | null = null;
  private currentBox: BoxCoords | null = null;
  private currentPolygon: { x: number; y: number }[] | null = null;
  private targetOpacity: number = 0;
  private currentOpacity: number = 0;

  private meta: OverlayMeta = {
    stepNumber: null,
    stepName: '',
    streakText: '',
    confidence: 0,
    method: '',
    fusionMode: '',
    state: 'idle',
  };

  private width: number = 0;
  private height: number = 0;
  private dpr: number = 1;
  private animFrameId: number | null = null;
  private cornerLength: number = 18;
  private resizeObserver: ResizeObserver | null = null;
  private windowResizeHandler: (() => void) | null = null;

  constructor(canvas: HTMLCanvasElement | null, videoElement: HTMLVideoElement | null) {
    this.canvas = canvas;
    this.ctx = canvas ? canvas.getContext('2d') : null;
    this.video = videoElement;

    if (this.canvas) {
      this.resize();
      this.setupResizeObserver();
      this.startRenderLoop();
    }
  }

  private setupResizeObserver(): void {
    this.cleanupResizeObserver();
    const win = typeof window !== 'undefined' ? window : (globalThis as any).window;
    const ResizeObserverClass = (win && win.ResizeObserver) || (globalThis as any).ResizeObserver;
    if (ResizeObserverClass && this.canvas) {
      const observer = new ResizeObserverClass(() => this.resize());
      observer.observe(this.canvas);
      this.resizeObserver = observer;
    } else if (win && typeof win.addEventListener === 'function') {
      this.windowResizeHandler = () => this.resize();
      win.addEventListener('resize', this.windowResizeHandler);
    }
  }

  private cleanupResizeObserver(): void {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    const win = typeof window !== 'undefined' ? window : (globalThis as any).window;
    if (this.windowResizeHandler && win && typeof win.removeEventListener === 'function') {
      win.removeEventListener('resize', this.windowResizeHandler);
      this.windowResizeHandler = null;
    }
  }

  public resize(): void {
    if (!this.canvas || !this.ctx) return;
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const win = typeof window !== 'undefined' ? window : (globalThis as any).window;
    this.dpr = (win && win.devicePixelRatio) || 1;
    this.width = rect.width;
    this.height = rect.height;
    this.canvas.width = Math.round(rect.width * this.dpr);
    this.canvas.height = Math.round(rect.height * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  public reset(): void {
    this.targetOpacity = 0;
    this.currentOpacity = 0;
    this.currentBox = null;
    this.targetBox = null;
    this.currentPolygon = null;
    this.stopRenderLoop();
    if (this.ctx) {
      this.ctx.clearRect(0, 0, this.width, this.height);
    }
  }

  public destroy(): void {
    this.reset();
    this.cleanupResizeObserver();
  }

  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) {
      this.reset();
    } else if (this.targetBox) {
      this.targetOpacity = 1;
      this.startRenderLoop();
    }
  }

  public updateFromAPI(
    data: { roi?: ROIData; recognized?: boolean; step_number?: number | null; prediction?: string; confidence?: number; step_name?: string },
    targetStep: number | null,
    stateOverride: 'tracking' | 'matched' | 'locked' | 'unrecognized' | null = null,
    streakText: string = ''
  ): void {
    if (!data || !data.roi || !data.roi.applied) {
      this.clear();
      return;
    }

    const detectedStepNum = data.step_number ?? null;

    // Suppress HUD box completely when frame is unrecognized or Non-Origami
    if (data.recognized === false || detectedStepNum === 0 || data.prediction === 'Chk0_NonOrigami') {
      this.clear();
      return;
    }

    let state = stateOverride || 'tracking';
    if (!stateOverride) {
      if (detectedStepNum !== null && targetStep !== null) {
        state = detectedStepNum === targetStep ? 'tracking' : 'locked';
      }
    }

    const stepName = data.step_name || (detectedStepNum ? STEPS[detectedStepNum - 1]?.name : '');
    this.updateOverlay({
      roi: data.roi,
      state,
      stepNumber: detectedStepNum,
      stepName,
      streakText,
      confidence: data.confidence || 0,
    });
  }

  public updateOverlay({
    roi,
    state = 'tracking',
    stepNumber = null,
    stepName = '',
    streakText = '',
    confidence = 0,
  }: {
    roi: ROIData;
    state?: 'tracking' | 'matched' | 'locked' | 'unrecognized' | 'idle';
    stepNumber?: number | null;
    stepName?: string;
    streakText?: string;
    confidence?: number;
  }): void {
    if (!this.enabled || !roi || !this.video || !this.video.videoWidth || !this.video.videoHeight) {
      this.clear();
      return;
    }

    if (!roi.applied || !roi.bbox || roi.bbox.length < 4) {
      this.clear();
      return;
    }

    const mapped = this.mapBBoxToCanvas(roi.bbox);
    if (!mapped || mapped.width <= 0 || mapped.height <= 0) {
      this.clear();
      return;
    }

    this.targetBox = mapped;
    if (!this.currentBox) {
      this.currentBox = { ...mapped };
    }
    this.currentPolygon = roi.polygon ? this.mapPolygonToCanvas(roi.polygon) : null;
    this.targetOpacity = 1;
    this.startRenderLoop();

    this.meta = {
      stepNumber,
      stepName,
      streakText,
      confidence: Number(confidence || 0),
      method: formatRoiMethod(roi.method || 'Auto-Crop'),
      fusionMode: roi.fusion_mode || '',
      state,
    };
  }

  public clear(): void {
    this.targetOpacity = 0;
    this.currentPolygon = null;
  }

  public mapPolygonToCanvas(polygon: [number, number][]): { x: number; y: number }[] | null {
    if (!this.video || !this.width || !this.height || !Array.isArray(polygon)) return null;

    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    if (vw === 0 || vh === 0) return null;

    const cw = this.width;
    const ch = this.height;

    const scale = Math.max(cw / vw, ch / vh);
    const renderW = vw * scale;
    const renderH = vh * scale;
    const offsetX = (cw - renderW) / 2;
    const offsetY = (ch - renderH) / 2;

    const points: { x: number; y: number }[] = [];
    for (const pt of polygon) {
      if (!Array.isArray(pt) || pt.length < 2) continue;
      const px = Number(pt[0]);
      const py = Number(pt[1]);
      if (!Number.isFinite(px) || !Number.isFinite(py)) return null;

      const x = Math.max(0, Math.min(cw, px * scale + offsetX));
      const y = Math.max(0, Math.min(ch, py * scale + offsetY));
      points.push({ x, y });
    }

    if (points.length < 3) return null;
    return points;
  }

  private mapBBoxToCanvas(bbox: [number, number, number, number]): BoxCoords | null {
    if (!this.video || !this.width || !this.height) return null;

    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    if (vw === 0 || vh === 0) return null;

    const cw = this.width;
    const ch = this.height;

    const scale = Math.max(cw / vw, ch / vh);
    const renderW = vw * scale;
    const renderH = vh * scale;
    const offsetX = (cw - renderW) / 2;
    const offsetY = (ch - renderH) / 2;

    const [bx1, by1, bx2, by2] = bbox;
    const x = Math.max(0, bx1 * scale + offsetX);
    const y = Math.max(0, by1 * scale + offsetY);
    const width = Math.min(cw - x, (bx2 - bx1) * scale);
    const height = Math.min(ch - y, (by2 - by1) * scale);

    return { x, y, width, height };
  }

  private startRenderLoop(): void {
    if (this.animFrameId !== null) return;
    if (typeof requestAnimationFrame !== 'function') return;
    const render = () => {
      this.render();
      if (this.currentOpacity > 0.01 || this.targetOpacity > 0.01) {
        this.animFrameId = requestAnimationFrame(render);
      } else {
        this.stopRenderLoop();
        if (this.ctx) this.ctx.clearRect(0, 0, this.width, this.height);
      }
    };
    this.animFrameId = requestAnimationFrame(render);
  }

  public stopRenderLoop(): void {
    if (this.animFrameId !== null) {
      if (typeof cancelAnimationFrame === 'function') {
        cancelAnimationFrame(this.animFrameId);
      }
      this.animFrameId = null;
    }
  }

  private render(): void {
    if (!this.ctx || !this.enabled) return;

    this.currentOpacity = lerp(this.currentOpacity, this.targetOpacity, 0.15);

    if (this.currentBox && this.targetBox) {
      this.currentBox.x = lerp(this.currentBox.x, this.targetBox.x, 0.22);
      this.currentBox.y = lerp(this.currentBox.y, this.targetBox.y, 0.22);
      this.currentBox.width = lerp(this.currentBox.width, this.targetBox.width, 0.22);
      this.currentBox.height = lerp(this.currentBox.height, this.targetBox.height, 0.22);
    }

    this.ctx.clearRect(0, 0, this.width, this.height);
    if (this.currentOpacity <= 0.02 || !this.currentBox) return;

    const { x, y, width, height } = this.currentBox;
    if (width <= 0 || height <= 0) return;

    this.ctx.save();
    this.ctx.globalAlpha = this.currentOpacity;

    let primaryColor = '#38bdf8';
    let glowColor = 'rgba(56, 189, 248, 0.35)';

    if (this.meta.state === 'matched') {
      primaryColor = '#22c55e';
      glowColor = 'rgba(34, 197, 94, 0.35)';
    } else if (this.meta.state === 'locked') {
      primaryColor = '#fb7185';
      glowColor = 'rgba(251, 113, 133, 0.30)';
    } else if (this.meta.state === 'unrecognized') {
      primaryColor = '#f59e0b';
      glowColor = 'rgba(245, 158, 11, 0.30)';
    }

    // Option 1 (Clean Viewfinder): If polygon exists, omit rectangular bounding box entirely!
    if (this.currentPolygon && this.currentPolygon.length >= 3) {
      // 1. Soft glassmorphism holographic fill
      this.ctx.beginPath();
      this.ctx.moveTo(this.currentPolygon[0].x, this.currentPolygon[0].y);
      for (let i = 1; i < this.currentPolygon.length; i++) {
        this.ctx.lineTo(this.currentPolygon[i].x, this.currentPolygon[i].y);
      }
      this.ctx.closePath();
      this.ctx.fillStyle = glowColor.replace(/0\.\d+\)/, '0.10)');
      this.ctx.fill();

      // 2. Smooth luminous AR stroke with soft ambient glow
      this.ctx.save();
      this.ctx.shadowColor = primaryColor;
      this.ctx.shadowBlur = 10;
      this.ctx.lineWidth = 2.2;
      this.ctx.lineJoin = 'round';
      this.ctx.lineCap = 'round';
      this.ctx.strokeStyle = primaryColor;
      this.ctx.stroke();
      this.ctx.restore();

      // 3. Subtle origami vertex nodes (clean CAD / AR touch)
      for (const pt of this.currentPolygon) {
        this.ctx.beginPath();
        this.ctx.arc(pt.x, pt.y, 2.5, 0, Math.PI * 2);
        this.ctx.fillStyle = primaryColor;
        this.ctx.fill();
      }
    } else {
      // Fallback: If no polygon available, render minimalist camera viewfinder brackets (no heavy solid box)
      this.ctx.fillStyle = glowColor.replace(/0\.\d+\)/, '0.04)');
      this.ctx.fillRect(x, y, width, height);

      const cl = Math.min(this.cornerLength, width / 2 - 2, height / 2 - 2);
      this.ctx.lineWidth = 2.5;
      this.ctx.lineCap = 'round';
      this.ctx.lineJoin = 'round';
      this.ctx.strokeStyle = primaryColor;
      this.ctx.beginPath();
      this.ctx.moveTo(x, y + cl); this.ctx.lineTo(x, y); this.ctx.lineTo(x + cl, y);
      this.ctx.moveTo(x + width - cl, y); this.ctx.lineTo(x + width, y); this.ctx.lineTo(x + width, y + cl);
      this.ctx.moveTo(x, y + height - cl); this.ctx.lineTo(x, y + height); this.ctx.lineTo(x + cl, y + height);
      this.ctx.moveTo(x + width - cl, y + height); this.ctx.lineTo(x + width, y + height); this.ctx.lineTo(x + width, y + height - cl);
      this.ctx.stroke();
    }

    // Anchor floating pill smoothly right above paper contour (or bbox)
    let anchorTopY = y;
    let anchorLeftX = x;

    if (this.currentPolygon && this.currentPolygon.length >= 3) {
      let minY = this.currentPolygon[0].y;
      let minX = this.currentPolygon[0].x;
      for (let i = 1; i < this.currentPolygon.length; i++) {
        if (this.currentPolygon[i].y < minY) minY = this.currentPolygon[i].y;
        if (this.currentPolygon[i].x < minX) minX = this.currentPolygon[i].x;
      }
      anchorTopY = minY;
      anchorLeftX = minX;
    }

    // Friendly label text (removes confusing technical jargon like yolo_detection and % by default)
    let labelText = '';
    if (this.meta.state === 'unrecognized') {
      labelText = 'จัดตำแหน่งกระดาษในกล้อง';
    } else if (this.meta.stepNumber) {
      const stepTitle = this.meta.stepName || `ขั้นตอนที่ ${this.meta.stepNumber}`;
      if (this.meta.state === 'matched') {
        const streakSuffix = this.meta.streakText ? ` (${this.meta.streakText})` : '';
        labelText = `✓ ขั้นตอนที่ ${this.meta.stepNumber}: ${stepTitle}${streakSuffix}`;
      } else if (this.meta.state === 'locked') {
        labelText = `ตรวจพบ: ขั้นตอนที่ ${this.meta.stepNumber} (${stepTitle})`;
      } else {
        const streakSuffix = this.meta.streakText ? ` (${this.meta.streakText})` : '';
        labelText = `ขั้นตอนที่ ${this.meta.stepNumber}: ${stepTitle}${streakSuffix}`;
      }
    } else {
      labelText = 'กระดาษโอริกามิ';
    }

    // Optional debug text for developer / advanced mode
    let debugText = '';
    if (this.showDebugInfo) {
      const confPart = this.meta.confidence > 0 ? `${Math.round(this.meta.confidence)}%` : '';
      const methodPart = this.meta.method ? formatRoiMethod(this.meta.method) : '';
      debugText = [confPart, methodPart].filter(Boolean).join(' • ');
    }

    this.ctx.font = '600 12.5px "Outfit", "Prompt", -apple-system, BlinkMacSystemFont, sans-serif';
    const mainMetrics = this.ctx.measureText(labelText);
    const debugMetrics = debugText ? this.ctx.measureText(debugText) : { width: 0 };

    const pillHeight = 28;
    const dotSpacing = 26;
    const debugSpacing = debugText ? debugMetrics.width + 16 : 0;
    const pillWidth = Math.max(130, mainMetrics.width + dotSpacing + debugSpacing + 18);
    const pillX = Math.max(8, Math.min(anchorLeftX, this.width - pillWidth - 8));
    const pillY = anchorTopY >= pillHeight + 8 ? anchorTopY - pillHeight - 6 : anchorTopY + 8;

    // Dark glass pill background
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    this.ctx.strokeStyle = glowColor.replace(/0\.\d+\)/, '0.40)');
    this.ctx.lineWidth = 1;
    this.ctx.shadowColor = 'rgba(0, 0, 0, 0.35)';
    this.ctx.shadowBlur = 6;
    this.ctx.shadowOffsetY = 2;
    this.ctx.beginPath();
    this.drawRoundedRect(pillX, pillY, pillWidth, pillHeight, 14);
    this.ctx.fill();
    this.ctx.stroke();
    this.ctx.restore();

    // Glowing status dot
    const dotX = pillX + 14;
    const dotY = pillY + pillHeight / 2;

    this.ctx.beginPath();
    this.ctx.arc(dotX, dotY, 6, 0, Math.PI * 2);
    this.ctx.strokeStyle = glowColor;
    this.ctx.lineWidth = 1.2;
    this.ctx.stroke();

    this.ctx.beginPath();
    this.ctx.arc(dotX, dotY, 3.5, 0, Math.PI * 2);
    this.ctx.fillStyle = primaryColor;
    this.ctx.fill();

    // Text label
    this.ctx.fillStyle = '#f8fafc';
    this.ctx.textBaseline = 'middle';
    this.ctx.textAlign = 'left';
    this.ctx.fillText(labelText, dotX + 11, dotY);

    if (debugText) {
      const dbgX = pillX + pillWidth - 10;
      this.ctx.textAlign = 'right';
      this.ctx.fillStyle = primaryColor;
      this.ctx.fillText(debugText, dbgX, dotY);
    }

    this.ctx.restore();
  }

  public setShowDebugInfo(show: boolean): void {
    this.showDebugInfo = show;
  }

  public isDebugInfoEnabled(): boolean {
    return this.showDebugInfo;
  }

  private drawRoundedRect(x: number, y: number, w: number, h: number, r: number): void {
    const radius = Math.min(r, w / 2, h / 2);
    this.ctx?.beginPath();
    this.ctx?.moveTo(x + radius, y);
    this.ctx?.lineTo(x + w - radius, y);
    this.ctx?.arcTo(x + w, y, x + w, y + radius, radius);
    this.ctx?.lineTo(x + w, y + h - radius);
    this.ctx?.arcTo(x + w, y + h, x + w - radius, y + h, radius);
    this.ctx?.lineTo(x + radius, y + h);
    this.ctx?.arcTo(x, y + h, x, y + h - radius, radius);
    this.ctx?.lineTo(x, y + radius);
    this.ctx?.arcTo(x, y, x + radius, y, radius);
    this.ctx?.closePath();
  }
}
