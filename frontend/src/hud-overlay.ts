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

  private targetBox: BoxCoords | null = null;
  private currentBox: BoxCoords | null = null;
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
      primaryColor = '#4ade80';
      glowColor = 'rgba(74, 222, 128, 0.40)';
    } else if (this.meta.state === 'locked') {
      primaryColor = '#f87171';
      glowColor = 'rgba(248, 113, 113, 0.35)';
    } else if (this.meta.state === 'unrecognized') {
      primaryColor = '#fbbf24';
      glowColor = 'rgba(251, 191, 36, 0.35)';
    }

    // Box fill
    this.ctx.fillStyle = glowColor.replace(/0\.\d+\)/, '0.04)');
    this.ctx.fillRect(x, y, width, height);

    // Box outline
    this.ctx.strokeStyle = primaryColor;
    this.ctx.lineWidth = 1.5;
    this.ctx.strokeRect(x, y, width, height);

    // Corners
    const cl = Math.min(this.cornerLength, width / 2 - 2, height / 2 - 2);
    this.ctx.lineWidth = 3;
    this.ctx.strokeStyle = primaryColor;
    this.ctx.beginPath();
    this.ctx.moveTo(x, y + cl); this.ctx.lineTo(x, y); this.ctx.lineTo(x + cl, y);
    this.ctx.moveTo(x + width - cl, y); this.ctx.lineTo(x + width, y); this.ctx.lineTo(x + width, y + cl);
    this.ctx.moveTo(x, y + height - cl); this.ctx.lineTo(x, y + height); this.ctx.lineTo(x + cl, y + height);
    this.ctx.moveTo(x + width - cl, y + height); this.ctx.lineTo(x + width, y + height); this.ctx.lineTo(x + width, y + height - cl);
    this.ctx.stroke();

    // Text Label Pill
    let labelText = '';
    if (this.meta.state === 'unrecognized') {
      labelText = 'Unrecognized';
    } else if (this.meta.stepNumber) {
      const stateSuffix = this.meta.state === 'locked' ? ' (Locked)' : this.meta.state === 'matched' ? ' (Matched)' : '';
      const streakSuffix = this.meta.streakText ? ` ${this.meta.streakText}` : '';
      labelText = `Step ${this.meta.stepNumber}: ${this.meta.stepName}${streakSuffix}${stateSuffix}`;
    } else {
      labelText = 'Origami Paper';
    }

    const confText = this.meta.confidence > 0 ? `${this.meta.confidence.toFixed(0)}%` : '';
    const methodText = this.meta.method ? `• ${this.meta.method}` : '';
    this.ctx.font = '600 12px "Outfit", sans-serif';
    const rightText = confText ? `${confText} ${methodText}`.trim() : methodText.replace(/^•\s*/, '');
    const fullString = `${labelText}   ${rightText}`.trim();
    const textMetrics = this.ctx.measureText(fullString);
    const pillWidth = Math.max(140, textMetrics.width + 30);
    const pillHeight = 24;
    const pillX = Math.max(6, Math.min(x, this.width - pillWidth - 6));
    const pillY = y >= pillHeight + 6 ? y - pillHeight - 4 : y + 6;

    this.ctx.fillStyle = 'rgba(15, 23, 42, 0.90)';
    this.ctx.strokeStyle = primaryColor;
    this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    this.drawRoundedRect(pillX, pillY, pillWidth, pillHeight, 6);
    this.ctx.fill();
    this.ctx.stroke();

    // Dot indicator
    const dotX = pillX + 12;
    const dotY = pillY + pillHeight / 2;
    this.ctx.fillStyle = primaryColor;
    this.ctx.beginPath();
    this.ctx.arc(dotX, dotY, 3.5, 0, Math.PI * 2);
    this.ctx.fill();

    // Text
    this.ctx.fillStyle = '#f8fafc';
    this.ctx.textBaseline = 'middle';
    this.ctx.textAlign = 'left';
    this.ctx.fillText(labelText, dotX + 8, dotY);

    if (rightText) {
      const confX = pillX + pillWidth - 8;
      this.ctx.textAlign = 'right';
      this.ctx.fillStyle = primaryColor;
      this.ctx.fillText(rightText, confX, dotY);
    }

    this.ctx.restore();
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
