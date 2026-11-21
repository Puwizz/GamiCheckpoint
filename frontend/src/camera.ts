import { FrameReadinessReason } from './types';

export type FrameUnreadyReason = FrameReadinessReason;

export interface FrameQuality {
  ready: boolean;
  reason: FrameUnreadyReason;
  meanLuma: number;
  contrast: number;
  darkRatio: number;
}

const FRAME_READY_RULE = {
  minMeanLuma: 22,
  maxDarkRatio: 0.7,
  minContrast: 6,
};

export class CameraStreamAdapter {
  private videoElement: HTMLVideoElement;
  private stream: MediaStream | null = null;
  private selectedDeviceId: string = '';
  private captureCanvas: HTMLCanvasElement | null = null;
  private captureCtx: CanvasRenderingContext2D | null = null;
  private startGeneration: number = 0;

  constructor(videoElement: HTMLVideoElement) {
    this.videoElement = videoElement;
  }

  public async requestPermission(): Promise<boolean> {
    if (!navigator.mediaDevices?.getUserMedia) return false;
    try {
      const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
      tempStream.getTracks().forEach((track) => track.stop());
      return true;
    } catch {
      return false;
    }
  }

  public async getAvailableVideoDevices(): Promise<MediaDeviceInfo[]> {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.filter((device) => device.kind === 'videoinput');
    } catch {
      return [];
    }
  }

  public setSelectedDeviceId(id: string): void {
    this.selectedDeviceId = id;
  }

  public getActiveTrackLabel(): string {
    return this.stream?.getVideoTracks()[0]?.label || '';
  }

  public getActiveDeviceId(): string {
    return this.stream?.getVideoTracks()[0]?.getSettings?.()?.deviceId || '';
  }

  private async acquireStreamForGeneration(
    constraints: MediaStreamConstraints,
    generation: number
  ): Promise<MediaStream> {
    const acquired = await navigator.mediaDevices.getUserMedia(constraints);
    if (this.startGeneration !== generation) {
      acquired.getTracks().forEach((track) => track.stop());
      const abortErr = new Error('Camera start superseded by a newer operation');
      abortErr.name = 'AbortError';
      throw abortErr;
    }
    return acquired;
  }

  public async start(): Promise<MediaStream> {
    this.stop();
    const generation = this.startGeneration;
    const hasValidDevice = Boolean(this.selectedDeviceId && this.selectedDeviceId.trim() !== '');

    let stream: MediaStream | null = null;

    if (hasValidDevice) {
      try {
        // Attempt exact device constraint first to prevent silent fallback to other cameras
        stream = await this.acquireStreamForGeneration(
          {
            video: { deviceId: { exact: this.selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } },
          },
          generation
        );
      } catch (exactErr: unknown) {
        if ((exactErr as { name?: string })?.name === 'AbortError') throw exactErr;
        console.warn('GamiSenpai: Exact camera constraint failed, attempting ideal fallback:', exactErr);
      }
    }

    if (!stream) {
      const videoConstraints: MediaTrackConstraints = hasValidDevice
        ? { deviceId: { ideal: this.selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } };

      try {
        stream = await this.acquireStreamForGeneration({ video: videoConstraints }, generation);
      } catch (primaryErr: unknown) {
        if ((primaryErr as { name?: string })?.name === 'AbortError') throw primaryErr;
        console.warn('GamiSenpai: Primary camera constraints failed, attempting fallback to generic facingMode:', primaryErr);
        try {
          stream = await this.acquireStreamForGeneration({ video: { facingMode: { ideal: 'environment' } } }, generation);
        } catch (secondaryErr: unknown) {
          if ((secondaryErr as { name?: string })?.name === 'AbortError') throw secondaryErr;
          console.warn('GamiSenpai: FacingMode ideal environment failed, attempting fallback to bare video: true:', secondaryErr);
          stream = await this.acquireStreamForGeneration({ video: true }, generation);
        }
      }
    }

    this.stream = stream;
    this.videoElement.srcObject = this.stream;
    return this.stream;
  }

  public stop(): void {
    this.startGeneration += 1;
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    this.videoElement.srcObject = null;
    this.videoElement.pause();
    this.videoElement.removeAttribute('src');
    this.videoElement.load();
    this.captureCanvas = null;
    this.captureCtx = null;
  }

  public hasActiveStream(): boolean {
    return this.stream !== null;
  }

  public isStreaming(): boolean {
    return this.stream !== null && this.videoElement.videoWidth > 0;
  }

  public captureFrameBlob(
    maxDimension: number = 640,
    quality: number = 0.7
  ): Promise<{ blob: Blob | null; quality: FrameQuality; captureWidth: number; captureHeight: number }> {
    return new Promise((resolve) => {
      const vw = this.videoElement.videoWidth;
      const vh = this.videoElement.videoHeight;

      if (vw === 0 || vh === 0) {
        resolve({
          blob: null,
          quality: { ready: false, reason: 'empty_frame', meanLuma: 0, contrast: 0, darkRatio: 1 },
          captureWidth: 0,
          captureHeight: 0,
        });
        return;
      }

      let targetW = vw;
      let targetH = vh;
      if (maxDimension > 0 && (vw > maxDimension || vh > maxDimension)) {
        if (vw >= vh) {
          targetW = maxDimension;
          targetH = Math.max(1, Math.round((vh / vw) * maxDimension));
        } else {
          targetH = maxDimension;
          targetW = Math.max(1, Math.round((vw / vh) * maxDimension));
        }
      }

      if (!this.captureCanvas) {
        this.captureCanvas = document.createElement('canvas');
      }
      if (this.captureCanvas.width !== targetW || this.captureCanvas.height !== targetH) {
        this.captureCanvas.width = targetW;
        this.captureCanvas.height = targetH;
        this.captureCtx = null;
      }
      if (!this.captureCtx) {
        this.captureCtx = this.captureCanvas.getContext('2d', { willReadFrequently: true });
      }

      const ctx = this.captureCtx;
      if (!ctx) {
        resolve({
          blob: null,
          quality: { ready: false, reason: 'empty_frame', meanLuma: 0, contrast: 0, darkRatio: 1 },
          captureWidth: 0,
          captureHeight: 0,
        });
        return;
      }

      ctx.drawImage(this.videoElement, 0, 0, targetW, targetH);
      const frameQuality = this.analyzeCanvasQuality(ctx, targetW, targetH);

      if (!frameQuality.ready) {
        resolve({ blob: null, quality: frameQuality, captureWidth: targetW, captureHeight: targetH });
        return;
      }

      this.captureCanvas.toBlob(
        (blob) => {
          resolve({ blob, quality: frameQuality, captureWidth: targetW, captureHeight: targetH });
        },
        'image/jpeg',
        quality
      );
    });
  }

  private analyzeCanvasQuality(ctx: CanvasRenderingContext2D, width: number, height: number): FrameQuality {
    const imageData = ctx.getImageData(0, 0, width, height).data;
    if (!imageData || imageData.length < 4) {
      return { ready: false, reason: 'empty_frame', meanLuma: 0, contrast: 0, darkRatio: 1 };
    }

    const stride = Math.max(4, Math.floor(imageData.length / 4 / 7000) * 4);
    let samples = 0;
    let darkPixels = 0;
    let lumaSum = 0;
    let lumaSquaredSum = 0;

    for (let index = 0; index < imageData.length; index += stride) {
      const red = imageData[index];
      const green = imageData[index + 1];
      const blue = imageData[index + 2];
      // Use Rec.601 weights to match OpenCV cv2.cvtColor(..., cv2.COLOR_BGR2GRAY) on backend
      const luma = 0.299 * red + 0.587 * green + 0.114 * blue;
      lumaSum += luma;
      lumaSquaredSum += luma * luma;
      if (luma < 25) darkPixels += 1;
      samples += 1;
    }

    if (samples === 0) {
      return { ready: false, reason: 'empty_frame', meanLuma: 0, contrast: 0, darkRatio: 1 };
    }

    const meanLuma = lumaSum / samples;
    const variance = Math.max(0, lumaSquaredSum / samples - meanLuma * meanLuma);
    const contrast = Math.sqrt(variance);
    const darkRatio = darkPixels / samples;

    let reason: FrameUnreadyReason = 'ok';
    if (meanLuma < FRAME_READY_RULE.minMeanLuma && darkRatio > FRAME_READY_RULE.maxDarkRatio) {
      reason = 'camera_placeholder_or_too_dark';
    } else if (contrast < FRAME_READY_RULE.minContrast) {
      reason = 'frame_has_too_little_detail';
    }
    const ready = reason === 'ok';

    return { ready, reason, meanLuma, contrast, darkRatio };
  }
}
