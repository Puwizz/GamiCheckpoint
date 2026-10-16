export interface FrameQuality {
  ready: boolean;
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

  public async start(): Promise<MediaStream> {
    this.stop();
    const hasValidDevice =
      this.selectedDeviceId &&
      this.selectedDeviceId.trim() !== '' &&
      this.selectedDeviceId !== 'No cameras found';

    let stream: MediaStream | null = null;

    if (hasValidDevice) {
      try {
        // Attempt exact device constraint first to prevent silent fallback to other cameras
        stream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: this.selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } },
        });
      } catch (exactErr) {
        console.warn('GamiSenpai: Exact camera constraint failed, attempting ideal fallback:', exactErr);
      }
    }

    if (!stream) {
      const videoConstraints: MediaTrackConstraints = hasValidDevice
        ? { deviceId: { ideal: this.selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } };

      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints });
      } catch (primaryErr) {
        console.warn('GamiSenpai: Primary camera constraints failed, attempting fallback to generic user facingMode:', primaryErr);
        try {
          stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
        } catch (secondaryErr) {
          console.warn('GamiSenpai: FacingMode user failed, attempting fallback to bare video: true:', secondaryErr);
          stream = await navigator.mediaDevices.getUserMedia({ video: true });
        }
      }
    }

    this.stream = stream;
    this.videoElement.srcObject = this.stream;
    return this.stream;
  }

  public stop(): void {
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

  public isStreaming(): boolean {
    return this.stream !== null && this.videoElement.videoWidth > 0;
  }

  public captureFrameBlob(): Promise<{ blob: Blob | null; quality: FrameQuality }> {
    return new Promise((resolve) => {
      const vw = this.videoElement.videoWidth;
      const vh = this.videoElement.videoHeight;

      if (vw === 0 || vh === 0) {
        resolve({
          blob: null,
          quality: { ready: false, meanLuma: 0, contrast: 0, darkRatio: 1 },
        });
        return;
      }

      if (!this.captureCanvas) {
        this.captureCanvas = document.createElement('canvas');
      }
      if (this.captureCanvas.width !== vw || this.captureCanvas.height !== vh) {
        this.captureCanvas.width = vw;
        this.captureCanvas.height = vh;
        this.captureCtx = null;
      }
      if (!this.captureCtx) {
        this.captureCtx = this.captureCanvas.getContext('2d');
      }

      const ctx = this.captureCtx;
      if (!ctx) {
        resolve({
          blob: null,
          quality: { ready: false, meanLuma: 0, contrast: 0, darkRatio: 1 },
        });
        return;
      }

      ctx.drawImage(this.videoElement, 0, 0, vw, vh);
      const quality = this.analyzeCanvasQuality(ctx, vw, vh);

      if (!quality.ready) {
        resolve({ blob: null, quality });
        return;
      }

      this.captureCanvas.toBlob(
        (blob) => {
          resolve({ blob, quality });
        },
        'image/jpeg',
        0.8
      );
    });
  }

  private analyzeCanvasQuality(ctx: CanvasRenderingContext2D, width: number, height: number): FrameQuality {
    const imageData = ctx.getImageData(0, 0, width, height).data;
    const stride = Math.max(4, Math.floor(imageData.length / 4 / 7000) * 4);
    let samples = 0;
    let darkPixels = 0;
    let lumaSum = 0;
    let lumaSquaredSum = 0;

    for (let index = 0; index < imageData.length; index += stride) {
      const red = imageData[index];
      const green = imageData[index + 1];
      const blue = imageData[index + 2];
      const luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
      lumaSum += luma;
      lumaSquaredSum += luma * luma;
      if (luma < 25) darkPixels += 1;
      samples += 1;
    }

    const meanLuma = lumaSum / samples;
    const variance = Math.max(0, lumaSquaredSum / samples - meanLuma * meanLuma);
    const contrast = Math.sqrt(variance);
    const darkRatio = darkPixels / samples;
    const ready = !(
      (meanLuma < FRAME_READY_RULE.minMeanLuma && darkRatio > FRAME_READY_RULE.maxDarkRatio) ||
      contrast < FRAME_READY_RULE.minContrast
    );

    return { ready, meanLuma, contrast, darkRatio };
  }
}
