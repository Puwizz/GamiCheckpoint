export interface VideoElementTarget {
  src: string;
  playbackRate: number;
  loop: boolean;
  error?: MediaError | null;
  load(): void;
  play(): Promise<void>;
  addEventListener(
    type: string,
    listener: EventListenerOrEventListenerObject,
    options?: boolean | AddEventListenerOptions
  ): void;
  removeEventListener(
    type: string,
    listener: EventListenerOrEventListenerObject,
    options?: boolean | EventListenerOptions
  ): void;
}

export interface VideoLoaderCallbacks {
  onStart?: () => void;
  onSuccess?: () => void;
  onFallback?: (fallbackUrl: string) => void;
}

export class VideoStepLoader {
  private activeLoadId = 0;
  private activeCleanup: (() => void) | null = null;

  public load(
    videoElement: VideoElementTarget,
    videoUrl: string | undefined,
    fallbackImageUrl: string,
    playbackSpeed: number,
    isLooping: boolean,
    callbacks: VideoLoaderCallbacks = {},
    autoPlay: boolean = false
  ): number {
    const loadId = ++this.activeLoadId;

    if (this.activeCleanup) {
      this.activeCleanup();
      this.activeCleanup = null;
    }

    callbacks.onStart?.();

    if (!videoUrl) {
      callbacks.onFallback?.(fallbackImageUrl);
      return loadId;
    }

    videoElement.src = videoUrl;
    videoElement.playbackRate = playbackSpeed;
    videoElement.loop = isLooping;

    const onCanPlay = () => {
      if (loadId !== this.activeLoadId) return;
      cleanup();
      callbacks.onSuccess?.();
      if (autoPlay) {
        videoElement.play().catch(() => {});
      }
    };

    const onError = () => {
      if (loadId !== this.activeLoadId) return;
      cleanup();
      console.warn(`[VideoLoader] Video failed to load/decode on ${videoUrl}:`, {
        code: videoElement.error?.code,
        message: videoElement.error?.message,
        error: videoElement.error,
      });
      callbacks.onFallback?.(fallbackImageUrl);
    };

    const cleanup = () => {
      videoElement.removeEventListener('canplay', onCanPlay);
      videoElement.removeEventListener('error', onError);
      if (this.activeCleanup === cleanup) {
        this.activeCleanup = null;
      }
    };

    this.activeCleanup = cleanup;
    videoElement.addEventListener('canplay', onCanPlay, { once: true });
    videoElement.addEventListener('error', onError, { once: true });
    videoElement.load();

    return loadId;
  }

  public getActiveLoadId(): number {
    return this.activeLoadId;
  }

  public cancel(): void {
    this.activeLoadId++;
    if (this.activeCleanup) {
      this.activeCleanup();
      this.activeCleanup = null;
    }
  }
}
