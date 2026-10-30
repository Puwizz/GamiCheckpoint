export interface VideoElementTarget {
  src: string;
  playbackRate: number;
  loop: boolean;
  readyState?: number;
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

export function isSameVideoUrl(currentSrc: string | undefined, targetUrl: string): boolean {
  if (!currentSrc || !targetUrl) return false;
  if (currentSrc === targetUrl) return true;
  try {
    const base = typeof window !== 'undefined' && window.location ? window.location.href : 'http://localhost';
    const currUrl = new URL(currentSrc, base);
    const targetUrlObj = new URL(targetUrl, base);
    return currUrl.pathname === targetUrlObj.pathname && currUrl.search === targetUrlObj.search;
  } catch {
    return currentSrc.endsWith(targetUrl) || targetUrl.endsWith(currentSrc);
  }
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
    const wasLoadingInFlight = this.activeCleanup !== null;

    if (this.activeCleanup) {
      this.activeCleanup();
      this.activeCleanup = null;
    }

    callbacks.onStart?.();

    if (!videoUrl) {
      callbacks.onFallback?.(fallbackImageUrl);
      return loadId;
    }

    const isSameSource = isSameVideoUrl(videoElement.src, videoUrl) && !videoElement.error;
    const isReady = typeof videoElement.readyState === 'number' ? videoElement.readyState >= 2 : false;

    videoElement.playbackRate = playbackSpeed;
    videoElement.loop = isLooping;

    if (isSameSource && isReady) {
      callbacks.onSuccess?.();
      if (autoPlay) {
        videoElement.play().catch(() => {});
      }
      return loadId;
    }

    const shouldStartFreshLoad = !isSameSource || !wasLoadingInFlight;
    if (shouldStartFreshLoad) {
      videoElement.src = videoUrl;
    }

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
    if (shouldStartFreshLoad) {
      videoElement.load();
    }

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
