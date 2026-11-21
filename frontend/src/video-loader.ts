export interface VideoElementTarget {
  src: string;
  playbackRate: number;
  loop: boolean;
  muted?: boolean;
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
    autoPlay: boolean = false,
    isMuted: boolean = false
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
    videoElement.muted = isMuted;

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

export interface FullscreenCapableContainer {
  classList?: {
    contains(cls: string): boolean;
    add(cls: string): void;
    remove(cls: string): void;
  };
  requestFullscreen?: () => Promise<void>;
  webkitRequestFullscreen?: () => Promise<void> | void;
  [key: string]: any;
}

export interface FullscreenCapableVideo {
  webkitEnterFullscreen?: () => void;
  webkitExitFullscreen?: () => void;
  webkitDisplayingFullscreen?: boolean;
  [key: string]: any;
}

export function isVideoFullscreen(
  container?: FullscreenCapableContainer | null,
  video?: FullscreenCapableVideo | null,
  doc?: any
): boolean {
  if (container?.classList?.contains('is-css-fullscreen')) return true;
  if (video?.webkitDisplayingFullscreen) return true;
  const d = doc !== undefined ? doc : (typeof document !== 'undefined' ? document : null);
  if (d && container && (d.fullscreenElement === container || d.webkitFullscreenElement === container)) {
    return true;
  }
  return false;
}

export async function toggleVideoFullscreen(
  container: FullscreenCapableContainer | null,
  video: FullscreenCapableVideo | null,
  doc?: any
): Promise<boolean> {
  const d = doc !== undefined ? doc : (typeof document !== 'undefined' ? document : null);
  if (isVideoFullscreen(container, video, d)) {
    await exitVideoFullscreenIfActive(container, video, d);
    return false;
  }

  if (container?.requestFullscreen) {
    try {
      await container.requestFullscreen();
      return true;
    } catch {
      // Fall through to video native or CSS
    }
  } else if (container?.webkitRequestFullscreen) {
    try {
      await container.webkitRequestFullscreen();
      return true;
    } catch {
      // Fall through
    }
  }

  if (typeof video?.webkitEnterFullscreen === 'function') {
    try {
      video.webkitEnterFullscreen();
      return true;
    } catch {
      // Fall through
    }
  }

  if (container?.classList) {
    container.classList.add('is-css-fullscreen');
    return true;
  }

  return false;
}

export async function exitVideoFullscreenIfActive(
  container: FullscreenCapableContainer | null,
  video: FullscreenCapableVideo | null,
  doc?: any
): Promise<void> {
  if (container?.classList?.contains('is-css-fullscreen')) {
    container.classList.remove('is-css-fullscreen');
  }

  if (video?.webkitDisplayingFullscreen && typeof video.webkitExitFullscreen === 'function') {
    try {
      video.webkitExitFullscreen();
    } catch {
      // ignore
    }
  }

  const d = doc !== undefined ? doc : (typeof document !== 'undefined' ? document : null);
  if (d && (d.fullscreenElement || d.webkitFullscreenElement)) {
    try {
      if (d.exitFullscreen) {
        await d.exitFullscreen();
      } else if (d.webkitExitFullscreen) {
        await d.webkitExitFullscreen();
      }
    } catch {
      // ignore
    }
  }
}
