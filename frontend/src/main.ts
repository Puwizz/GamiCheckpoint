import { StepSessionEngine, UIStatusState } from './session-engine';
import { DetectionOverlayRenderer, formatRoiMethod } from './hud-overlay';
import { CameraStreamAdapter } from './camera';
import { TutorialCoordinator } from './tutorial-coordinator';
import {
  TOTAL_STEPS,
  STEPS,
  STEP_FOLD_GUIDES,
  StepDefinition,
  StepFoldGuide,
  VerificationResult,
  PlaybackSpeed,
  TutorMode,
  OrigamiFigureSummary,
  OrigamiFigureDetail,
} from './types';

const API_PREDICT_BASE = '/api/predict';
const PREDICT_INTERVAL_MS = 1000;

document.addEventListener('DOMContentLoaded', () => {
  // --- View Containers & Navigation ---
  const viewGallery = document.getElementById('view-gallery');
  const viewWorkspace = document.getElementById('view-workspace');
  const figuresGrid = document.getElementById('figures-grid');
  const btnBackToGallery = document.getElementById('btn-back-to-gallery') as HTMLButtonElement | null;
  const brandHomeLink = document.getElementById('brand-home-link');
  const brandTitleLink = document.getElementById('brand-title-link');
  const workspaceFigureTag = document.getElementById('workspace-figure-tag');
  const brandHeaderTag = document.getElementById('brand-header-tag');
  const btnCompletionBackGallery = document.getElementById('btn-completion-back-gallery') as HTMLButtonElement | null;
  const filterTabs = document.querySelectorAll<HTMLButtonElement>('.gallery-filters .filter-tab');
  const countAll = document.getElementById('count-all');
  const countAvailable = document.getElementById('count-available');
  const countComingSoon = document.getElementById('count-coming-soon');
  let loadedFigures: OrigamiFigureSummary[] = [];
  let currentGalleryFilter: string = 'all';

  // --- Toast Notification ---
  const appToast = document.getElementById('app-toast');
  const toastIcon = document.getElementById('toast-icon');
  const toastMessage = document.getElementById('toast-message');
  let toastTimeout: any = null;

  // --- DOM Elements: Video Tutorial (Watch View) ---
  const watchStage = document.getElementById('watch-stage');
  const verifyStage = document.getElementById('verify-stage');
  const tutorialVideo = document.getElementById('tutorial-video') as HTMLVideoElement | null;
  const videoFallbackContainer = document.getElementById('video-fallback-container');
  const fallbackGuideImage = document.getElementById('fallback-guide-image') as HTMLImageElement | null;
  const videoLoadingOverlay = document.getElementById('video-loading-overlay');

  const modePillWatch = document.getElementById('mode-pill-watch');
  const modePillVerify = document.getElementById('mode-pill-verify');

  const btnVideoPlay = document.getElementById('btn-video-play') as HTMLButtonElement | null;
  const btnSpeed05 = document.getElementById('btn-speed-05') as HTMLButtonElement | null;
  const btnSpeed075 = document.getElementById('btn-speed-075') as HTMLButtonElement | null;
  const btnSpeed10 = document.getElementById('btn-speed-10') as HTMLButtonElement | null;
  const btnVideoLoop = document.getElementById('btn-video-loop') as HTMLButtonElement | null;
  const btnVideoMute = document.getElementById('btn-video-mute') as HTMLButtonElement | null;
  const btnStartVerify = document.getElementById('btn-start-verify') as HTMLButtonElement | null;

  // --- DOM Elements: Camera & AI Verification (Verify View) ---
  const webcam = document.getElementById('webcam') as HTMLVideoElement | null;
  const detectionCanvas = document.getElementById('detection-overlay') as HTMLCanvasElement | null;
  const toggleDetectionBox = document.getElementById('toggle-detection-box') as HTMLInputElement | null;
  const roiMethodBadge = document.getElementById('roi-method-badge');

  const btnBackToWatch = document.getElementById('btn-back-to-watch') as HTMLButtonElement | null;
  const btnStart = document.getElementById('btn-start') as HTMLButtonElement | null;
  const btnStop = document.getElementById('btn-stop') as HTMLButtonElement | null;
  const btnRefreshCameras = document.getElementById('btn-refresh-cameras') as HTMLButtonElement | null;
  const cameraSelect = document.getElementById('camera-select') as HTMLSelectElement | null;
  const cameraHint = document.getElementById('camera-hint');

  // --- DOM Elements: Escape Hatch & Modals ---
  const escapeHatchBanner = document.getElementById('escape-hatch-banner');
  const btnSkipStep = document.getElementById('btn-skip-step') as HTMLButtonElement | null;

  const celebrationModal = document.getElementById('celebration-modal');
  const celebrationTitle = document.getElementById('celebration-title');
  const celebrationSubtitle = document.getElementById('celebration-subtitle');

  const completionModal = document.getElementById('completion-modal');
  const completionTitle = document.getElementById('completion-title');
  const completionSubtitle = document.getElementById('completion-subtitle');
  const btnRestartCourse = document.getElementById('btn-restart-course') as HTMLButtonElement | null;

  // --- DOM Elements: Right Panel (Info & Status) ---
  const btnReset = document.getElementById('btn-reset') as HTMLButtonElement | null;
  const resultDisplay = document.getElementById('result-display');
  const lblCurrentStep = document.getElementById('lbl-current-step');
  const lblStep = document.getElementById('lbl-step');
  const lblConfidence = document.getElementById('lbl-confidence');
  const senpaiTip = document.getElementById('senpai-tip');
  const currentStepTitle = document.getElementById('current-step-title');
  const stepProgress = document.getElementById('step-progress');
  const referenceStepTitle = document.getElementById('reference-step-title');
  const referenceStepDescription = document.getElementById('reference-step-description');
  const stepReferenceImage = document.getElementById('step-reference-image') as HTMLImageElement | null;

  if (!webcam || !tutorialVideo) return;

  // --- State Variables ---
  let currentFigureId: string = 'crane';
  let currentFigureDetail: OrigamiFigureDetail | null = null;
  let activeStepDefs: StepDefinition[] = STEPS;
  let activeGuides: StepFoldGuide[] = STEP_FOLD_GUIDES;

  // --- Domain Engines ---
  const sessionEngine = new StepSessionEngine(1, TOTAL_STEPS, activeStepDefs);
  const hudOverlay = new DetectionOverlayRenderer(detectionCanvas, webcam);
  const cameraAdapter = new CameraStreamAdapter(webcam);

  let intervalId: number | null = null;
  let isProcessing = false;
  let flowVersion = 0;
  let activePredictAbortController: AbortController | null = null;

  // --- Initialize Tutorial Coordinator ---
  const tutor = new TutorialCoordinator({
    storageKey: 'gamisenpai_progress_crane',
    totalSteps: TOTAL_STEPS,
    guides: activeGuides,
    onModeChange: (mode: TutorMode) => {
      applyModeUI(mode);
    },
    onStepChange: (step: number) => {
      sessionEngine.setStep(step, tutor.getCompletedSteps());
      updateCurrentStepUI();
      loadVideoForStep(step);
    },
    onSpeedChange: (speed: PlaybackSpeed) => {
      if (tutorialVideo) tutorialVideo.playbackRate = speed;
      updateSpeedPillsUI(speed);
    },
    onLoopChange: (loop: boolean) => {
      if (tutorialVideo) tutorialVideo.loop = loop;
      if (btnVideoLoop) {
        btnVideoLoop.textContent = loop ? 'วนลูป: เปิด' : 'วนลูป: ปิด';
        btnVideoLoop.classList.toggle('active', loop);
      }
    },
    onEscapeHatch: (active: boolean) => {
      if (escapeHatchBanner) {
        escapeHatchBanner.classList.toggle('hidden', !active);
      }
    },
    onCelebrationStart: (step: number) => {
      stopCamera();
      const total = tutor.getTotalSteps();
      if (celebrationTitle) {
        celebrationTitle.textContent = `ยอดเยี่ยมมาก! Step ${step} ผ่านเรียบร้อย`;
      }
      if (celebrationSubtitle) {
        celebrationSubtitle.textContent =
          step < total
            ? `เตรียมพบกับวิดีโอสอนขั้นตอนที่ ${step + 1} ใน 2 วินาที...`
            : 'ยินดีด้วยครับ! ผลงานโอริกามิเสร็จสมบูรณ์แล้ว 🎉';
      }
      if (celebrationModal) celebrationModal.classList.remove('hidden');
    },
    onCelebrationEnd: (nextStep: number | null) => {
      if (celebrationModal) celebrationModal.classList.add('hidden');
      if (nextStep === null) {
        // Course completed
        if (completionTitle && currentFigureDetail) {
          completionTitle.textContent = `ยินดีด้วยครับ! ${currentFigureDetail.title_th} เสร็จสมบูรณ์แล้ว 🪶`;
        }
        if (completionSubtitle && currentFigureDetail) {
          completionSubtitle.textContent = `คุณผ่านการพับ ${currentFigureDetail.title_th} ครบทั้ง ${currentFigureDetail.total_steps} ขั้นตอนด้วยความช่วยเหลือของ GamiSenpai`;
        }
        if (completionModal) completionModal.classList.remove('hidden');
      }
    },
  });

  // Sync initial state from coordinator to engine
  sessionEngine.setStep(tutor.getCurrentStep(), tutor.getCompletedSteps());

  // --- Toast Notification Helper ---
  function getToastIconSvg(icon: string): string {
    if (icon.startsWith('<svg')) return icon;
    switch (icon) {
      case 'check':
      case 'success':
        return '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="var(--color-bamboo)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
      case 'warning':
      case 'alert':
        return '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="var(--color-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
      case 'info':
      case 'time':
        return '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="var(--color-indigo)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';
      case 'sparkle':
      default:
        return '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="var(--color-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>';
    }
  }

  function showToast(icon: string, message: string, durationMs: number = 3500): void {
    if (!appToast || !toastIcon || !toastMessage) return;
    toastIcon.innerHTML = getToastIconSvg(icon);
    toastMessage.textContent = message;
    appToast.classList.remove('hidden');

    if (toastTimeout) clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
      appToast.classList.add('hidden');
    }, durationMs);
  }

  // --- UI Helper Functions ---
  function setStatusBox(status: UIStatusState): void {
    if (!resultDisplay) return;
    const statusIcon = resultDisplay.querySelector('.status-icon');
    const statusTitle = resultDisplay.querySelector('.status-text h3') as HTMLElement | null;
    const statusDesc = resultDisplay.querySelector('.status-text p');

    resultDisplay.className = `status-box ${status.kind}`;
    if (statusIcon) statusIcon.textContent = status.icon;
    if (statusTitle) {
      statusTitle.textContent = status.title;
      statusTitle.style.color =
        status.kind === 'correct' ? 'var(--success)' : status.kind === 'incorrect' ? 'var(--danger)' : '';
    }
    if (statusDesc) statusDesc.textContent = status.description;
  }

  function setCameraHint(message: string, variant: 'info' | 'success' | 'warning' = 'info'): void {
    if (!cameraHint) return;
    cameraHint.textContent = message;
    cameraHint.className = `camera-hint ${variant}`;
  }

  function renderStepProgress(): void {
    if (!stepProgress) return;
    stepProgress.innerHTML = '';
    const current = tutor.getCurrentStep();
    const guides = tutor.getGuides();

    guides.forEach((g, index) => {
      const stepNumber = index + 1;
      const item = document.createElement('div');
      item.className = 'step-pill';
      let stateLabel = 'Locked';

      if (tutor.isStepCompleted(stepNumber)) {
        item.classList.add('done');
        stateLabel = 'Matched';
      } else if (stepNumber === current) {
        item.classList.add('active');
        stateLabel = tutor.getMode() === 'watch' ? 'Learning' : 'Checking';
      } else {
        item.classList.add('locked');
      }

      item.setAttribute('aria-label', `Step ${stepNumber}: ${g.name} — ${stateLabel}`);
      item.innerHTML = `
        <span class="step-index">${stepNumber}</span>
        <span class="step-content">
            <span class="step-name">${g.name}</span>
            <span class="step-state">${stateLabel}</span>
        </span>
      `;
      stepProgress.appendChild(item);
    });
  }

  function updateCurrentStepUI(): void {
    const currentStep = tutor.getCurrentStep();
    const guide = tutor.getCurrentGuide();
    const total = tutor.getTotalSteps();
    const currentDef = activeStepDefs[currentStep - 1];

    if (lblCurrentStep) {
      lblCurrentStep.textContent = currentStep <= total ? `Step ${currentStep}` : 'Complete';
    }
    if (currentStepTitle) {
      currentStepTitle.textContent = guide
        ? `Step ${currentStep}: ${guide.thaiName}`
        : 'All checkpoints complete';
    }
    if (senpaiTip) {
      senpaiTip.textContent = guide
        ? `${guide.foldTip} — จุดสังเกต: ${guide.checkPointTip}`
        : 'All checkpoints matched. Nice work.';
    }

    if (currentDef) {
      if (referenceStepTitle) referenceStepTitle.textContent = `Step ${currentStep}: ${currentDef.name}`;
      if (referenceStepDescription) referenceStepDescription.textContent = currentDef.reference;
      if (stepReferenceImage) {
        stepReferenceImage.src = guide?.fallbackImageUrl || `/references/step-${currentStep}.png`;
        stepReferenceImage.alt = `Reference shape for Step ${currentStep}: ${currentDef.name}`;
      }
    }

    renderStepProgress();
  }

  function updateSpeedPillsUI(speed: PlaybackSpeed): void {
    [btnSpeed05, btnSpeed075, btnSpeed10].forEach((btn) => {
      if (!btn) return;
      btn.classList.remove('active');
    });
    if (speed === 0.5 && btnSpeed05) btnSpeed05.classList.add('active');
    if (speed === 0.75 && btnSpeed075) btnSpeed075.classList.add('active');
    if (speed === 1.0 && btnSpeed10) btnSpeed10.classList.add('active');
  }

  // --- Video Loading & Fallback ---
  function loadVideoForStep(step: number): void {
    if (!tutorialVideo) return;
    const guide = tutor.getCurrentGuide();

    // Hide fallback container while attempting video load
    if (videoFallbackContainer) videoFallbackContainer.classList.add('hidden');
    tutorialVideo.classList.remove('hidden');
    if (videoLoadingOverlay) videoLoadingOverlay.classList.remove('hidden');

    if (!guide.videoUrl) {
      // Direct fallback to image guide mode
      triggerFallback(guide.fallbackImageUrl);
      return;
    }

    tutorialVideo.src = guide.videoUrl;
    tutorialVideo.playbackRate = tutor.getPlaybackSpeed();
    tutorialVideo.loop = tutor.getIsLooping();

    // Reset Play button text
    if (btnVideoPlay) btnVideoPlay.textContent = 'หยุด';

    const onCanPlay = () => {
      tutorialVideo.removeEventListener('canplay', onCanPlay);
      tutorialVideo.removeEventListener('error', onError);
      if (videoLoadingOverlay) videoLoadingOverlay.classList.add('hidden');
      tutorialVideo.play().catch(() => {
        // Autoplay may require user interaction
      });
    };

    const onError = () => {
      tutorialVideo.removeEventListener('canplay', onCanPlay);
      tutorialVideo.removeEventListener('error', onError);
      console.warn(`[Tutor] Video load failed for Step ${step} (${guide.videoUrl}), triggering fallback.`);
      triggerFallback(guide.fallbackImageUrl);
    };

    tutorialVideo.addEventListener('canplay', onCanPlay, { once: true });
    tutorialVideo.addEventListener('error', onError, { once: true });
    tutorialVideo.load();
  }

  function triggerFallback(fallbackImageUrl: string): void {
    if (videoLoadingOverlay) videoLoadingOverlay.classList.add('hidden');
    if (tutorialVideo) {
      tutorialVideo.pause();
      tutorialVideo.classList.add('hidden');
    }
    if (videoFallbackContainer && fallbackGuideImage) {
      fallbackGuideImage.src = fallbackImageUrl || '/references/step-1.png';
      videoFallbackContainer.classList.remove('hidden');
    }
  }

  // --- Mode Switching (Watch vs Verify) ---
  function applyModeUI(mode: TutorMode): void {
    if (mode === 'watch') {
      watchStage?.classList.remove('hidden');
      verifyStage?.classList.add('hidden');
      modePillWatch?.classList.add('active');
      modePillVerify?.classList.remove('active');

      stopCamera();

      if (tutorialVideo && !tutorialVideo.classList.contains('hidden')) {
        tutorialVideo.play().catch(() => {});
        if (btnVideoPlay) btnVideoPlay.textContent = 'หยุด';
      }

      setStatusBox({
        kind: 'waiting',
        icon: '...',
        title: 'โหมดดูคลิปสอน (Watch & Fold)',
        description: 'ดูวิดีโอสอนพับกระดาษตามขั้นตอน เมื่อพับเสร็จแล้วให้กดปุ่มเปิดกล้องเพื่อตรวจ',
      });
    } else {
      watchStage?.classList.add('hidden');
      verifyStage?.classList.remove('hidden');
      modePillWatch?.classList.remove('active');
      modePillVerify?.classList.add('active');

      if (tutorialVideo) {
        tutorialVideo.pause();
        if (btnVideoPlay) btnVideoPlay.textContent = 'เล่น';
      }

      startCamera();

      setStatusBox({
        kind: 'waiting',
        icon: '...',
        title: 'ส่องไปที่กระดาษโอริกามิ',
        description: 'แสดงชิ้นงานที่พับเสร็จต่อหน้ากล้อง และถือค้างไว้ให้นิ่ง',
      });
    }

    renderStepProgress();
  }

  // --- Camera Operations ---
  async function refreshCameraList(): Promise<void> {
    if (!cameraSelect || !btnRefreshCameras) return;
    btnRefreshCameras.disabled = true;
    const devices = await cameraAdapter.getAvailableVideoDevices();

    cameraSelect.innerHTML = '';
    if (devices.length === 0) {
      const opt = document.createElement('option');
      opt.text = 'No cameras found';
      cameraSelect.appendChild(opt);
    } else {
      devices.forEach((dev: MediaDeviceInfo, idx: number) => {
        const opt = document.createElement('option');
        opt.value = dev.deviceId;
        opt.text = dev.label || `Camera ${idx + 1}`;
        cameraSelect.appendChild(opt);
      });
      cameraAdapter.setSelectedDeviceId(cameraSelect.value);
    }
    btnRefreshCameras.disabled = false;
  }

  async function startCamera(): Promise<void> {
    try {
      if (cameraSelect) cameraAdapter.setSelectedDeviceId(cameraSelect.value);
      await cameraAdapter.start();

      if (btnStart) btnStart.disabled = true;
      if (btnStop) btnStop.disabled = false;
      if (cameraSelect) cameraSelect.disabled = true;
      if (btnRefreshCameras) btnRefreshCameras.disabled = true;

      const selectedLabel = cameraSelect?.options[cameraSelect.selectedIndex]?.text || 'selected camera';
      setCameraHint(`Streaming from ${selectedLabel}. If this is not Iriun, stop camera and choose another.`, 'success');

      if (intervalId !== null) clearInterval(intervalId);
      intervalId = window.setInterval(captureAndPredict, PREDICT_INTERVAL_MS);
    } catch (err) {
      console.error('Camera start error:', err);
      showToast('warning', 'ไม่สามารถเข้าถึงกล้องได้ กรุณาตรวจสอบการอนุญาตสิทธิ์ในเบราว์เซอร์');
      setCameraHint('ไม่สามารถเข้าถึงกล้องได้ กรุณาตรวจสอบสิทธิ์การใช้งานกล้อง', 'warning');
    }
  }

  function stopCamera(): void {
    if (activePredictAbortController) {
      activePredictAbortController.abort();
      activePredictAbortController = null;
    }
    cameraAdapter.stop();
    if (intervalId !== null) {
      clearInterval(intervalId);
      intervalId = null;
    }

    hudOverlay.reset();
    if (roiMethodBadge) roiMethodBadge.classList.add('hidden');

    if (btnStart) btnStart.disabled = false;
    if (btnStop) btnStop.disabled = true;
    if (cameraSelect) cameraSelect.disabled = false;
    if (btnRefreshCameras) btnRefreshCameras.disabled = false;

    flowVersion += 1;
  }

  // --- AI Frame Polling & Decision Loop ---
  async function captureAndPredict(): Promise<void> {
    if (isProcessing || !cameraAdapter.isStreaming()) return;
    const currentState = sessionEngine.getState();
    const total = tutor.getTotalSteps();
    if (currentState.currentStep > total || currentState.isAdvancing) return;

    isProcessing = true;
    const reqStep = currentState.currentStep;
    const reqFlowVersion = flowVersion;

    try {
      const { blob, quality } = await cameraAdapter.captureFrameBlob();

      if (!quality.ready || !blob) {
        hudOverlay.clear();
        if (roiMethodBadge) roiMethodBadge.classList.add('hidden');
        if (lblStep) lblStep.textContent = '-';
        if (lblConfidence) lblConfidence.textContent = '0%';
        setStatusBox({
          kind: 'waiting',
          icon: '...',
          title: 'ส่องไปที่กระดาษโอริกามิ',
          description: 'ส่องไปที่กระดาษโอริกามิเพื่อตรวจเช็คขั้นตอนปัจจุบัน',
        });
        return;
      }

      if (activePredictAbortController) {
        activePredictAbortController.abort();
      }
      activePredictAbortController = new AbortController();

      const formData = new FormData();
      formData.append('file', blob, 'frame.jpg');

      const response = await fetch(`${API_PREDICT_BASE}?figure=${encodeURIComponent(currentFigureId)}`, {
        method: 'POST',
        body: formData,
        signal: activePredictAbortController.signal,
      });

      if (!response.ok) {
        let errorMsg = `Server error (${response.status})`;
        try {
          const errData = await response.json();
          if (errData.error) {
            errorMsg = errData.error;
          } else if (errData.detail) {
            errorMsg = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
          }
        } catch {
          // ignore non-json response
        }
        console.warn('Prediction API error:', response.status, errorMsg);
        showToast('warning', errorMsg);
        setStatusBox({
          kind: 'incorrect',
          icon: '!',
          title: 'ไม่สามารถตรวจจับได้',
          description: errorMsg,
        });
        hudOverlay.clear();
        return;
      }

      const data: VerificationResult = await response.json();

      if (data.error) {
        console.error('API Error:', data.error);
        showToast('warning', data.error);
        hudOverlay.clear();
        return;
      }

      if (reqStep !== sessionEngine.getState().currentStep || reqFlowVersion !== flowVersion) {
        return; // Stale frame
      }

      // Update badge
      if (data.roi && data.roi.applied && data.roi.method && roiMethodBadge) {
        roiMethodBadge.textContent = `ROI: ${formatRoiMethod(data.roi.method)}`;
        roiMethodBadge.classList.remove('hidden');
      } else if (roiMethodBadge) {
        roiMethodBadge.classList.add('hidden');
      }

      // Update UI labels
      if (lblStep) lblStep.textContent = data.step_name || data.step || '-';
      if (lblConfidence) {
        lblConfidence.textContent = data.confidence ? `${Math.round(data.confidence)}%` : '0%';
      }

      // Evaluate via Session Engine
      const action = sessionEngine.evaluatePrediction(data);

      switch (action.type) {
        case 'FRAME_NOT_READY':
          setStatusBox(action.status);
          hudOverlay.clear();
          break;

        case 'UNRECOGNIZED':
          setStatusBox(action.status);
          hudOverlay.clear();
          break;

        case 'LOW_CONFIDENCE':
          setStatusBox(action.status);
          hudOverlay.updateFromAPI(data, reqStep, 'tracking');
          break;

        case 'LOCKED_MISMATCH':
          setStatusBox(action.status);
          hudOverlay.updateFromAPI(data, reqStep, action.overlayState);
          break;

        case 'CONFIRMING':
          setStatusBox(action.status);
          hudOverlay.updateFromAPI(data, reqStep, 'tracking', `${action.streak}/3`);
          break;

        case 'MATCH_SUCCESS':
          setStatusBox(action.status);
          hudOverlay.updateFromAPI(data, reqStep, 'matched', 'PASS');
          tutor.handleStepVerified(action.finishedStep);
          break;

        case 'ALL_COMPLETE':
          setStatusBox(action.status);
          hudOverlay.updateFromAPI(data, reqStep, 'matched', 'COMPLETE');
          tutor.handleStepVerified(total);
          break;
      }
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        return;
      }
      console.error('Fetch error:', err);
      showToast('warning', 'การเชื่อมต่อเซิร์ฟเวอร์ขัดข้อง กรุณาตรวจสอบสถานะเซิร์ฟเวอร์');
      setStatusBox({
        kind: 'incorrect',
        icon: '!',
        title: 'การเชื่อมต่อขัดข้อง',
        description: 'ไม่สามารถติดต่อ AI Server ได้ กรุณาตรวจสอบสถานะเซิร์ฟเวอร์',
      });
      hudOverlay.clear();
    } finally {
      isProcessing = false;
    }
  }

  // --- Figure Selection & Hub Routing ---
  function updateFilterCounts(figures: OrigamiFigureSummary[]): void {
    const total = figures.length;
    const available = figures.filter((f) => f.status === 'available').length;
    const comingSoon = total - available;
    if (countAll) countAll.textContent = String(total);
    if (countAvailable) countAvailable.textContent = String(available);
    if (countComingSoon) countComingSoon.textContent = String(comingSoon);
  }

  // Setup gallery filter tabs listener
  filterTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const filter = tab.getAttribute('data-filter') || 'all';
      currentGalleryFilter = filter;
      filterTabs.forEach((t) => {
        const isActive = t === tab;
        t.classList.toggle('active', isActive);
        t.setAttribute('aria-selected', isActive ? 'true' : 'false');
      });
      if (loadedFigures.length > 0) {
        renderFiguresGrid(loadedFigures, currentGalleryFilter);
      }
    });
  });

  async function fetchFigures(): Promise<void> {
    if (!figuresGrid) return;
    try {
      const res = await fetch('/api/figures');
      const figures: OrigamiFigureSummary[] = await res.json();
      loadedFigures = figures;
      updateFilterCounts(figures);
      renderFiguresGrid(figures, currentGalleryFilter);
    } catch (e) {
      console.error('Failed to load figures:', e);
      figuresGrid.innerHTML = `
        <div class="gallery-error">
          <div class="gallery-error-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="var(--color-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <p>ไม่สามารถดึงข้อมูลคลังโอริกามิได้ กรุณาลองใหม่อีกครั้ง</p>
          <button id="btn-retry-figures" class="btn primary" type="button">ลองใหม่</button>
        </div>
      `;
      document.getElementById('btn-retry-figures')?.addEventListener('click', fetchFigures);
    }
  }

  function getStoredProgress(figureId: string, totalSteps: number): { completedCount: number; currentStep: number; percent: number } {
    try {
      const raw = localStorage.getItem(`gamisenpai_progress_${figureId}`);
      if (!raw) return { completedCount: 0, currentStep: 1, percent: 0 };
      const parsed = JSON.parse(raw);
      const completed = Array.isArray(parsed.completedSteps) ? parsed.completedSteps.length : 0;
      const current = typeof parsed.currentStep === 'number' ? parsed.currentStep : 1;
      const percent = Math.min(100, Math.round((completed / totalSteps) * 100));
      return { completedCount: completed, currentStep: current, percent };
    } catch {
      return { completedCount: 0, currentStep: 1, percent: 0 };
    }
  }

  function renderFiguresGrid(figures: OrigamiFigureSummary[], filter: string = 'all'): void {
    if (!figuresGrid) return;
    figuresGrid.innerHTML = '';

    const filtered = figures.filter((fig) => {
      if (filter === 'available') return fig.status === 'available';
      if (filter === 'coming_soon') return fig.status !== 'available';
      return true;
    });

    figuresGrid.classList.toggle('is-single-card', filtered.length === 1);

    if (filtered.length === 0) {
      figuresGrid.innerHTML = `
        <div class="gallery-empty">
          <p>ยังไม่มีโมเดลโอริกามิในหมวดหมู่นี้</p>
        </div>
      `;
      return;
    }

    filtered.forEach((fig) => {
      const card = document.createElement('div');
      const isAvailable = fig.status === 'available';
      card.className = `figure-card glass-panel ${isAvailable ? 'is-available' : 'is-coming-soon'}`;
      card.setAttribute('data-figure-id', fig.id);

      const progress = getStoredProgress(fig.id, fig.total_steps);
      const diffLevel = fig.difficulty_level || (fig.difficulty === 'ระดับง่าย' ? 1 : fig.difficulty === 'ระดับกลาง' ? 2 : 3);
      const diffClass = `diff-lvl-${diffLevel}`;
      const diffDots = diffLevel === 1 ? '●○○' : diffLevel === 2 ? '●●○' : '●●●';

      const clockSvg = `<svg class="meta-icon" viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`;
      const stepsSvg = `<svg class="meta-icon" viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>`;
      const checkSvg = `<svg class="meta-icon-success" viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>`;
      const arrowSvg = `<svg class="btn-arrow" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>`;

      let progressLabel = 'ยังไม่ได้เริ่ม';
      let progressLabelClass = 'status-new';
      if (progress.percent === 100) {
        progressLabel = `${checkSvg} สำเร็จแล้ว 100%`;
        progressLabelClass = 'status-complete';
      } else if (progress.completedCount > 0) {
        progressLabel = `ผ่านแล้ว ${progress.completedCount}/${fig.total_steps} ขั้น (${progress.percent}%)`;
        progressLabelClass = 'status-progress';
      }

      const buttonLabel = isAvailable
        ? (progress.completedCount > 0 && progress.percent < 100 ? 'ฝึกพับต่อจากเดิม' : 'เริ่มฝึกพับ')
        : 'เปิดให้ฝึกพับเร็วๆ นี้';

      card.innerHTML = `
        <div class="figure-card-media">
          <img class="figure-card-img" src="${fig.thumbnail}" alt="${fig.title_th}" loading="lazy">
          <div class="figure-card-badges">
            <span class="status-badge ${isAvailable ? 'badge-available' : 'badge-coming-soon'}">
              <span class="status-badge-dot"></span>
              <span>${isAvailable ? 'พร้อมฝึกพับ' : 'เร็วๆ นี้'}</span>
            </span>
            <span class="difficulty-badge ${diffClass}" title="ระดับความยาก: ${fig.difficulty}">
              <span>${fig.difficulty} ${diffDots}</span>
            </span>
          </div>
          <div class="figure-card-time">
            ${clockSvg}
            <span>~${fig.estimated_minutes} นาที</span>
          </div>
        </div>

        <div class="figure-card-body">
          <div class="figure-card-header">
            <h3 class="figure-card-title">${fig.title_th}</h3>
            <span class="figure-card-subtitle">${fig.title_en}</span>
          </div>
          <p class="figure-card-desc">${fig.description_th}</p>

          <div class="figure-card-meta">
            <span class="meta-steps">
              ${stepsSvg}
              <span>${fig.total_steps} ขั้นตอน</span>
            </span>
            <span class="meta-progress-label ${progressLabelClass}">
              ${progressLabel}
            </span>
          </div>

          <div class="figure-card-progress" role="progressbar" aria-valuenow="${progress.percent}" aria-valuemin="0" aria-valuemax="100">
            <div class="figure-progress-fill" style="width: ${progress.percent}%"></div>
          </div>

          <div class="figure-card-footer">
            <button class="btn ${isAvailable ? 'primary btn-select-figure' : 'secondary btn-disabled'}" type="button" ${isAvailable ? '' : 'disabled'}>
              <span>${buttonLabel}</span>
              ${isAvailable ? arrowSvg : ''}
            </button>
          </div>
        </div>
      `;

      card.addEventListener('click', (e) => {
        e.preventDefault();
        if (isAvailable) {
          selectFigure(fig.id);
        } else {
          showToast('info', `ชิ้นงาน "${fig.title_th}" จะเปิดให้ฝึกพับเร็วๆ นี้ครับ`);
        }
      });

      figuresGrid.appendChild(card);
    });
  }

  async function selectFigure(figureId: string): Promise<void> {
    try {
      const res = await fetch(`/api/figures/${figureId}`);
      if (!res.ok) throw new Error('Figure not found');
      const detail: OrigamiFigureDetail = await res.json();
      enterFigureWorkspace(detail);
    } catch (e) {
      console.error('Failed to load figure detail:', e);
      showToast('warning', 'เกิดข้อผิดพลาดในการโหลดข้อมูลตัวพับ กรุณาลองใหม่อีกครั้ง');
    }
  }

  function enterFigureWorkspace(fig: OrigamiFigureDetail): void {
    currentFigureId = fig.id;
    currentFigureDetail = fig;

    // Map step definitions
    activeStepDefs = fig.steps.map((s) => ({
      name: s.name_en,
      tip: `${s.fold_tip} — จุดสังเกต: ${s.checkpoint_tip}`,
      reference: s.reference_description,
    }));

    activeGuides = fig.steps.map((s) => ({
      stepNumber: s.step_number,
      name: s.name_en,
      thaiName: s.name_th,
      foldTip: s.fold_tip,
      checkPointTip: s.checkpoint_tip,
      videoUrl: s.video_url,
      fallbackImageUrl: s.fallback_image_url,
    }));

    // Configure session engine & coordinator
    sessionEngine.reconfigure(fig.total_steps, activeStepDefs);
    tutor.reconfigure(fig.total_steps, activeGuides, `gamisenpai_progress_${fig.id}`);
    sessionEngine.setStep(tutor.getCurrentStep(), tutor.getCompletedSteps());

    // Update Header Navigation UI
    if (btnBackToGallery) btnBackToGallery.classList.remove('hidden');
    if (workspaceFigureTag) workspaceFigureTag.textContent = `กำลังฝึกพับ: ${fig.title_th}`;
    if (brandHeaderTag) brandHeaderTag.textContent = fig.subtitle || `${fig.title_en} · ORIGAMI STUDIO`;

    // Switch View
    viewGallery?.classList.add('hidden');
    viewWorkspace?.classList.remove('hidden');

    // Reset modals
    completionModal?.classList.add('hidden');
    celebrationModal?.classList.add('hidden');

    // Render Workspace UI for current restored step
    updateCurrentStepUI();
    loadVideoForStep(tutor.getCurrentStep());
    applyModeUI(tutor.getMode());
    window.scrollTo({ top: 0, behavior: 'smooth' });

    showToast('sparkle', `เข้าสู่สตูดิโอสอนพับ: ${fig.title_th}`);
  }

  function returnToGallery(): void {
    stopCamera();
    if (tutorialVideo) {
      tutorialVideo.pause();
      tutorialVideo.removeAttribute('src');
      tutorialVideo.load();
    }
    tutor.destroy();

    viewWorkspace?.classList.add('hidden');
    viewGallery?.classList.remove('hidden');

    if (btnBackToGallery) btnBackToGallery.classList.add('hidden');
    if (brandHeaderTag) brandHeaderTag.textContent = 'ORIGAMI STUDIO';

    fetchFigures();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // --- Event Bindings ---
  btnBackToGallery?.addEventListener('click', returnToGallery);
  brandHomeLink?.addEventListener('click', (e) => {
    e.preventDefault();
    returnToGallery();
  });
  brandTitleLink?.addEventListener('click', (e) => {
    e.preventDefault();
    returnToGallery();
  });
  btnCompletionBackGallery?.addEventListener('click', () => {
    completionModal?.classList.add('hidden');
    returnToGallery();
  });

  // Mode controls
  modePillWatch?.addEventListener('click', () => tutor.enterWatchMode());
  modePillVerify?.addEventListener('click', () => tutor.enterVerifyMode());
  btnStartVerify?.addEventListener('click', () => tutor.enterVerifyMode());
  btnBackToWatch?.addEventListener('click', () => tutor.enterWatchMode());

  // Video playback controls
  btnVideoPlay?.addEventListener('click', () => {
    if (!tutorialVideo) return;
    if (tutorialVideo.paused) {
      tutorialVideo.play();
      btnVideoPlay.textContent = 'หยุด';
    } else {
      tutorialVideo.pause();
      btnVideoPlay.textContent = 'เล่น';
    }
  });

  btnSpeed05?.addEventListener('click', () => tutor.setPlaybackSpeed(0.5));
  btnSpeed075?.addEventListener('click', () => tutor.setPlaybackSpeed(0.75));
  btnSpeed10?.addEventListener('click', () => tutor.setPlaybackSpeed(1.0));
  btnVideoLoop?.addEventListener('click', () => tutor.toggleLoop());

  btnVideoMute?.addEventListener('click', () => {
    if (!tutorialVideo) return;
    tutorialVideo.muted = !tutorialVideo.muted;
    btnVideoMute.textContent = tutorialVideo.muted ? 'ปิดเสียง' : 'เปิดเสียง';
  });

  // Camera toolbar
  btnStart?.addEventListener('click', startCamera);
  btnStop?.addEventListener('click', stopCamera);
  btnRefreshCameras?.addEventListener('click', refreshCameraList);
  if (cameraSelect) {
    cameraSelect.addEventListener('change', () => {
      cameraAdapter.setSelectedDeviceId(cameraSelect.value);
    });
  }

  if (toggleDetectionBox) {
    toggleDetectionBox.addEventListener('change', (e) => {
      hudOverlay.setEnabled((e.target as HTMLInputElement).checked);
    });
  }

  // Escape hatch & reset
  btnSkipStep?.addEventListener('click', () => {
    tutor.skipCurrentStep();
  });

  btnReset?.addEventListener('click', () => {
    if (confirm('ต้องการรีเซ็ตความคืบหน้าเพื่อเริ่มพับ Step 1 ใหม่หรือไม่?')) {
      tutor.resetProgress();
      stopCamera();
    }
  });

  btnRestartCourse?.addEventListener('click', () => {
    completionModal?.classList.add('hidden');
    tutor.resetProgress();
    stopCamera();
  });

  // --- Initial Mount Execution ---
  fetchFigures();
  refreshCameraList();
});
