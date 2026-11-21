import {
  VerificationResult,
  TOTAL_STEPS,
  MATCH_STREAK_REQUIRED,
  MIN_DISPLAY_CONFIDENCE,
  STEP_MATCH_RULES,
  STEPS,
  getStepNumber,
  StepDefinition,
} from './types';

export type SessionStatusKind = 'waiting' | 'correct' | 'incorrect';

export interface UIStatusState {
  kind: SessionStatusKind;
  icon: string;
  title: string;
  description: string;
}

export type EvaluationAction =
  | { type: 'FRAME_NOT_READY'; status: UIStatusState }
  | { type: 'UNRECOGNIZED'; status: UIStatusState; tip: string }
  | { type: 'LOW_CONFIDENCE'; status: UIStatusState; tip: string; detectedStep: number; confidence: number }
  | { type: 'LOCKED_MISMATCH'; status: UIStatusState; tip: string; detectedStep: number; confidence: number; overlayState: 'tracking' | 'locked' }
  | { type: 'CONFIRMING'; status: UIStatusState; detectedStep: number; confidence: number; streak: number; tip?: string }
  | { type: 'MATCH_SUCCESS'; status: UIStatusState; finishedStep: number; nextStep: number | null; confidence: number }
  | { type: 'ALL_COMPLETE'; status: UIStatusState; confidence: number };

export interface SessionEngineState {
  currentStep: number;
  completedSteps: Set<number>;
  stepStreak: number;
  isAdvancing: boolean;
}

export function formatFrameNotReadyStatus(reason?: string): UIStatusState {
  if (reason === 'camera_placeholder_or_too_dark') {
    return {
      kind: 'waiting',
      icon: '💡',
      title: 'แสงสว่างไม่เพียงพอ',
      description: 'ภาพจากกล้องมืดเกินไป กรุณาเปิดไฟหรือขยับกระดาษมาในจุดที่มีแสงสว่าง',
    };
  }
  if (reason === 'frame_has_too_little_detail') {
    return {
      kind: 'waiting',
      icon: '🔍',
      title: 'ภาพไม่ชัดเจนหรือแสงสะท้อนจ้า',
      description: 'ภาพขาดความคมชัด กรุณาถือกระดาษให้อยู่นิ่งๆ และหลีกเลี่ยงแสงสะท้อนจ้า',
    };
  }
  if (reason === 'empty_frame') {
    return {
      kind: 'waiting',
      icon: '📷',
      title: 'ไม่พบสัญญาณภาพจากกล้อง',
      description: 'กรุณาเปิดกล้องและจัดตำแหน่งกระดาษให้อยู่กึ่งกลางหน้าจอ',
    };
  }
  return {
    kind: 'waiting',
    icon: '📷',
    title: 'จัดตำแหน่งกระดาษในกล้อง',
    description: 'กรุณาส่องกล้องไปที่กระดาษโอริกามิให้เห็นชัดเจน',
  };
}

export class StepSessionEngine {
  private currentStep: number = 1;
  private totalSteps: number = TOTAL_STEPS;
  private steps: StepDefinition[] = STEPS;
  private completedSteps: Set<number> = new Set();
  private stepStreak: number = 0;
  private isAdvancing: boolean = false;

  constructor(initialStep: number = 1, totalSteps: number = TOTAL_STEPS, steps: StepDefinition[] = STEPS) {
    this.currentStep = initialStep;
    this.totalSteps = totalSteps;
    this.steps = steps;
  }

  public configure(totalSteps: number, steps: StepDefinition[]): void {
    this.totalSteps = totalSteps;
    this.steps = steps;
    this.reset();
  }

  public reconfigure(totalSteps: number, steps: StepDefinition[]): void {
    this.configure(totalSteps, steps);
  }

  public getState(): SessionEngineState {
    return {
      currentStep: this.currentStep,
      completedSteps: new Set(this.completedSteps),
      stepStreak: this.stepStreak,
      isAdvancing: this.isAdvancing,
    };
  }

  public reset(): void {
    this.currentStep = 1;
    this.completedSteps.clear();
    this.stepStreak = 0;
    this.isAdvancing = false;
  }

  public setStep(step: number, completed?: number[]): void {
    this.currentStep = Math.max(1, Math.min(this.totalSteps + 1, step));
    this.completedSteps.clear();
    if (completed) {
      completed.forEach((s) => this.completedSteps.add(s));
    } else {
      for (let i = 1; i < this.currentStep; i++) {
        this.completedSteps.add(i);
      }
    }
    this.stepStreak = 0;
    this.isAdvancing = false;
  }

  public decayStreak(): number {
    this.stepStreak = Math.max(0, this.stepStreak - 1);
    return this.stepStreak;
  }

  public advanceToNextStep(): void {
    if (this.currentStep <= this.totalSteps) {
      this.completedSteps.add(this.currentStep);
    }
    this.currentStep += 1;
    this.stepStreak = 0;
    this.isAdvancing = false;
  }

  public evaluatePrediction(data: VerificationResult): EvaluationAction {
    if (this.isAdvancing) {
      return {
        type: 'CONFIRMING',
        status: {
          kind: 'waiting',
          icon: '...',
          title: `Step ${this.currentStep} unlocked`,
          description: 'Transitioning to next step...',
        },
        detectedStep: this.currentStep,
        confidence: 0,
        streak: 0,
      };
    }

    if (this.currentStep > this.totalSteps) {
      return {
        type: 'ALL_COMPLETE',
        status: {
          kind: 'correct',
          icon: 'OK',
          title: 'All steps matched!',
          description: `The model detected all ${this.totalSteps} checkpoints in order.`,
        },
        confidence: data.confidence || 0,
      };
    }

    // Frame not ready
    if (data.frame_ready === false) {
      this.decayStreak();
      return {
        type: 'FRAME_NOT_READY',
        status: formatFrameNotReadyStatus(data.reason || data.recognition_reason),
      };
    }

    // Unrecognized or Non-Origami
    if (data.recognized === false) {
      this.decayStreak();
      return {
        type: 'UNRECOGNIZED',
        status: this.formatUnrecognizedStatus(data),
        tip: data.coaching_hint || this.steps[this.currentStep - 1]?.tip || '',
      };
    }

    const confidence = Number(data.confidence || 0);

    // Below display confidence
    if (confidence < MIN_DISPLAY_CONFIDENCE) {
      this.decayStreak();
      return {
        type: 'LOW_CONFIDENCE',
        status: {
          kind: 'waiting',
          icon: '...',
          title: 'ยังไม่มั่นใจเพียงพอ',
          description: 'โมเดลยังไม่แน่ใจ ถือรอยพับปัจจุบันให้ชัดเจน สว่างขึ้น และอยู่กลางภาพ',
        },
        tip: data.coaching_hint || this.steps[this.currentStep - 1]?.tip || '',
        detectedStep: getStepNumber(data.step),
        confidence,
      };
    }

    const detectedStep = getStepNumber(data.step);

    // Mismatched step
    if (detectedStep !== this.currentStep) {
      this.decayStreak();

      // Special Handling for Step 6: Step 6 (Completed Crane) shares morphology with Step 5 (Head Fold).
      if (this.currentStep === 6 && detectedStep === 5) {
        return {
          type: 'LOCKED_MISMATCH',
          status: {
            kind: 'waiting',
            icon: 'WING',
            title: 'เกือบเสร็จแล้ว! กางปีกออกให้กว้าง',
            description: 'กางปีกนกทั้งสองข้างออกให้กว้างและหงายเข้าหากล้อง เพื่อให้นกกระเรียนสมบูรณ์ (Step 6)',
          },
          tip: 'กางปีกทั้งสองข้างออกให้ขนานกับกล้อง นกกระเรียนจะสมบูรณ์ใน Step 6 ครับ',
          detectedStep,
          confidence,
          overlayState: 'tracking',
        };
      }

      // Gentle transition if user shows previous step
      if (detectedStep === this.currentStep - 1) {
        return {
          type: 'LOCKED_MISMATCH',
          status: {
            kind: 'waiting',
            icon: '...',
            title: `กำลังเข้าสู่ขั้นตอนที่ ${this.currentStep}`,
            description: `พับต่อจากขั้นตอนที่ ${detectedStep} ตามตัวอย่างด้านขวาเพื่อเข้าสู่ขั้นตอนที่ ${this.currentStep}: ${this.steps[this.currentStep - 1]?.name || ''}`,
          },
          tip: this.steps[this.currentStep - 1]?.tip || '',
          detectedStep,
          confidence,
          overlayState: 'tracking',
        };
      }

      // Hard locked mismatch
      const direction = detectedStep > this.currentStep ? 'ahead' : 'earlier';
      return {
        type: 'LOCKED_MISMATCH',
        status: {
          kind: 'incorrect',
          icon: 'LOCK',
          title: direction === 'ahead' ? `ขั้นตอนที่ ${detectedStep} ยังไม่ปลดล็อก` : `ย้อนกลับมาขั้นตอนที่ ${this.currentStep}`,
          description: `AI ตรวจพบขั้นตอนที่ ${detectedStep} แต่ขั้นตอนปัจจุบันคือ ${this.steps[this.currentStep - 1]?.name || `ขั้นตอนที่ ${this.currentStep}`}`,
        },
        tip: direction === 'ahead'
          ? `อย่าเพิ่งข้ามขั้นตอนครับ ทำให้ผ่านขั้นตอนที่ ${this.currentStep} ก่อน แล้วขั้นตอนที่ ${detectedStep} จะปลดล็อกเอง`
          : `ขั้นตอนที่ ${detectedStep} ผ่านไปแล้วครับ กรุณาทำขั้นตอนปัจจุบันที่ ${this.currentStep}`,
        detectedStep,
        confidence,
        overlayState: 'locked',
      };
    }

    // Step matches: Accumulate streak with Leaky Bucket
    this.stepStreak = Math.min(MATCH_STREAK_REQUIRED, this.stepStreak + 1);

    // If streak reached requirement, evaluate match rule
    if (this.stepStreak >= MATCH_STREAK_REQUIRED) {
      const matchRule = STEP_MATCH_RULES[this.currentStep] || { minConfidence: 55, minTopGap: 10 };
      const topGap = Number(data.top_gap || 0);

      if (confidence < matchRule.minConfidence || topGap < matchRule.minTopGap) {
        this.decayStreak();
        return {
          type: 'CONFIRMING',
          status: {
            kind: 'waiting',
            icon: 'WAIT',
            title: `ตรวจพบขั้นตอนที่ ${this.currentStep} แล้ว แต่ความชัดเจนยังน้อยไป`,
            description: `ความมั่นใจ ${confidence.toFixed(1)}%, กรุณาถือกระดาษให้อยู่กลางกล้องและนิ่งขึ้นอีกนิด`,
          },
          detectedStep,
          confidence,
          streak: this.stepStreak,
          tip: data.coaching_hint || this.steps[this.currentStep - 1]?.tip || '',
        };
      }

      // Checkpoint matched successfully
      this.isAdvancing = true;
      const finished = this.currentStep;
      const next = finished === this.totalSteps ? null : finished + 1;

      if (next === null) {
        this.completedSteps.add(finished);
        this.currentStep = this.totalSteps + 1;
        this.isAdvancing = false;
        return {
          type: 'ALL_COMPLETE',
          status: {
            kind: 'correct',
            icon: 'OK',
            title: 'พับสำเร็จครบทุกขั้นตอนแล้ว',
            description: `ระบบตรวจพบความถูกต้องครบทั้ง ${this.totalSteps} ขั้นตอนเรียบร้อยแล้ว`,
          },
          confidence,
        };
      }

      return {
        type: 'MATCH_SUCCESS',
        status: {
          kind: 'correct',
          icon: 'OK',
          title: `ขั้นตอนที่ ${finished} ถูกต้องเรียบร้อย!`,
          description: `กำลังนำท่านเข้าสู่วิดีโอสอนขั้นตอนที่ ${next}...`,
        },
        finishedStep: finished,
        nextStep: next,
        confidence,
      };
    }

    // Accumulating streak
    return {
      type: 'CONFIRMING',
      status: {
        kind: 'waiting',
        icon: '...',
        title: `กำลังตรวจความถูกต้อง (${this.stepStreak}/${MATCH_STREAK_REQUIRED})`,
        description: 'ถือชิ้นงานท่านี้ไว้นิ่งๆ เพื่อให้ระบบยืนยันผล',
      },
      detectedStep,
      confidence,
      streak: this.stepStreak,
      tip: data.coaching_hint || this.steps[this.currentStep - 1]?.tip || '',
    };
  }

  private formatUnrecognizedStatus(data: VerificationResult): UIStatusState {
    const confidence = Number(data.confidence || 0);
    const topGap = Number(data.top_gap || 0);

    if (data.recognition_reason === 'ambiguous_step') {
      let desc = `โมเดลเห็นความคล้ายคลึงระหว่างขั้นตอน (top-gap ${topGap.toFixed(1)}%) กรุณาจัดกระดาษให้ตรงตามตัวอย่าง`;
      if (data.top_predictions && data.top_predictions.length >= 2) {
        const first = data.top_predictions[0].step_name || data.top_predictions[0].class_name;
        const second = data.top_predictions[1].step_name || data.top_predictions[1].class_name;
        if (first && second) {
          desc = `โมเดลเห็นความคล้ายคลึงระหว่าง ${first} และ ${second} (top-gap ${topGap.toFixed(1)}%) กรุณาจัดกระดาษให้ตรงตามตัวอย่าง`;
        }
      }
      return {
        kind: 'waiting',
        icon: '?',
        title: 'รอยพับก้ำกึ่งระหว่างขั้นตอน',
        description: desc,
      };
    }

    if (
      data.prediction === 'Chk0_NonOrigami' ||
      data.recognition_reason === 'non_origami' ||
      !data.step_number
    ) {
      return {
        kind: 'waiting',
        icon: 'SCAN',
        title: 'ส่องไปที่กระดาษโอริกามิ',
        description: 'นำกระดาษโอริกามิมาวางไว้กลางกล้องเพื่อเริ่มการตรวจเช็ค',
      };
    }

    if (data.recognition_reason === 'invalid_shape_for_step1') {
      return {
        kind: 'waiting',
        icon: '--',
        title: 'ยังไม่ใช่ทรงฐานสี่เหลี่ยม',
        description: 'ขั้นตอนที่ 1 ต้องพับประกบมุมเป็นรูปสี่เหลี่ยมจัตุรัสซ้อนกัน (Square Base)',
      };
    }

    if (data.recognition_reason === 'low_confidence') {
      return {
        kind: 'waiting',
        icon: '...',
        title: `เห็นรอยพับแล้ว แต่ยังไม่ชัดเจน (${confidence.toFixed(0)}%)`,
        description: 'กรุณาถือกระดาษให้อยู่กลางกล้อง และปรับมุมให้แสงตกกระทบรอยพับชัดเจนขึ้น',
      };
    }

    return {
      kind: 'waiting',
      icon: '...',
      title: 'กำลังจัดตำแหน่งชิ้นงาน...',
      description: 'แสดงกระดาษที่พับเสร็จให้อยู่ตรงกลางกรอบภาพ',
    };
  }
}
