/**
 * Canonical Types & API Contracts for GamiSenpai
 */

export type OrigamiStepClass =
  | 'Chk0_NonOrigami'
  | 'Chk1'
  | 'Chk2'
  | 'Chk3'
  | 'Chk4'
  | 'Chk5'
  | 'Chk6';

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface ROIData {
  applied: boolean;
  method?: string;
  bbox?: [number, number, number, number];
  polygon?: [number, number][] | null;
  fusion_mode?: string;
  crop_weight?: number;
  original_top_confidence?: number;
  cropped_top_confidence?: number;
}

export interface PredictionCandidate {
  class_name: OrigamiStepClass;
  step_number: number;
  step_name: string;
  confidence: number;
}

export interface VerificationResult {
  frame_ready?: boolean;
  reason?: string;
  recognized?: boolean;
  recognition_reason?: string;
  prediction?: OrigamiStepClass | string;
  step?: OrigamiStepClass | string;
  step_number?: number | null;
  step_name?: string;
  confidence?: number;
  top_gap?: number;
  recognition_thresholds?: {
    min_confidence: number;
    min_top_gap: number;
  };
  roi?: ROIData;
  top_predictions?: PredictionCandidate[];
  coaching_hint?: string | null;
  error?: string;
}

export interface StepDefinition {
  name: string;
  tip: string;
  reference: string;
}

export const TOTAL_STEPS = 6;
export const MATCH_STREAK_REQUIRED = 3;
export const MIN_DISPLAY_CONFIDENCE = 25;

export const STEPS: StepDefinition[] = [
  {
    name: 'Square Base',
    tip: 'Place flat on table centered in frame. Collapse the sheet into a neat layered square base and show the open end clearly.',
    reference: 'Make a neat layered square base and show the open end clearly.',
  },
  {
    name: 'Kite Folds',
    tip: 'Fold both side edges toward the center line. Keep the piece flat and make the left and right folds balanced without gaps or overlap.',
    reference: 'Look for a long, flat diamond with both side edges folded to the center.',
  },
  {
    name: 'Bird Base',
    tip: 'Flatten the long diamond shape cleanly on both sides. The two lower legs should separate and point in the correct direction.',
    reference: 'Look for the narrow bird base with two long lower points ready to become neck and tail.',
  },
  {
    name: 'Neck & Tail Setup',
    tip: 'Lift the two narrow points to form the neck and tail. Keep both sides balanced; the head is not folded yet.',
    reference: 'Raise both narrow points into a neck and tail. The neck tip is still straight, with no head fold.',
  },
  {
    name: 'Head Fold',
    tip: 'Fold the tip of the neck downward to form the head. Keep the neck, tail, and folded wings in place.',
    reference: 'Fold the neck tip down to form the head while the wings remain folded.',
  },
  {
    name: 'Completed Crane',
    tip: 'Place the crane on table or spread both wings facing the camera with head, tail, and wings clearly visible.',
    reference: 'Show the finished crane: folded head, tail, and both wings clearly visible.',
  },
];

export interface StepMatchRule {
  minConfidence: number;
  minTopGap: number;
}

export const STEP_MATCH_RULES: Record<number, StepMatchRule> = {
  1: { minConfidence: 35, minTopGap: 5 },
  2: { minConfidence: 35, minTopGap: 5 },
  3: { minConfidence: 35, minTopGap: 5 },
  4: { minConfidence: 45, minTopGap: 5 },
  5: { minConfidence: 48, minTopGap: 5 },
  6: { minConfidence: 45, minTopGap: 5 },
};

export function getStepNumber(stepKey?: string | null): number {
  if (!stepKey || stepKey.includes('NonOrigami') || stepKey === 'Chk0') return 0;
  return Number(stepKey.replace('Chk', ''));
}

export type TutorMode = 'watch' | 'verify';
export type PlaybackSpeed = 0.5 | 0.75 | 1.0;

export interface TutorialProgress {
  currentStep: number;
  completedSteps: number[];
  lastActiveTime: string;
}

export interface StepFoldGuide {
  stepNumber: number;
  name: string;
  thaiName: string;
  foldTip: string;
  checkPointTip: string;
  videoUrl: string;
  fallbackImageUrl: string;
}

export const STEP_FOLD_GUIDES: StepFoldGuide[] = [
  {
    stepNumber: 1,
    name: 'Square Base',
    thaiName: 'ฐานสี่เหลี่ยม (Square Base)',
    foldTip: 'พับครึ่งทแยงมุมและแนวขนาน รีดรอยพับให้คมชัด แล้วยุบมุมเข้าหากันจนเป็นสี่เหลี่ยมจัตุรัสซ้อนกัน',
    checkPointTip: 'วางชิ้นงานให้ปลายเปิดชี้เข้าหากล้อง ตรวจสอบให้สันพับทั้งสี่ด้านเรียบเสมอกัน',
    videoUrl: '/videos/step-1.mp4',
    fallbackImageUrl: '/references/step-1.jpg',
  },
  {
    stepNumber: 2,
    name: 'Kite Folds',
    thaiName: 'พับสามเหลี่ยมว่าว (Kite Folds)',
    foldTip: 'หันด้านปลายเปิดลง แล้วพับขอบด้านข้างทั้งซ้ายและขวาเข้าไปชิดเส้นกึ่งกลาง (ทำทั้งด้านหน้าและด้านหลัง)',
    checkPointTip: 'รอยพับสองข้างต้องประกบชิดเส้นกึ่งกลางพอดี ไม่เกยทับหรือห่างกันจนเกินไป',
    videoUrl: '/videos/step-2.mp4',
    fallbackImageUrl: '/references/step-2.jpg',
  },
  {
    stepNumber: 3,
    name: 'Bird Base',
    thaiName: 'ฐานนก (Bird Base / Petal Fold)',
    foldTip: 'กางรอยพับด้านข้างออก เปิดแผ่นกระดาษขึ้นแล้วรีดพับกลีบ (Petal fold) รีดแบนให้เป็นรูปข้าวหลามตัดยาว',
    checkPointTip: 'ด้านล่างจะมีสองแฉกที่แยกออกจากกันอิสระ พร้อมจะดึงขึ้นเป็นคอและหาง',
    videoUrl: '/videos/step-3.mp4',
    fallbackImageUrl: '/references/step-3.jpg',
  },
  {
    stepNumber: 4,
    name: 'Neck & Tail Setup',
    thaiName: 'ดึงคอและหาง (Neck & Tail)',
    foldTip: 'พับทบด้านข้าง แล้วดึงปลายแหลมด้านล่างทั้งสองข้างเฉียงขึ้นเฉียงออก เพื่อสร้างคอและหาง',
    checkPointTip: 'ทั้งคอและหางยกขึ้นทำมุมสวยงาม ปลายทั้งสองข้างยังตรงอยู่ ยังไม่พับหัวนก',
    videoUrl: '/videos/step-4.mp4',
    fallbackImageUrl: '/references/step-4.jpg',
  },
  {
    stepNumber: 5,
    name: 'Head Fold',
    thaiName: 'พับหัวนก (Head Fold)',
    foldTip: 'เลือกปลายด้านใดด้านหนึ่งเป็นคอ จากนั้นกดรอยพับย้อนกลับ (Reverse fold) พับปลายแหลมลงมาเป็นหัวนก',
    checkPointTip: 'หัวนกต้องพับชี้ลงมาชัดเจน ขณะที่ปีกทั้งสองข้างยังคงประกบแนบสนิทอยู่',
    videoUrl: '/videos/step-5.mp4',
    fallbackImageUrl: '/references/step-5.jpg',
  },
  {
    stepNumber: 6,
    name: 'Completed Crane',
    thaiName: 'นกกระเรียนสมบูรณ์ (Completed Crane)',
    foldTip: 'ใช้นิ้วค่อยๆ ดึงปีกทั้งสองข้างกางออกจากกันเบาๆ ให้ลำตัวนกพองออกอย่างเป็นธรรมชาติ',
    checkPointTip: 'กางปีกทั้งสองข้างหันเข้าหากล้อง ลำตัวพองสวยงาม มองเห็นทั้งหัว คอ ปีก และหางชัดเจน',
    videoUrl: '/videos/step-6.mp4',
    fallbackImageUrl: '/references/step-6.jpg',
  },
];

export interface OrigamiFigureSummary {
  id: string;
  title_th: string;
  title_en: string;
  subtitle?: string;
  description_th: string;
  description_en: string;
  status: 'available' | 'coming_soon';
  difficulty: string;
  difficulty_level: number;
  estimated_minutes: number;
  thumbnail: string;
  icon: string;
  total_steps: number;
}

export interface FigureStepDetail {
  step_number: number;
  class_name: string;
  name_en: string;
  name_th: string;
  fold_tip: string;
  checkpoint_tip: string;
  reference_description: string;
  video_url: string;
  fallback_image_url: string;
  min_confidence?: number;
}

export interface OrigamiFigureDetail extends OrigamiFigureSummary {
  steps: FigureStepDetail[];
}

