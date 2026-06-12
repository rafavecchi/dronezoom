import * as ort from 'onnxruntime-web';

// Keep in sync with the pinned onnxruntime-web version in package.json —
// the .wasm binaries must match the installed JS exactly.
const ORT_VERSION = '1.26.0';
const MODEL_URL = `${import.meta.env.BASE_URL}models/yolo11n-detect.onnx`;
const INPUT_SIZE = 640;
const PERSON_CLASS = 0;
const SCORE_THRESHOLD = 0.35;
const IOU_THRESHOLD = 0.45;

export interface Detection {
  cx: number;
  cy: number;
  w: number;
  h: number;
  score: number;
}

let session: ort.InferenceSession | null = null;

const workCanvas = document.createElement('canvas');
workCanvas.width = INPUT_SIZE;
workCanvas.height = INPUT_SIZE;
const workCtx = workCanvas.getContext('2d', { willReadFrequently: true })!;

export async function initDetector(): Promise<string> {
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  let lastError: unknown;
  for (const ep of ['webgpu', 'wasm'] as const) {
    try {
      session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: [ep],
        graphOptimizationLevel: 'all',
      });
      return ep;
    } catch (e) {
      lastError = e;
    }
  }
  throw new Error(`Could not initialize ONNX Runtime: ${lastError}`);
}

export interface Region {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Detect persons in the frame, or in a sub-region of it. Passing a region
 * effectively zooms the detector in — essential for riders that are only
 * a few dozen pixels tall in a full 4K frame downscaled to 640.
 */
export async function detectPersons(
  frame: CanvasImageSource,
  srcW: number,
  srcH: number,
  region?: Region,
): Promise<Detection[]> {
  if (!session) throw new Error('Detector not initialized');

  const rx = region?.x ?? 0;
  const ry = region?.y ?? 0;
  const rw = region?.w ?? srcW;
  const rh = region?.h ?? srcH;

  const scale = Math.min(INPUT_SIZE / rw, INPUT_SIZE / rh);
  const drawW = Math.round(rw * scale);
  const drawH = Math.round(rh * scale);
  const dx = (INPUT_SIZE - drawW) / 2;
  const dy = (INPUT_SIZE - drawH) / 2;

  workCtx.fillStyle = '#727272';
  workCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  workCtx.drawImage(frame, rx, ry, rw, rh, dx, dy, drawW, drawH);

  const { data } = workCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const pixels = INPUT_SIZE * INPUT_SIZE;
  const input = new Float32Array(3 * pixels);
  for (let i = 0; i < pixels; i++) {
    input[i] = data[i * 4] / 255;
    input[pixels + i] = data[i * 4 + 1] / 255;
    input[2 * pixels + i] = data[i * 4 + 2] / 255;
  }

  const tensor = new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const outputs = await session.run({ [session.inputNames[0]]: tensor });
  const out = outputs[session.outputNames[0]];

  // YOLO11 detect head: [1, 4 + numClasses, numAnchors], boxes as cx,cy,w,h in input scale.
  const [, channels, anchors] = out.dims as number[];
  const scores = out.data as Float32Array;
  if (channels < 4 + PERSON_CLASS + 1) return [];

  const candidates: Detection[] = [];
  const personOffset = (4 + PERSON_CLASS) * anchors;
  for (let i = 0; i < anchors; i++) {
    const score = scores[personOffset + i];
    if (score < SCORE_THRESHOLD) continue;
    candidates.push({
      cx: rx + (scores[i] - dx) / scale,
      cy: ry + (scores[anchors + i] - dy) / scale,
      w: scores[2 * anchors + i] / scale,
      h: scores[3 * anchors + i] / scale,
      score,
    });
  }
  return nms(candidates);
}

function nms(dets: Detection[]): Detection[] {
  dets.sort((a, b) => b.score - a.score);
  const kept: Detection[] = [];
  for (const det of dets) {
    if (kept.every((k) => iou(k, det) < IOU_THRESHOLD)) kept.push(det);
  }
  return kept;
}

function iou(a: Detection, b: Detection): number {
  const ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2, ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
  const bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2, bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;
  const ix = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
  const iy = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  const inter = ix * iy;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}
