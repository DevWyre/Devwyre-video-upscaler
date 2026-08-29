import type { ResolutionLike, UpscalerLike, UpscalerSource } from './upscaler';

/**
 * Pure-JavaScript reimplementation of the Anime4K "cnn-2x" super-resolution
 * networks that WebSR normally runs as handwritten WGSL compute shaders on
 * WebGPU.
 *
 * This lets the app actually upscale on any browser that supports WebCodecs
 * (that is, every modern evergreen browser) even when WebGPU is unavailable —
 * Firefox, older Safari, most mobile browsers, hardened/enterprise Chromium,
 * etc. It is intentionally faithful to the shader math so the output matches
 * the GPU path:
 *
 *   - conv3  : zero-padded 3x3 convolution from the RGBA input texture (no ReLU)
 *   - conv8  : 3x3 convolution on a 4-ch vec4 buffer with twin pos/neg kernels
 *              (max(x,0) * Wpos + max(-x,0) * Wneg) — Anime4K's ReLU trick
 *   - conv16 : same twin-kernel trick over two 4-ch input buffers (Large net)
 *   - conv56 : 1x1 twin-kernel conv over seven 4-ch buffers (Medium tail)
 *   - conv112: 1x1 twin-kernel conv over seven 4-ch buffers (Large tail)
 *   - concat : pointwise sum of two buffers + bias (Large tail)
 *   - display: 2x pixel-shuffle of the residual buffer(s) added on top of a
 *              bilinear upscale of the original frame
 *
 * Buffers are stored per-pixel as vec4 (RGBA in float 0..1), matching the
 * `array<vec4f>` storage buffers used by the WebGPU shaders. Weights are laid
 * out exactly as WebGPU receives them: every `kernels[i]` is a WGSL
 * `mat4x4f` in column-major order, so matrix element (column c, row r) lives
 * at `block + c*4 + r`.
 */

type NetworkArch = 's' | 'm' | 'l';

const TAP_OFFSETS: ReadonlyArray<readonly [number, number]> = [
  [-1, -1], [-1, 0], [-1, 1],
  [0, -1], [0, 0], [0, 1],
  [1, -1], [1, 0], [1, 1],
];

function archFromName(name: string): NetworkArch {
  if (name === 'anime4k/cnn-2x-s') return 's';
  if (name === 'anime4k/cnn-2x-l') return 'l';
  return 'm';
}

/** Round a float and clamp into an 8-bit channel value (like rgba8unorm). */
function clampByte(v: number): number {
  if (v <= 0) return 0;
  if (v >= 1) return 255;
  return (v * 255 + 0.5) | 0;
}

/**
 * Bilinear sample a 0..1 float RGBA image at floating pixel coords (sx, sy).
 * Mirrors WebGPU's linear-filtered textureSample (repeat address mode is a
 * no-op for the pixel interiors the display layer ever samples).
 */
function sampleBilinear(
  img: Float32Array,
  w: number,
  h: number,
  sx: number,
  sy: number,
  c: 0 | 1 | 2 | 3
): number {
  let x0 = Math.floor(sx);
  let y0 = Math.floor(sy);
  let tx = sx - x0;
  let ty = sy - y0;

  if (x0 < 0) { x0 = 0; tx = 0; }
  if (y0 < 0) { y0 = 0; ty = 0; }

  let x1 = x0 + 1;
  let y1 = y0 + 1;
  if (x1 > w - 1) { x1 = w - 1; tx = 1; }
  if (y1 > h - 1) { y1 = h - 1; ty = 1; }

  const p00 = (y0 * w + x0) * 4 + c;
  const p10 = (y0 * w + x1) * 4 + c;
  const p01 = (y1 * w + x0) * 4 + c;
  const p11 = (y1 * w + x1) * 4 + c;

  const wx0 = 1 - tx;
  const wy0 = 1 - ty;

  return (
    img[p00] * wx0 * wy0 +
    img[p10] * tx * wy0 +
    img[p01] * wx0 * ty +
    img[p11] * tx * ty
  );
}

/**
 * 3x3 conv from the RGBA input image (zero-padded edges — matches WGSL
 * out-of-bounds textureLoad returning zero). 4 input -> 4 output channels.
 * One 4x4 matrix per tap, no ReLU.
 */
function conv3x4(
  img: Float32Array,
  k: Float32Array,
  b: Float32Array,
  out: Float32Array,
  w: number,
  h: number
): void {
  const imgW = w * 4;
  for (let y = 0; y < h; y++) {
    const yBase = y * imgW;
    for (let x = 0; x < w; x++) {
      let r0 = 0, r1 = 0, r2 = 0, r3 = 0;
      let kk = 0;

      for (let d = 0; d < 9; d++) {
        const xx = x + TAP_OFFSETS[d][0];
        const yy = y + TAP_OFFSETS[d][1];
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
          kk += 16;
          continue;
        }

        const p = yy * imgW + xx * 4;
        const c0 = img[p], c1 = img[p + 1], c2 = img[p + 2], c3 = img[p + 3];

        r0 += k[kk] * c0 + k[kk + 4] * c1 + k[kk + 8] * c2 + k[kk + 12] * c3;
        r1 += k[kk + 1] * c0 + k[kk + 5] * c1 + k[kk + 9] * c2 + k[kk + 13] * c3;
        r2 += k[kk + 2] * c0 + k[kk + 6] * c1 + k[kk + 10] * c2 + k[kk + 14] * c3;
        r3 += k[kk + 3] * c0 + k[kk + 7] * c1 + k[kk + 11] * c2 + k[kk + 15] * c3;
        kk += 16;
      }

      const o = yBase + x * 4;
      out[o] = r0 + b[0];
      out[o + 1] = r1 + b[1];
      out[o + 2] = r2 + b[2];
      out[o + 3] = r3 + b[3];
    }
  }
}

/**
 * 3x3 conv over a vec4 storage buffer with the twin pos/neg kernel trick.
 * 18 mat4x4 kernels: kernels[tap] acts on max(pix,0), kernels[tap+9] on
 * max(-pix,0). Edges clamp to the border (the GPU path reads adjacent
 * undefined memory here; clamping is the sensible interpretation).
 */
function conv8x4(
  src: Float32Array,
  k: Float32Array,
  b: Float32Array,
  out: Float32Array,
  w: number,
  h: number
): void {
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r0 = 0, r1 = 0, r2 = 0, r3 = 0;
      let kk = 0;

      for (let d = 0; d < 9; d++) {
        let xx = x + TAP_OFFSETS[d][0];
        let yy = y + TAP_OFFSETS[d][1];
        if (xx < 0) xx = 0;
        if (yy < 0) yy = 0;
        if (xx >= w) xx = w - 1;
        if (yy >= h) yy = h - 1;

        const p = (yy * w + xx) * 4;
        const c0 = src[p], c1 = src[p + 1], c2 = src[p + 2], c3 = src[p + 3];

        const p0 = c0 > 0 ? c0 : 0, n0 = c0 < 0 ? -c0 : 0;
        const p1 = c1 > 0 ? c1 : 0, n1 = c1 < 0 ? -c1 : 0;
        const p2 = c2 > 0 ? c2 : 0, n2 = c2 < 0 ? -c2 : 0;
        const p3 = c3 > 0 ? c3 : 0, n3 = c3 < 0 ? -c3 : 0;

        const pos = kk;
        const neg = kk + 144;
        r0 += k[pos] * p0 + k[pos + 4] * p1 + k[pos + 8] * p2 + k[pos + 12] * p3
          + k[neg] * n0 + k[neg + 4] * n1 + k[neg + 8] * n2 + k[neg + 12] * n3;
        r1 += k[pos + 1] * p0 + k[pos + 5] * p1 + k[pos + 9] * p2 + k[pos + 13] * p3
          + k[neg + 1] * n0 + k[neg + 5] * n1 + k[neg + 9] * n2 + k[neg + 13] * n3;
        r2 += k[pos + 2] * p0 + k[pos + 6] * p1 + k[pos + 10] * p2 + k[pos + 14] * p3
          + k[neg + 2] * n0 + k[neg + 6] * n1 + k[neg + 10] * n2 + k[neg + 14] * n3;
        r3 += k[pos + 3] * p0 + k[pos + 7] * p1 + k[pos + 11] * p2 + k[pos + 15] * p3
          + k[neg + 3] * n0 + k[neg + 7] * n1 + k[neg + 11] * n2 + k[neg + 15] * n3;
        kk += 16;
      }

      const o = (y * w + x) * 4;
      out[o] = r0 + b[0];
      out[o + 1] = r1 + b[1];
      out[o + 2] = r2 + b[2];
      out[o + 3] = r3 + b[3];
    }
  }
}

/**
 * 3x3 conv over two vec4 buffers with four kernel sets (Large net).
 * kernels[tap] on max(a,0), kernels[tap+9] on max(b,0),
 * kernels[tap+18] on max(-a,0), kernels[tap+27] on max(-b,0).
 */
function conv16x4(
  a: Float32Array,
  b2: Float32Array,
  k: Float32Array,
  bias: Float32Array,
  out: Float32Array,
  w: number,
  h: number
): void {
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r0 = 0, r1 = 0, r2 = 0, r3 = 0;
      let kk = 0;

      for (let d = 0; d < 9; d++) {
        let xx = x + TAP_OFFSETS[d][0];
        let yy = y + TAP_OFFSETS[d][1];
        if (xx < 0) xx = 0;
        if (yy < 0) yy = 0;
        if (xx >= w) xx = w - 1;
        if (yy >= h) yy = h - 1;

        const p = (yy * w + xx) * 4;
        const a0 = a[p], a1 = a[p + 1], a2 = a[p + 2], a3 = a[p + 3];
        const bb0 = b2[p], bb1 = b2[p + 1], bb2 = b2[p + 2], bb3 = b2[p + 3];

        const pa0 = a0 > 0 ? a0 : 0, na0 = a0 < 0 ? -a0 : 0;
        const pa1 = a1 > 0 ? a1 : 0, na1 = a1 < 0 ? -a1 : 0;
        const pa2 = a2 > 0 ? a2 : 0, na2 = a2 < 0 ? -a2 : 0;
        const pa3 = a3 > 0 ? a3 : 0, na3 = a3 < 0 ? -a3 : 0;
        const pb0 = bb0 > 0 ? bb0 : 0, nb0 = bb0 < 0 ? -bb0 : 0;
        const pb1 = bb1 > 0 ? bb1 : 0, nb1 = bb1 < 0 ? -bb1 : 0;
        const pb2 = bb2 > 0 ? bb2 : 0, nb2 = bb2 < 0 ? -bb2 : 0;
        const pb3 = bb3 > 0 ? bb3 : 0, nb3 = bb3 < 0 ? -bb3 : 0;

        const pos0 = kk;
        const pos1 = kk + 144;
        const neg0 = kk + 288;
        const neg1 = kk + 432;

        r0 +=
          k[pos0] * pa0 + k[pos0 + 4] * pa1 + k[pos0 + 8] * pa2 + k[pos0 + 12] * pa3 +
          k[pos1] * pb0 + k[pos1 + 4] * pb1 + k[pos1 + 8] * pb2 + k[pos1 + 12] * pb3 +
          k[neg0] * na0 + k[neg0 + 4] * na1 + k[neg0 + 8] * na2 + k[neg0 + 12] * na3 +
          k[neg1] * nb0 + k[neg1 + 4] * nb1 + k[neg1 + 8] * nb2 + k[neg1 + 12] * nb3;
        r1 +=
          k[pos0 + 1] * pa0 + k[pos0 + 5] * pa1 + k[pos0 + 9] * pa2 + k[pos0 + 13] * pa3 +
          k[pos1 + 1] * pb0 + k[pos1 + 5] * pb1 + k[pos1 + 9] * pb2 + k[pos1 + 13] * pb3 +
          k[neg0 + 1] * na0 + k[neg0 + 5] * na1 + k[neg0 + 9] * na2 + k[neg0 + 13] * na3 +
          k[neg1 + 1] * nb0 + k[neg1 + 5] * nb1 + k[neg1 + 9] * nb2 + k[neg1 + 13] * nb3;
        r2 +=
          k[pos0 + 2] * pa0 + k[pos0 + 6] * pa1 + k[pos0 + 10] * pa2 + k[pos0 + 14] * pa3 +
          k[pos1 + 2] * pb0 + k[pos1 + 6] * pb1 + k[pos1 + 10] * pb2 + k[pos1 + 14] * pb3 +
          k[neg0 + 2] * na0 + k[neg0 + 6] * na1 + k[neg0 + 10] * na2 + k[neg0 + 14] * na3 +
          k[neg1 + 2] * nb0 + k[neg1 + 6] * nb1 + k[neg1 + 10] * nb2 + k[neg1 + 14] * nb3;
        r3 +=
          k[pos0 + 3] * pa0 + k[pos0 + 7] * pa1 + k[pos0 + 11] * pa2 + k[pos0 + 15] * pa3 +
          k[pos1 + 3] * pb0 + k[pos1 + 7] * pb1 + k[pos1 + 11] * pb2 + k[pos1 + 15] * pb3 +
          k[neg0 + 3] * na0 + k[neg0 + 7] * na1 + k[neg0 + 11] * na2 + k[neg0 + 15] * na3 +
          k[neg1 + 3] * nb0 + k[neg1 + 7] * nb1 + k[neg1 + 11] * nb2 + k[neg1 + 15] * nb3;
        kk += 16;
      }

      const o = (y * w + x) * 4;
      out[o] = r0 + bias[0];
      out[o + 1] = r1 + bias[1];
      out[o + 2] = r2 + bias[2];
      out[o + 3] = r3 + bias[3];
    }
  }
}

/**
 * 1x1 conv over seven vec4 buffers (Medium tail). For buffer i:
 * result += kernels[2i]*max(pix,0) + kernels[2i+1]*max(-pix,0).
 */
function conv56x4(
  inBufs: Float32Array[],
  k: Float32Array,
  b: Float32Array,
  out: Float32Array,
  w: number,
  h: number
): void {
  const n = w * h;
  for (let i = 0; i < n; i++) {
    let r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    const p = i * 4;

    for (let bi = 0; bi < 7; bi++) {
      const c0 = inBufs[bi][p], c1 = inBufs[bi][p + 1], c2 = inBufs[bi][p + 2], c3 = inBufs[bi][p + 3];
      const pos = bi * 32;
      const neg = pos + 16;

      const p0 = c0 > 0 ? c0 : 0, n0 = c0 < 0 ? -c0 : 0;
      const p1 = c1 > 0 ? c1 : 0, n1 = c1 < 0 ? -c1 : 0;
      const p2 = c2 > 0 ? c2 : 0, n2 = c2 < 0 ? -c2 : 0;
      const p3 = c3 > 0 ? c3 : 0, n3 = c3 < 0 ? -c3 : 0;

      r0 += k[pos] * p0 + k[pos + 4] * p1 + k[pos + 8] * p2 + k[pos + 12] * p3
        + k[neg] * n0 + k[neg + 4] * n1 + k[neg + 8] * n2 + k[neg + 12] * n3;
      r1 += k[pos + 1] * p0 + k[pos + 5] * p1 + k[pos + 9] * p2 + k[pos + 13] * p3
        + k[neg + 1] * n0 + k[neg + 5] * n1 + k[neg + 9] * n2 + k[neg + 13] * n3;
      r2 += k[pos + 2] * p0 + k[pos + 6] * p1 + k[pos + 10] * p2 + k[pos + 14] * p3
        + k[neg + 2] * n0 + k[neg + 6] * n1 + k[neg + 10] * n2 + k[neg + 14] * n3;
      r3 += k[pos + 3] * p0 + k[pos + 7] * p1 + k[pos + 11] * p2 + k[pos + 15] * p3
        + k[neg + 3] * n0 + k[neg + 7] * n1 + k[neg + 11] * n2 + k[neg + 15] * n3;
    }

    out[p] = r0 + b[0];
    out[p + 1] = r1 + b[1];
    out[p + 2] = r2 + b[2];
    out[p + 3] = r3 + b[3];
  }
}

/**
 * 1x1 conv over seven vec4 buffers (Large tail). With `first` (mode 0) the
 * seven stacked branch-A buffers use kernels[4i] / kernels[4i+2]; otherwise
 * branch-B buffers use kernels[4i+1] / kernels[4i+3].
 */
function conv112x4(
  inBufs: Float32Array[],
  k: Float32Array,
  first: boolean,
  out: Float32Array,
  w: number,
  h: number
): void {
  const n = w * h;
  for (let i = 0; i < n; i++) {
    let r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    const p = i * 4;

    for (let bi = 0; bi < 7; bi++) {
      const c0 = inBufs[bi][p], c1 = inBufs[bi][p + 1], c2 = inBufs[bi][p + 2], c3 = inBufs[bi][p + 3];
      const pos = bi * 64 + (first ? 0 : 16);
      const neg = pos + 32;

      const p0 = c0 > 0 ? c0 : 0, n0 = c0 < 0 ? -c0 : 0;
      const p1 = c1 > 0 ? c1 : 0, n1 = c1 < 0 ? -c1 : 0;
      const p2 = c2 > 0 ? c2 : 0, n2 = c2 < 0 ? -c2 : 0;
      const p3 = c3 > 0 ? c3 : 0, n3 = c3 < 0 ? -c3 : 0;

      r0 += k[pos] * p0 + k[pos + 4] * p1 + k[pos + 8] * p2 + k[pos + 12] * p3
        + k[neg] * n0 + k[neg + 4] * n1 + k[neg + 8] * n2 + k[neg + 12] * n3;
      r1 += k[pos + 1] * p0 + k[pos + 5] * p1 + k[pos + 9] * p2 + k[pos + 13] * p3
        + k[neg + 1] * n0 + k[neg + 5] * n1 + k[neg + 9] * n2 + k[neg + 13] * n3;
      r2 += k[pos + 2] * p0 + k[pos + 6] * p1 + k[pos + 10] * p2 + k[pos + 14] * p3
        + k[neg + 2] * n0 + k[neg + 6] * n1 + k[neg + 10] * n2 + k[neg + 14] * n3;
      r3 += k[pos + 3] * p0 + k[pos + 7] * p1 + k[pos + 11] * p2 + k[pos + 15] * p3
        + k[neg + 3] * n0 + k[neg + 7] * n1 + k[neg + 11] * n2 + k[neg + 15] * n3;
    }

    out[p] = r0;
    out[p + 1] = r1;
    out[p + 2] = r2;
    out[p + 3] = r3;
  }
}

class CpuWebSR implements UpscalerLike {
  private resolution: ResolutionLike;
  private canvas: OffscreenCanvas | HTMLCanvasElement;
  private weights: any;
  private arch: NetworkArch = 'm';

  // Per-frame scratch, sized once per resolution.
  private inBytes: Uint8ClampedArray<ArrayBuffer> = new Uint8ClampedArray(0);
  private inF: Float32Array<ArrayBuffer> = new Float32Array(0);
  private outU8: Uint8ClampedArray<ArrayBuffer> = new Uint8ClampedArray(0);
  private fBuffers: Record<string, Float32Array> = {};

  private kernels: Record<string, Float32Array> = {};
  private biases: Record<string, Float32Array> = {};

  private lastBitmap: ImageBitmap | null = null;

  // Pixel reader: draw each frame into a tiny 2D canvas, then read raw RGBA.
  private reader: OffscreenCanvas;
  private readerCtx: OffscreenCanvasRenderingContext2D;

  constructor(params: {
    network_name: string;
    weights: any;
    resolution: ResolutionLike;
    canvas: OffscreenCanvas | HTMLCanvasElement;
  }) {
    this.resolution = { ...params.resolution };
    this.canvas = params.canvas;
    this.weights = params.weights;
    this.arch = archFromName(params.network_name);

    this.reader = new OffscreenCanvas(this.resolution.width, this.resolution.height);
    const ctx = this.reader.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      throw new Error('CPU upscaler needs a 2D canvas for reading frame pixels');
    }
    this.readerCtx = ctx as OffscreenCanvasRenderingContext2D;

    this.allocateBuffers();
  }

  private allocateBuffers(): void {
    const w = this.resolution.width;
    const h = this.resolution.height;
    const n = w * h * 4;

    this.inBytes = new Uint8ClampedArray(n);
    this.inF = new Float32Array(n);
    this.outU8 = new Uint8ClampedArray(w * 2 * h * 2 * 4);

    const layerNames = Object.keys(this.weights.layers).filter(
      (k) => k !== 'pixel_shuffle'
    );

    for (const name of layerNames) {
      this.kernels[name] = new Float32Array(this.weights.layers[name].weights || []);
      this.biases[name] = new Float32Array(this.weights.layers[name].bias || []);
    }

    const bufferNames = this.bufferNamesFor();
    for (const name of bufferNames) {
      this.fBuffers[name] = new Float32Array(n);
    }
    // The Large net reuses two scratch buffers for its intermediate halves.
    this.fBuffers['_pt1'] = new Float32Array(n);
    this.fBuffers['_pt2'] = new Float32Array(n);
  }

  private bufferNamesFor(): string[] {
    if (this.arch === 's') {
      return ['conv2d_tf', 'conv2d_1_tf', 'conv2d_2_tf', 'conv2d_last_tf'];
    }
    if (this.arch === 'l') {
      const names: string[] = ['conv2d_tf', 'conv2d_tf1'];
      for (let i = 1; i <= 6; i++) {
        names.push(`conv2d_${i}_tf`, `conv2d_${i}_tf1`);
      }
      names.push('conv2d_last_tf', 'conv2d_last_tf1', 'conv2d_last_tf2');
      return names;
    }
    const names: string[] = ['conv2d_tf'];
    for (let i = 1; i <= 6; i++) names.push(`conv2d_${i}_tf`);
    names.push('conv2d_7_tf', 'conv2d_7_tf1', 'conv2d_7_tf2');
    return names;
  }

  private forwardImage(): void {
    const w = this.resolution.width;
    const h = this.resolution.height;
    const k = this.kernels;
    const b = this.biases;
    const buf = this.fBuffers;

    if (this.arch === 's') {
      conv3x4(this.inF, k['conv2d_tf'], b['conv2d_tf'], buf['conv2d_tf'], w, h);
      conv8x4(buf['conv2d_tf'], k['conv2d_1_tf'], b['conv2d_1_tf'], buf['conv2d_1_tf'], w, h);
      conv8x4(buf['conv2d_1_tf'], k['conv2d_2_tf'], b['conv2d_2_tf'], buf['conv2d_2_tf'], w, h);
      conv8x4(buf['conv2d_2_tf'], k['conv2d_last_tf'], b['conv2d_last_tf'], buf['conv2d_last_tf'], w, h);
      this.display1(buf['conv2d_last_tf']);
    } else if (this.arch === 'm') {
      conv3x4(this.inF, k['conv2d_tf'], b['conv2d_tf'], buf['conv2d_tf'], w, h);
      conv8x4(buf['conv2d_tf'], k['conv2d_1_tf'], b['conv2d_1_tf'], buf['conv2d_1_tf'], w, h);
      conv8x4(buf['conv2d_1_tf'], k['conv2d_2_tf'], b['conv2d_2_tf'], buf['conv2d_2_tf'], w, h);
      conv8x4(buf['conv2d_2_tf'], k['conv2d_3_tf'], b['conv2d_3_tf'], buf['conv2d_3_tf'], w, h);
      conv8x4(buf['conv2d_3_tf'], k['conv2d_4_tf'], b['conv2d_4_tf'], buf['conv2d_4_tf'], w, h);
      conv8x4(buf['conv2d_4_tf'], k['conv2d_5_tf'], b['conv2d_5_tf'], buf['conv2d_5_tf'], w, h);
      conv8x4(buf['conv2d_5_tf'], k['conv2d_6_tf'], b['conv2d_6_tf'], buf['conv2d_6_tf'], w, h);

      const mid = [
        buf['conv2d_tf'], buf['conv2d_1_tf'], buf['conv2d_2_tf'],
        buf['conv2d_3_tf'], buf['conv2d_4_tf'], buf['conv2d_5_tf'], buf['conv2d_6_tf'],
      ];
      conv56x4(mid, k['conv2d_7_tf'], b['conv2d_7_tf'], buf['conv2d_7_tf'], w, h);
      conv56x4(mid, k['conv2d_7_tf1'], b['conv2d_7_tf1'], buf['conv2d_7_tf1'], w, h);
      conv56x4(mid, k['conv2d_7_tf2'], b['conv2d_7_tf2'], buf['conv2d_7_tf2'], w, h);

      this.display3(
        buf['conv2d_7_tf'], buf['conv2d_7_tf1'], buf['conv2d_7_tf2']
      );
    } else {
      const branchA: Float32Array[] = [];
      const branchB: Float32Array[] = [];

      conv3x4(this.inF, k['conv2d_tf'], b['conv2d_tf'], buf['conv2d_tf'], w, h);
      conv3x4(this.inF, k['conv2d_tf1'], b['conv2d_tf1'], buf['conv2d_tf1'], w, h);
      branchA.push(buf['conv2d_tf']);
      branchB.push(buf['conv2d_tf1']);

      for (let i = 1; i <= 6; i++) {
        const srcA = i === 1 ? 'conv2d_tf' : `conv2d_${i - 1}_tf`;
        const srcB = srcA + '1';
        const dstA = `conv2d_${i}_tf`;
        const dstB = `conv2d_${i}_tf1`;

        conv16x4(buf[srcA], buf[srcB], k[dstA], b[dstA], buf[dstA], w, h);
        conv16x4(buf[srcA], buf[srcB], k[dstB], b[dstB], buf[dstB], w, h);
        branchA.push(buf[dstA]);
        branchB.push(buf[dstB]);
      }

      for (let c = 0; c < 3; c++) {
        const dest = c === 0 ? 'conv2d_last_tf' : `conv2d_last_tf${c}`;

        conv112x4(branchA, k[dest], true, buf['_pt1'], w, h);
        conv112x4(branchB, k[dest], false, buf['_pt2'], w, h);

        const bb = b[dest];
        const out = buf[dest];
        const pt1 = buf['_pt1'];
        const pt2 = buf['_pt2'];
        const n = w * h * 4;
        for (let i = 0; i < n; i += 4) {
          out[i] = pt1[i] + pt2[i] + bb[0];
          out[i + 1] = pt1[i + 1] + pt2[i + 1] + bb[1];
          out[i + 2] = pt1[i + 2] + pt2[i + 2] + bb[2];
          out[i + 3] = pt1[i + 3] + pt2[i + 3] + bb[3];
        }
      }

      this.display3(
        buf['conv2d_last_tf'], buf['conv2d_last_tf1'], buf['conv2d_last_tf2']
      );
    }
  }

  /** Small net: residual is a single scalar added to every channel. */
  private display1(buf: Float32Array): void {
    const w = this.resolution.width;
    const h = this.resolution.height;
    const img = this.inF;
    const out = this.outU8;

    for (let oy = 0; oy < h * 2; oy++) {
      const sy = h - 0.75 - oy / 2;
      const y2 = oy >> 1;
      const cy = oy & 1;
      const outRow = oy * w * 2 * 4;

      for (let ox = 0; ox < w * 2; ox++) {
        const sx = ox / 2 - 0.25;
        const x2 = ox >> 1;
        const cx = ox & 1;

        const v = buf[(y2 * w + x2) * 4 + (cx + cy * 2)];

        const r = sampleBilinear(img, w, h, sx, sy, 0) + v;
        const g = sampleBilinear(img, w, h, sx, sy, 1) + v;
        const bb = sampleBilinear(img, w, h, sx, sy, 2) + v;
        const a = sampleBilinear(img, w, h, sx, sy, 3) + v;

        const o = outRow + ox * 4;
        out[o] = clampByte(r);
        out[o + 1] = clampByte(g);
        out[o + 2] = clampByte(bb);
        out[o + 3] = clampByte(a);
      }
    }
  }

  /** Medium/Large nets: three residual channels, alpha mirrors the blue one. */
  private display3(buf0: Float32Array, buf1: Float32Array, buf2: Float32Array): void {
    const w = this.resolution.width;
    const h = this.resolution.height;
    const img = this.inF;
    const out = this.outU8;

    for (let oy = 0; oy < h * 2; oy++) {
      const sy = h - 0.75 - oy / 2;
      const y2 = oy >> 1;
      const cy = oy & 1;
      const outRow = oy * w * 2 * 4;

      for (let ox = 0; ox < w * 2; ox++) {
        const sx = ox / 2 - 0.25;
        const x2 = ox >> 1;
        const cx = ox & 1;

        const i = (y2 * w + x2) * 4 + (cx + cy * 2);
        const v0 = buf0[i];
        const v1 = buf1[i];
        const v2 = buf2[i];

        const r = sampleBilinear(img, w, h, sx, sy, 0) + v0;
        const g = sampleBilinear(img, w, h, sx, sy, 1) + v1;
        const bb = sampleBilinear(img, w, h, sx, sy, 2) + v2;
        const a = sampleBilinear(img, w, h, sx, sy, 3) + v2;

        const o = outRow + ox * 4;
        out[o] = clampByte(r);
        out[o + 1] = clampByte(g);
        out[o + 2] = clampByte(bb);
        out[o + 3] = clampByte(a);
      }
    }
  }

  private forward(sourceBytes: Uint8ClampedArray<ArrayBuffer>): void {
    const n = this.inF.length;
    const bytes = this.inBytes;
    bytes.set(sourceBytes);
    const f = this.inF;
    for (let i = 0; i < n; i++) {
      f[i] = bytes[i] / 255;
    }
    this.forwardImage();
  }

  async render(source: UpscalerSource): Promise<void> {
    let img: ImageBitmap | CanvasImageSource = source;
    if (source instanceof VideoFrame) {
      img = await createImageBitmap(source);
    }

    const w = this.resolution.width;
    const h = this.resolution.height;
    if (this.reader.width !== w || this.reader.height !== h) {
      this.reader.width = w;
      this.reader.height = h;
    }

    this.readerCtx.drawImage(img as any, 0, 0, w, h);
    const data = this.readerCtx.getImageData(0, 0, w, h).data;
    this.forward(data);

    const outBitmap = await createImageBitmap(new ImageData(this.outU8, w * 2, h * 2));
    this.lastBitmap = outBitmap;

    // Paint the preview so the before/after slider shows the latest frame.
    const ctx = this.canvas.getContext('bitmaprenderer') as ImageBitmapRenderingContext | null;
    if (ctx) {
      const preview = await createImageBitmap(new ImageData(this.outU8, w * 2, h * 2));
      ctx.transferFromImageBitmap(preview);
    }
  }

  switchNetwork(name: string, weightData: any): void {
    this.weights = weightData;
    this.arch = archFromName(name);

    const layerNames = Object.keys(weightData.layers).filter((k) => k !== 'pixel_shuffle');
    const w = this.resolution.width;
    const h = this.resolution.height;
    const n = w * h * 4;

    this.fBuffers = {};
    this.kernels = {};
    this.biases = {};

    for (const layerName of layerNames) {
      this.kernels[layerName] = new Float32Array(weightData.layers[layerName].weights || []);
      this.biases[layerName] = new Float32Array(weightData.layers[layerName].bias || []);
    }
    for (const name of this.bufferNamesFor()) {
      this.fBuffers[name] = new Float32Array(n);
    }
    this.fBuffers['_pt1'] = new Float32Array(n);
    this.fBuffers['_pt2'] = new Float32Array(n);
  }

  setResolution(res: ResolutionLike): void {
    this.resolution = { ...res };
    this.allocateBuffers();
  }

  async toVideoFrame(timestamp: number, duration?: number): Promise<VideoFrame> {
    if (!this.lastBitmap) {
      throw new Error('CPU upscaler: no frame rendered yet');
    }
    return new VideoFrame(this.lastBitmap, {
      timestamp,
      duration,
      alpha: 'discard',
    });
  }
}

export default CpuWebSR;