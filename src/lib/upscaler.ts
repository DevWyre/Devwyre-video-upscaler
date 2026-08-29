import WebSR from '@websr/websr';

/**
 * Abstraction over the two upscaling backends (WebGPU and CPU) so the worker
 * and the video pipeline never have to care which one is active.
 *
 * - WebGPU: `@websr/websr` drives hand-written WGSL compute shaders on the GPU
 *   (fast, but only on Chromium-based browsers and Safari 18+).
 * - CPU: a pure JavaScript reimplementation of the same Anime4K CNN networks
 *   (src/lib/cpu-upscaler.ts). It works in every browser that supports
 *   WebCodecs, at the cost of much slower throughput.
 */

export interface ResolutionLike {
  width: number;
  height: number;
}

export type UpscalerBackend = 'webgpu' | 'cpu';

export type UpscalerSource = ImageBitmap | VideoFrame | CanvasImageSource;

export interface UpscalerLike {
  /**
   * Upscale a single frame. The result is painted to the upscaled canvas so
   * the before/after comparison always shows the latest frame, and is cached
   * for `toVideoFrame` (used by the encoding pipeline).
   */
  render(source: UpscalerSource): Promise<void>;

  /** Swap the active AI network (and its weights). */
  switchNetwork(name: string, weightData: any): void;

  /**
   * Build an encoded `VideoFrame` for the frame most recently rendered.
   * Returns the frame still in its native high-res form so the encoder sees
   * exactly the same pixels that were shown in the preview.
   */
  toVideoFrame(timestamp: number, duration?: number): Promise<VideoFrame>;
}

/**
 * Thin wrapper that adapts WebSR's canvas-based rendering to the shared
 * `UpscalerLike` interface used by the worker and the pipeline.
 */
export class WebGPUWebSR implements UpscalerLike {
  private websr: WebSR;
  private canvas: HTMLCanvasElement | OffscreenCanvas;

  constructor(websr: WebSR, canvas: HTMLCanvasElement | OffscreenCanvas) {
    this.websr = websr;
    this.canvas = canvas;
  }

  async render(source: UpscalerSource): Promise<void> {
    await this.websr.render(source as any);
  }

  switchNetwork(name: string, weightData: any): void {
    this.websr.switchNetwork(name as any, weightData);
  }

  async toVideoFrame(timestamp: number, duration?: number): Promise<VideoFrame> {
    return new VideoFrame(this.canvas as any, {
      timestamp,
      duration,
      alpha: 'discard',
    });
  }
}