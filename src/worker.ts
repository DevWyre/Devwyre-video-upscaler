import WebSR from '@websr/websr';

import type {
  WorkerRequestMessage,
  WorkerResponseMessage,
  InitData,
  NetworkData,
  Resolution
} from './types/worker-messages';

// Processors
import pipelineProcessor from './processors/pipeline-processor';

// Worker state
let gpu: any | false;
let websr: WebSR;
let upscaled_canvas: OffscreenCanvas;
let original_canvas: OffscreenCanvas;
let resolution: Resolution;
let ctx: ImageBitmapRenderingContext | null;
let pauseLock: Promise<void> | null = null;
let resolvePause: (() => void) | null = null;
let abortController: AbortController | null = null;
let currentNetworkName = "anime4k/cnn-2x-m";
let currentWeights: any = require('./weights/cnn-2x-m-rl.json');

// Default weights
const weights = require('./weights/cnn-2x-m-rl.json');

/**
 * Check if WebGPU is supported in this environment
 */
async function isSupported(): Promise<void> {
  gpu = await WebSR.initWebGPU();

  postMessage({
    cmd: 'isSupported',
    data: gpu !== false
  } satisfies WorkerResponseMessage);
}

/**
 * Initialize the worker with canvases and create WebSR instance
 */
async function init(config: InitData): Promise<void> {
  if (!gpu) {
    gpu = await WebSR.initWebGPU();
  }

  websr = new WebSR({
    network_name: "anime4k/cnn-2x-m",
    weights,
    resolution: config.resolution,
    gpu: gpu,
    canvas: config.upscaled as any // OffscreenCanvas is valid but types may be strict
  });

  resolution = config.resolution;
  upscaled_canvas = config.upscaled;
  original_canvas = config.original;

  ctx = original_canvas.getContext('bitmaprenderer');

  // WebSR must sample input at exactly the resolution it was built with
  let renderInput: any = config.bitmap;
  if (config.bitmap.width !== resolution.width || config.bitmap.height !== resolution.height) {
    renderInput = await createImageBitmap(config.bitmap, {
      resizeHeight: resolution.height,
      resizeWidth: resolution.width,
    });
  }

  await websr.render(renderInput as any);

  if (renderInput !== config.bitmap) {
    (renderInput as ImageBitmap).close();
  }

  const bitmap2 = await createImageBitmap(config.bitmap, {
    resizeHeight: resolution.height * 2,
    resizeWidth: resolution.width * 2,
  });

  if (ctx) {
    ctx.transferFromImageBitmap(bitmap2);
  }
}

/**
 * Switch to a different AI upscaling network
 */
async function switchNetwork(name: string, weightData: any, bitmap: ImageBitmap): Promise<void> {
  currentNetworkName = name;
  currentWeights = weightData;
  websr.switchNetwork(name as any, weightData);

  let renderInput: any = bitmap;
  if (bitmap.width !== resolution.width || bitmap.height !== resolution.height) {
    renderInput = await createImageBitmap(bitmap, {
      resizeHeight: resolution.height,
      resizeWidth: resolution.width,
    });
  }

  await websr.render(renderInput as any);

  if (renderInput !== bitmap) {
    (renderInput as ImageBitmap).close();
  }

  // Refresh the "before" preview at the current output resolution
  const before = await createImageBitmap(bitmap, {
    resizeHeight: resolution.height * 2,
    resizeWidth: resolution.width * 2,
  });

  if (ctx) {
    ctx.transferFromImageBitmap(before);
  }
}

/**
 * Change the output resolution: resize the canvases and rebuild WebSR for
 * the new input resolution (the canvas is target-sized = resolution x 2).
 *
 * The old WebSR instance is abandoned rather than destroyed: its
 * WebGPUContext#destroy() tears down the whole GPU device, and the canvas
 * context cannot be reconfigured afterwards. The same device is reused.
 */
async function setResolution(res: Resolution): Promise<void> {
  resolution = res;

  upscaled_canvas.width = res.width * 2;
  upscaled_canvas.height = res.height * 2;
  original_canvas.width = res.width * 2;
  original_canvas.height = res.height * 2;

  if (!gpu) {
    gpu = await WebSR.initWebGPU();
    if (!gpu) {
      postMessage({ cmd: 'error', data: 'WebGPU unavailable' } satisfies WorkerResponseMessage);
      return;
    }
  }

  websr = new WebSR({
    network_name: currentNetworkName as any,
    weights: currentWeights,
    resolution: res,
    gpu: gpu,
    canvas: upscaled_canvas as any
  });
}






// Processing functions moved to processors/

/**
 * Worker message handler with type-safe message routing
 */
self.onmessage = async function (event: MessageEvent<WorkerRequestMessage>) {
  if (!event.data.cmd) return;

  try {
    switch (event.data.cmd) {
    case 'init':
      await init(event.data.data);
      break;

    case 'isSupported':
      await isSupported();
      break;

    case 'pause':
      if (!pauseLock) {
        pauseLock = new Promise(resolve => { resolvePause = resolve; });
        postMessage({ cmd: 'paused' } satisfies WorkerResponseMessage);
      }
      break;

    case 'resume':
      if (pauseLock && resolvePause) {
        resolvePause();
        pauseLock = null;
        resolvePause = null;
        postMessage({ cmd: 'resumed' } satisfies WorkerResponseMessage);
      }
      break;

    case 'cancel':
      if (abortController) {
        // Release any pause so a cancelled pipeline can finish aborting
        if (pauseLock && resolvePause) {
          resolvePause();
          pauseLock = null;
          resolvePause = null;
        }
        abortController.abort();
        abortController = null;
      }
      postMessage({ cmd: 'cancelled' } satisfies WorkerResponseMessage);
      break;

    case 'process':

      abortController = new AbortController();

      await pipelineProcessor({
        file: event.data.file,
        outputHandle: event.data.outputHandle,
        websr,
        upscaled_canvas,
        original_canvas,
        resolution,
        getPauseLock: () => pauseLock,
        signal: abortController.signal
      });

      abortController = null;
      break;

    case 'network':
      await switchNetwork(
        event.data.data.name,
        event.data.data.weights,
        event.data.data.bitmap
      );
      break;

    case 'resolution':
      await setResolution(event.data.data);
      break;
    }
  } catch (e: any) {
    // Never let an error escape as an unhandled rejection in the worker.
    // The pipeline processor already reports progress/finished/cancelled, so
    // this only fires for genuinely unexpected failures.
    postMessage({
      cmd: 'error',
      data: e?.message ? `Unexpected worker error: ${e.message}` : `Unexpected worker error: ${e}`
    } satisfies WorkerResponseMessage);
  }
};
