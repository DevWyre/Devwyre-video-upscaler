import WebSR from '@websr/websr';

import type {
  WorkerRequestMessage,
  WorkerResponseMessage,
  InitData,
  NetworkData,
  Resolution
} from './types/worker-messages';
import type { UpscalerBackend, UpscalerLike } from './lib/upscaler';
import { WebGPUWebSR } from './lib/upscaler';
import CpuWebSR from './lib/cpu-upscaler';

// Processors
import pipelineProcessor from './processors/pipeline-processor';

// Worker state
let gpu: any | false;
let upscaler: UpscalerLike;
let upscaled_canvas: OffscreenCanvas;
let original_canvas: OffscreenCanvas;
let resolution: Resolution;
let ctx: ImageBitmapRenderingContext | null;
let pauseLock: Promise<void> | null = null;
let resolvePause: (() => void) | null = null;
let abortController: AbortController | null = null;
let currentNetworkName = "anime4k/cnn-2x-m";
let currentWeights: any = require('./weights/cnn-2x-m-rl.json');

// Which backend to run the AI on, decided once per worker lifetime.
let backend: UpscalerBackend = 'cpu';
let backendResolved: Promise<UpscalerBackend> | null = null;

/**
 * Detect the best available backend. WebGPU is used when it exists, otherwise
 * the pure-JS CPU engine makes the app work in every other browser.
 */
function resolveBackend(): Promise<UpscalerBackend> {
  if (!backendResolved) {
    backendResolved = (async () => {
      try {
        gpu = await WebSR.initWebGPU();
        backend = gpu !== false ? 'webgpu' : 'cpu';
      } catch (e) {
        console.warn('WebGPU init failed, falling back to CPU upscaler', e);
        gpu = false;
        backend = 'cpu';
      }
      return backend;
    })();
  }
  return backendResolved;
}

/**
 * Check what's supported and report which backend will be used.
 */
async function isSupported(): Promise<void> {
  const result = await resolveBackend();

  postMessage({
    cmd: 'isSupported',
    data: true,
    backend: result
  } satisfies WorkerResponseMessage);
}

/**
 * Build the chosen upscaler for the current resolution. Falls back to the CPU
 * engine if WebGPU setup throws for any reason.
 */
function buildWebGPUUpscaler(): UpscalerLike {
  const websr = new WebSR({
    network_name: currentNetworkName as any,
    weights: currentWeights,
    resolution,
    gpu: gpu,
    canvas: upscaled_canvas as any
  });
  return new WebGPUWebSR(websr, upscaled_canvas);
}

function buildUpscaler(): UpscalerLike {
  if (backend === 'webgpu') {
    try {
      return buildWebGPUUpscaler();
    } catch (e) {
      console.warn('WebGPU upscaler failed to build, falling back to CPU', e);
      backend = 'cpu';
    }
  }
  return new CpuWebSR({
    network_name: currentNetworkName,
    weights: currentWeights,
    resolution,
    canvas: upscaled_canvas
  });
}

/**
 * Initialize the worker with canvases and create the upscaler instance.
 *
 * In compatibility mode (main thread has no WebCodecs) the main thread does
 * not transfer its canvases — the worker creates its own render surfaces and
 * returns every rendered frame back to the main thread as an ImageBitmap.
 */
async function init(config: InitData): Promise<void> {
  await resolveBackend();

  resolution = config.resolution;
  const compat = !config.upscaled;
  if (compat) backend = 'cpu';

  if (compat) {
    // Worker-owned render surfaces; the main thread only ever sees ImageBitmaps.
    upscaled_canvas = new OffscreenCanvas(resolution.width * 2, resolution.height * 2);
    original_canvas = new OffscreenCanvas(resolution.width * 2, resolution.height * 2);
  } else {
    upscaled_canvas = config.upscaled!;
    original_canvas = config.original!;
  }

  upscaler = buildUpscaler();

  ctx = original_canvas.getContext('bitmaprenderer');

  // The upscaler must sample input at exactly the resolution it was built with
  let renderInput: any = config.bitmap;
  if (config.bitmap.width !== resolution.width || config.bitmap.height !== resolution.height) {
    renderInput = await createImageBitmap(config.bitmap, {
      resizeHeight: resolution.height,
      resizeWidth: resolution.width,
    });
  }

  await upscaler!.render(renderInput as any);

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

  if (config.returnBitmap && compat) {
    const bmp = (upscaler as CpuWebSR).currentBitmap;
    if (bmp) {
      postMessage({ cmd: 'renderResult', bitmap: bmp } satisfies WorkerResponseMessage, [bmp]);
    }
  }
}

/**
 * Switch to a different AI upscaling network
 */
async function switchNetwork(name: string, weightData: any, bitmap: ImageBitmap, returnBitmap?: boolean): Promise<void> {
  currentNetworkName = name;
  currentWeights = weightData;
  upscaler!.switchNetwork(name, weightData);

  let renderInput: any = bitmap;
  if (bitmap.width !== resolution.width || bitmap.height !== resolution.height) {
    renderInput = await createImageBitmap(bitmap, {
      resizeHeight: resolution.height,
      resizeWidth: resolution.width,
    });
  }

  await upscaler!.render(renderInput as any);

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

  if (returnBitmap) {
    const bmp = (upscaler as CpuWebSR).currentBitmap;
    if (bmp) {
      postMessage({ cmd: 'renderResult', bitmap: bmp } satisfies WorkerResponseMessage, [bmp]);
    }
  }
}

/**
 * Change the output resolution: resize the canvases and rebuild the upscaler
 * for the new input resolution (the canvas is target-sized = resolution x 2).
 *
 * On the WebGPU path the old WebSR instance is abandoned rather than destroyed:
 * its WebGPUContext#destroy() tears down the whole GPU device, and the canvas
 * context cannot be reconfigured afterwards. The same device is reused.
 */
async function setResolution(res: Resolution): Promise<void> {
  if (!resolution || resolution.width !== res.width || resolution.height !== res.height) {
    resolution = res;

    upscaled_canvas.width = res.width * 2;
    upscaled_canvas.height = res.height * 2;
    original_canvas.width = res.width * 2;
    original_canvas.height = res.height * 2;
  }

  const wasCpu = backend === 'cpu';

  if (wasCpu || !gpu) {
    await resolveBackend();
    if (backend === 'cpu') {
      (upscaler as unknown as CpuWebSR).setResolution(res);
      return;
    }
  }

  upscaler = buildUpscaler();
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
        upscaler,
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
        event.data.data.bitmap,
        event.data.data.returnBitmap === true
      );
      break;

    case 'resolution':
      await setResolution(event.data.data);
      break;

    case 'compatFrame': {
      // Compatibility mode: upscale a single frame handed over from the main
      // thread and return the result as a transferable ImageBitmap.
      const data = event.data.data;
      let src: any = data.bitmap;
      if (src.width !== resolution.width || src.height !== resolution.height) {
        src = await createImageBitmap(data.bitmap, {
          resizeWidth: resolution.width,
          resizeHeight: resolution.height,
        });
      }

      await upscaler!.render(src as any);

      if (src !== data.bitmap) {
        src.close();
      }

      const bmp = (upscaler as CpuWebSR).currentBitmap;
      if (bmp) {
        postMessage({ cmd: 'compatFrameResult', bitmap: bmp } satisfies WorkerResponseMessage, [bmp]);
      }
      break;
    }
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