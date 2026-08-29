/**
 * Type-safe worker message definitions for communication between
 * the main thread and the video processing worker.
 */

export interface Resolution {
  width: number;
  height: number;
}

// Messages sent FROM main thread TO worker
export type WorkerRequestMessage =
  | { cmd: 'isSupported' }
  | { cmd: 'init'; data: InitData }
  | { cmd: 'network'; data: NetworkData }
  | { cmd: 'resolution'; data: Resolution }
  | { cmd: 'process'; file: File; outputHandle?: FileSystemFileHandle }
  | { cmd: 'compatFrame'; data: CompatFrameData }
  | { cmd: 'pause' }
  | { cmd: 'resume' }
  | { cmd: 'cancel' };

export interface InitData {
  bitmap: ImageBitmap;
  upscaled?: OffscreenCanvas;
  original?: OffscreenCanvas;
  resolution: Resolution;
  /** When set (compatibility mode), the worker returns each rendered frame
   *  to the main thread as an ImageBitmap via a 'renderResult' message. */
  returnBitmap?: boolean;
}

export interface NetworkData {
  name: string;
  bitmap: ImageBitmap;
  weights: any; // TODO: Type this based on WebSR weight structure
  returnBitmap?: boolean;
}

export interface CompatFrameData {
  bitmap: ImageBitmap;
  width: number;
  height: number;
}

// Messages sent FROM worker TO main thread
export type WorkerResponseMessage =
  | { cmd: 'isSupported'; data: boolean; backend?: 'webgpu' | 'cpu' }
  | { cmd: 'progress'; data: number }
  | { cmd: 'eta'; data: string }
  | { cmd: 'process' }
  | { cmd: 'error'; data: string }
  | { cmd: 'finished'; data: Blob | null }
  | { cmd: 'paused' }
  | { cmd: 'resumed' }
  | { cmd: 'cancelled' }
  /** Single upscaled frame for the compatibility (no-WebCodecs) path. */
  | { cmd: 'renderResult'; bitmap: ImageBitmap }
  /** One processed frame ready to be drawn to the recording canvas. */
  | { cmd: 'compatFrameResult'; bitmap: ImageBitmap };

// Type guard helpers
export function isWorkerRequestMessage(msg: any): msg is WorkerRequestMessage {
  return msg && typeof msg.cmd === 'string';
}

export function isWorkerResponseMessage(msg: any): msg is WorkerResponseMessage {
  return msg && typeof msg.cmd === 'string';
}
