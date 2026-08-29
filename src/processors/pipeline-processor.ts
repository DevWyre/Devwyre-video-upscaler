import { WebDemuxer } from "web-demuxer";
import {
  Output,
  Mp4OutputFormat,
  StreamTarget,
  EncodedVideoPacketSource,
  EncodedAudioPacketSource,
  EncodedPacket,
} from 'mediabunny';
import WebSR from '@websr/websr';
import InMemoryStorage from './in-memory-storage';

interface ProcessorArgs {
  inputHandle: FileSystemFileHandle;
  outputHandle?: FileSystemFileHandle;
  websr: WebSR;
  upscaled_canvas: OffscreenCanvas;
  original_canvas: OffscreenCanvas;
  resolution: { width: number; height: number };
  getPauseLock?: () => Promise<void> | null;
  signal?: AbortSignal;
}


/**
 * Track demuxed chunks with indices for keyframe detection
 */
class DemuxerTrackingStream extends TransformStream<EncodedVideoChunk, { chunk: EncodedVideoChunk; index: number }> {
  constructor(signal?: AbortSignal) {
    let chunkIndex = 0;
    super(
      {

        async transform(chunk, controller) {
          // Apply backpressure if downstream is full
          while (!signal?.aborted && controller.desiredSize !== null && controller.desiredSize < 0) {
            await new Promise((r) => setTimeout(r, 10));
          }

          if (signal?.aborted) return;

          controller.enqueue({ chunk, index: chunkIndex++ });
        },
      },
      { highWaterMark: 20 } // Buffer up to 20 chunks
    );
  }
}

/**
 * Decode video chunks into frames with backpressure management
 */
class VideoDecoderStream extends TransformStream<
  { chunk: EncodedVideoChunk; index: number },
  { frame: VideoFrame; index: number }
> {
  constructor(config: VideoDecoderConfig, getPauseLock?: () => Promise<void> | null, signal?: AbortSignal) {
    let pendingIndices: number[] = [];
    let decoder: VideoDecoder;


    super(
      {
        start(controller) {
          decoder = new VideoDecoder({
            output: (frame) => {
              // The pipeline may already be cancelled while a frame is decoded:
              // drop it instead of pushing into a destroyed stream.
              const index = pendingIndices.shift();
              if (index === undefined) {
                frame.close();
                return;
              }
              try {
                controller.enqueue({ frame, index });
              } catch (e) {
                console.error('Decoder output dropped after cancellation:', e);
                frame.close();
              }
            },
            error: (e) => {
              console.error('Decoder error:', e);
              try {
                controller.error(e);
              } catch {
                // Stream already destroyed (e.g. pipeline cancelled)
              }
            },
          });

          decoder.configure(config);
        },

        async transform(item, controller) {
          if (signal?.aborted) return;

          if (getPauseLock) {
            const lock = getPauseLock();
            if (lock) {
              await lock;
            }
          }
          if (signal?.aborted) return;

          // Check decoder queue backpressure
          while (!signal?.aborted && decoder.decodeQueueSize >= 20) {
            await new Promise((r) => setTimeout(r, 10));
          }

          // Check downstream backpressure
          while (!signal?.aborted && controller.desiredSize !== null && controller.desiredSize < 0) {
            await new Promise((r) => setTimeout(r, 10));
          }

          if (signal?.aborted) return;

          pendingIndices.push(item.index);
          decoder.decode(item.chunk);
        },

        async flush(controller) {
          await decoder.flush();
          try {
            decoder.close();
          } catch (e) {
            console.error('Error closing decoder:', e);
          }
        },
      },
      { highWaterMark: 10 }
    );
  }
}

/**
 * Upscale frames using WebSR and render "before" preview
 */
class VideoUpscaleStream extends TransformStream<
  { frame: VideoFrame; index: number },
  { frame: VideoFrame; index: number }
> {
  constructor(
    private websr: WebSR,
    private upscaled_canvas: OffscreenCanvas,
    private original_canvas: OffscreenCanvas,
    getPauseLock?: () => Promise<void> | null,
    private signal?: AbortSignal,
    private resolution: { width: number; height: number } = { width: 0, height: 0 }
  ) {
    super(
      {

        async transform(item, controller) {
          if (signal?.aborted) {
            item.frame.close();
            return;
          }

          if (getPauseLock) {
            const lock = getPauseLock();
            if (lock) {
              await lock;
            }
          }
          if (signal?.aborted) {
            item.frame.close();
            return;
          }

          const { frame, index } = item;

          // Create "before" preview resized to the output resolution
          const beforeBitmap = await createImageBitmap(frame, {
            resizeHeight: resolution.height * 2,
            resizeWidth: resolution.width * 2
          });

          // WebSR samples its input at exactly its configured resolution,
          // so resize the frame to match when a fixed output was chosen.
          let renderInput: any = frame;
          if (frame.displayWidth !== resolution.width || frame.displayHeight !== resolution.height) {
            renderInput = await createImageBitmap(frame, {
              resizeHeight: resolution.height,
              resizeWidth: resolution.width
            });
          }

          // Render upscaled frame to canvas
          await websr.render(renderInput as any);

          if (renderInput !== frame) {
            (renderInput as ImageBitmap).close();
          }

          // Update "before" preview canvas
          const ctx = original_canvas.getContext('bitmaprenderer');
          if (ctx) {
            ctx.transferFromImageBitmap(beforeBitmap);
          }

          // Create upscaled VideoFrame from canvas
          const upscaledFrame = new VideoFrame(upscaled_canvas, {
            timestamp: frame.timestamp,
            duration: frame.duration,
            alpha: "discard"
          });

          // Clean up original frame
          frame.close();

          if (signal?.aborted) {
            upscaledFrame.close();
            return;
          }

          try {
            controller.enqueue({ frame: upscaledFrame, index });
          } catch (e) {
            console.error('Upscaled frame dropped after cancellation:', e);
            upscaledFrame.close();
          }
        },
      },
      { highWaterMark: 5 } // Keep small - frames are large
    );
  }
}

/**
 * Encode upscaled frames with backpressure management
 */
class VideoEncoderStream extends TransformStream<
  { frame: VideoFrame; index: number },
  { chunk: EncodedVideoChunk; meta: EncodedVideoChunkMetadata }
> {
  constructor(config: VideoEncoderConfig, signal?: AbortSignal) {
    let encoder: VideoEncoder;
    super(
      {
        start(controller) {
          encoder = new VideoEncoder({
            output: (chunk, meta) => {
              // The pipeline may already be cancelled while a chunk is encoded:
              // drop it instead of pushing into a destroyed stream.
              try {
                controller.enqueue({ chunk, meta });
              } catch (e) {
                console.error('Encoder output dropped after cancellation:', e);
              }
            },
            error: (e) => {
              console.error('Encoder error:', e);
              try {
                controller.error(e);
              } catch {
                // Stream already destroyed (e.g. pipeline cancelled)
              }
            },
          });

          encoder.configure(config);
        },

        async transform(item, controller) {
          if (signal?.aborted) {
            item.frame.close();
            return;
          }

          // Check encoder queue backpressure
          while (!signal?.aborted && encoder.encodeQueueSize >= 20) {
            await new Promise((r) => setTimeout(r, 10));
          }

          // Check downstream backpressure
          while (!signal?.aborted && controller.desiredSize !== null && controller.desiredSize < 0) {
            await new Promise((r) => setTimeout(r, 10));
          }

          if (signal?.aborted) {
            item.frame.close();
            return;
          }

          // Encode with keyframe every 60 frames
          encoder.encode(item.frame, { keyFrame: item.index % 60 === 0 });
          item.frame.close();
        },

        async flush(controller) {
          await encoder.flush();
          try {
            encoder.close();
          } catch (e) {
            console.error('Error closing encoder:', e);
          }
        },
      },
      { highWaterMark: 10 }
    );
  }
}

/**
 * Create WritableStream for video chunks with progress reporting
 */
function createVideoMuxerWriter(
  videoSource: EncodedVideoPacketSource,
  duration: number
) {
  const startTime = performance.now();
  let frameCount = 0;

  return new WritableStream<{ chunk: EncodedVideoChunk; meta: EncodedVideoChunkMetadata }>({
    async write(value) {
      try {
        await videoSource.add(EncodedPacket.fromEncodedChunk(value.chunk), value.meta);
      } catch (e) {
        console.error('Video muxer writer error:', e);
        throw e;
      }
      frameCount++;

      // Report progress every 30 frames
      if (frameCount % 30 === 0) {
        const elapsed = performance.now() - startTime;
        const progress = Math.floor((value.chunk.timestamp / 1000000) / duration * 100);

        postMessage({ cmd: 'progress', data: progress });

        if (progress > 0 && elapsed > 0) {
          const processingRate = progress / elapsed;
          const secondsLeft = ((100 - progress) / processingRate) / 1000;
          const eta = Math.max(1, Math.round(secondsLeft));
          postMessage({ cmd: 'eta', data: prettyTime(eta) });
        }
      }
    },

    close() {
      console.log('All video frames written to muxer');
    },

    abort(reason) {
      console.error('Video muxer writer aborted:', reason);
    }
  });
}

/**
 * Create WritableStream for audio chunks (passthrough)
 */
function createAudioMuxerWriter(
  audioSource: EncodedAudioPacketSource,
  audioConfig: AudioDecoderConfig
) {
  let configWritten = false;

  return new WritableStream<EncodedAudioChunk>({
    async write(chunk) {
      if (chunk.timestamp >= 0) {
        const config = configWritten ? undefined : { decoderConfig: audioConfig };
        configWritten = true;
        await audioSource.add(EncodedPacket.fromEncodedChunk(chunk), config);
      }
    },

    close() {
      console.log('All audio chunks written to muxer');
    },

    abort(reason) {
      console.error('Audio muxer writer aborted:', reason);
    }
  });
}

/**
 * Format seconds into HH:MM:SS
 */
function prettyTime(secs: number): string {
  const sec_num = parseInt(secs.toString(), 10);
  const hours = Math.floor(sec_num / 3600);
  const minutes = Math.floor(sec_num / 60) % 60;
  const seconds = sec_num % 60;

  return [hours, minutes, seconds]
    .map(v => v < 10 ? "0" + v : v)
    .filter((v, i) => v !== "00" || i > 0)
    .join(":");
}

/**
 * Main pipeline processor using Streams API
 */
export default async function pipelineProcessor(args: ProcessorArgs): Promise<void> {
  const { inputHandle, outputHandle, websr, upscaled_canvas, original_canvas, resolution, getPauseLock, signal } = args;

  console.log('Starting pipeline processor with Streams API');

  // Get file from handle
  const file = await inputHandle.getFile();

  // Initialize demuxer
  const demuxer = new WebDemuxer({
    wasmFilePath: "https://cdn.jsdelivr.net/npm/web-demuxer@latest/dist/wasm-files/web-demuxer.wasm",
  });

  await demuxer.load(file);

  // Get media info
  const mediaInfo = await demuxer.getMediaInfo();
  const videoTrack = mediaInfo.streams.find((s: any) => s.codec_type_string === 'video');
  const audioTrack = mediaInfo.streams.find((s: any) => s.codec_type_string === 'audio');

  if (!videoTrack) {
    return postMessage({ cmd: 'error', data: 'No video track found' });
  }

  const videoDecoderConfig = await demuxer.getDecoderConfig('video');
  const audioConfig = audioTrack ? await demuxer.getDecoderConfig('audio') : null;

  const duration = videoTrack.duration;
  const width = resolution.width;
  const height = resolution.height;

  // Set up MediaBunny output
  let target: StreamTarget;
  let writer: FileSystemWritableFileStream | undefined;
  let storage: InMemoryStorage | undefined;

  if (outputHandle) {
    writer = await outputHandle.createWritable();
    target = new StreamTarget(writer);
  } else {
    storage = new InMemoryStorage();
    const writableStream = new WritableStream({
      write(chunk) {
        storage!.write(chunk.data, chunk.position);
      }
    });
    target = new StreamTarget(writableStream);
  }

  const output = new Output({
    format: new Mp4OutputFormat(),
    target,
  });

  // Parse framerate from demuxer (e.g. "30/1" or "24000/1001"), fall back to 30
  const [fpsNum, fpsDen] = (videoTrack.r_frame_rate || '30/1').split('/').map(Number);
  const framerate = (fpsNum && fpsDen) ? fpsNum / fpsDen : 30;

  // Configure encoder
  const bitrate = 2.5e6 * (width * height * 4) / (1280 * 720);

  const videoEncoderConfig: VideoEncoderConfig = {
    codec: 'avc1.4d0034',
    width: width * 2,
    height: height * 2,
    bitrate: Math.round(bitrate),
    framerate: framerate,
  };

  const videoSource = new EncodedVideoPacketSource('avc');
  output.addVideoTrack(videoSource);

  let audioSource: EncodedAudioPacketSource | undefined;
  if (audioConfig) {
    audioSource = new EncodedAudioPacketSource('aac');
    output.addAudioTrack(audioSource);
  }

  // Build the pipeline!
  const chunkStream = demuxer.read('video', 0) as ReadableStream<EncodedVideoChunk>;

  const videoWriter = createVideoMuxerWriter(videoSource, duration);

  const pipeline = chunkStream
    .pipeThrough(new DemuxerTrackingStream(signal))
    .pipeThrough(new VideoDecoderStream(videoDecoderConfig, getPauseLock, signal))
    .pipeThrough(new VideoUpscaleStream(websr, upscaled_canvas, original_canvas, getPauseLock, signal, resolution))
    .pipeThrough(new VideoEncoderStream(videoEncoderConfig, signal))
    .pipeTo(videoWriter, { signal: signal });

  await output.start();

  // Process video
  try {
    await pipeline;
  } catch (e: any) {
    if (e?.name === 'AbortError' || signal?.aborted) {
      console.log('Pipeline cancelled');
      // Release the file-writer lock so a new job can start immediately
      if (writer) {
        try {
          await writer.abort();
        } catch {
          // Ignore: nothing more we can do during cancellation
        }
      }
      return;
    }
    throw e;
  }

  // Process audio (passthrough)
  if (audioConfig && audioSource) {
    console.log('Processing audio...');
    const audioStream = demuxer.read('audio', 0) as ReadableStream<EncodedAudioChunk>;
    const audioWriter = createAudioMuxerWriter(audioSource, audioConfig);
    await audioStream.pipeTo(audioWriter);
  }

  // Finalize
  await output.finalize();

  if (writer) {
    await writer.close();
    postMessage({ cmd: 'finished', data: null }, []);
  } else {
    const blob = storage!.toBlob('video/mp4');
    postMessage({ cmd: 'finished', data: blob });
  }

  console.log('Pipeline processing complete!');
}
