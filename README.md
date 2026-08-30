# Devwyre Video Upscaler

A free, browser-based AI video upscaler by **Devwyre**. No uploads, no signups — everything runs entirely on your device using WebGPU.

**Open source** — this project is free and open source. Star it and follow development on GitHub: [github.com/DevWyre/Devwyre-video-upscaler](https://github.com/DevWyre/Devwyre-video-upscaler)

**Support us** — if you find this tool useful, consider [making a donation](https://flutterwave.com/store/devwyredonations/vm88qusbzwfm) to help keep it free and continue development.

## What It Does

Pick an MP4, choose your settings, and let AI upscale your video right in the browser. The app uses Devwyre's own convolutional neural networks, accelerated by hand-written WebGPU compute shaders, to double your video's resolution in real time.

## Features

- **100% local processing** — nothing leaves your machine. No servers, no cloud.
- **GPU-accelerated AI** — CNN super-resolution models run on your graphics card via WebGPU.
- **Before/after comparison** — drag slider to preview the upscale on any frame.
- **Multiple AI models** — choose from Fast, Balanced, or Ultra quality presets.
- **Content-aware weights** — dedicated models for real-life footage, anime, and 3D renders.
- **Resolution presets** — output to 1080p, 2K, 4K, or 8K.
- **Audio preserved** — the original audio track is carried through untouched.
- **Pause, resume, cancel** — full control over long processing jobs.
- **Live progress with ETA** — see exactly how long is left.
- **Up to 2 GB input files** — handles large videos.
- **Stream or download** — save directly to disk for large files, or download as a blob.

## How It Works

1. Load an MP4 file.
2. Pick an AI model (Fast / Balanced / Ultra) and output resolution.
3. Preview the before/after on any frame with the comparison slider.
4. Hit process — the app demuxes, decodes, upscales each frame with the CNN, re-encodes, and muxes the audio back.
5. Download your upscaled video.

## Tech Stack

- **WebGPU** — GPU compute for Devwyre's neural network inference
- **WebCodecs** — hardware-accelerated video decode/encode
- **Streams API** — streaming pipeline with backpressure for memory efficiency
- **File System Access API** — direct read/write to disk for large files
- **Alpine.js** + **Bootstrap** + **Tailwind CSS** — UI layer
- **TypeScript** + **Webpack** — build tooling

## Requirements

Requires a modern browser with WebGPU support:

- **Google Chrome** or **Microsoft Edge** (latest version recommended)
- A desktop computer with a compatible GPU

## Development

```bash
npm install
npm run serve    # dev server at http://localhost:8080
npm run build    # build into ./dist
npm run type-check    # run TypeScript type checking
```

## License

MIT — see LICENSE file for details.