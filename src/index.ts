import Alpine from 'alpinejs';
import ImageCompare from './lib/image-compare-viewer.min';
import WebSR from '@websr/websr';
import type { WorkerRequestMessage, WorkerResponseMessage, Resolution } from './types/worker-messages';

import 'bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import "./index.css";
import "./lib/image-compare-viewer.min.css";

const MAX_FILE_BLOB_SIZE=1900*1024*1024; //Just under 2GB, max ArrayBufferSize

// Web Worker for video processing
const worker = new Worker(new URL('./worker.ts', import.meta.url));

// Canvas and video elements
let upscaled_canvas: HTMLCanvasElement;
let original_canvas: HTMLCanvasElement;
let video: HTMLVideoElement;

// Network selection
type NetworkSize = 'small' | 'medium' | 'large';
type ContentType = 'rl' | 'an' | '3d';

let size: NetworkSize = 'medium';
let content: ContentType = 'rl';

// Output resolution selection
type ResolutionPreset = '1080' | '2k' | '4k' | '8k';
const RESOLUTION_HEIGHTS: Record<ResolutionPreset, number> = {
    '1080': 1080,
    '2k': 1440,
    '4k': 2160,
    '8k': 4320
};
let resolutionPreset: ResolutionPreset = '4k';
let outputWidth = 0;
let outputHeight = 0;

// Video data
let download_name: string;
let inputFile: File;
let outputHandle: FileSystemFileHandle | undefined;
let gpu: any;
let websr: WebSR;

// AI model weights for different network sizes and content types
type WeightsMap = {
    [K in NetworkSize]: {
        [C in ContentType]: any;
    };
};

const weights: WeightsMap = {
    'large': {
        'rl': require('./weights/cnn-2x-l-rl.json'),
        'an': require('./weights/cnn-2x-l-an.json'),
        '3d': require('./weights/cnn-2x-l-3d.json'),
    },
    'medium': {
        'rl': require('./weights/cnn-2x-m-rl.json'),
        'an': require('./weights/cnn-2x-m-an.json'),
        '3d': require('./weights/cnn-2x-m-3d.json'),
    },
    'small': {
        'rl': require('./weights/cnn-2x-s-rl.json'),
        'an': require('./weights/cnn-2x-s-an.json'),
        '3d': require('./weights/cnn-2x-s-3d.json'),
    }
};

// Network name mapping
const networks: Record<NetworkSize, { name: string }> = {
    'small': {
        name: "anime4k/cnn-2x-s",
    },
    'medium': {
        name: "anime4k/cnn-2x-m",
    },
    'large': {
        name: "anime4k/cnn-2x-l",
    }
};

// Declare global window functions for Alpine to call and File System Access API
declare global {
    interface Window {
        chooseFile: (e?: Event) => Promise<void>;
        initRecording: () => Promise<void>;
        fullScreenPreview: (e?: Event) => Promise<void>;
        switchNetworkSize: (el: HTMLInputElement) => Promise<void>;
        switchNetworkStyle: (el: HTMLInputElement) => Promise<void>;
        switchResolution: (el: HTMLInputElement) => Promise<void>;
        showSaveFilePicker: (options?: any) => Promise<FileSystemFileHandle>;
        showOpenFilePicker: (options?: any) => Promise<FileSystemFileHandle[]>;
        togglePause: () => void;
        cancelUpscaling: () => void;
        goHome: () => void;
        seekPreview: (el: HTMLInputElement, commit: boolean) => Promise<void>;
    }
}

document.addEventListener("DOMContentLoaded", index);

//===================  Initial Load ===========================

/**
 * Main initialization function called on page load
 */
async function index(): Promise<void> {
    Alpine.store('state', 'init');
    Alpine.store('eta', '--:--');
    Alpine.store('duration', 0);
    Alpine.store('preview', 0);
    Alpine.store('previewPos', '0:00');
    Alpine.store('previewLeft', '0:00');
    Alpine.store('previewTotal', '0:00');
    Alpine.store('resSizes', { '1080': '', '2k': '', '4k': '', '8k': '' });

    window.goHome = goHome;
    window.seekPreview = seekPreview;

    Alpine.start();
    document.body.style.display = "block";

    upscaled_canvas = document.getElementById("upscaled") as HTMLCanvasElement;
    original_canvas = document.getElementById('original') as HTMLCanvasElement;

    if (!("VideoEncoder" in window)) return showUnsupported("WebCodecs");

    // Quick, friendly check: WebGPU must exist on the main thread too. The
    // worker re-verifies before starting, but this lets us fail fast and
    // show a clear message on Firefox / Safari / most mobile browsers where
    // the AI models simply cannot run.
    if (!("gpu" in navigator) || !((navigator as any).gpu)) {
        return showUnsupported("WebGPU");
    }

    // Wire up the universal file-input fallback (works on all browsers
    // including those without the File System Access API and on mobile).
    document.getElementById('file-input')?.addEventListener('change', (e) => {
        const input = e.target as HTMLInputElement;
        const selected = input.files && input.files[0];
        input.value = '';
        if (selected) loadVideo(selected);
    });

    worker.postMessage({ cmd: 'isSupported' } satisfies WorkerRequestMessage);

    window.chooseFile = chooseFile;
}

/**
 * Show unsupported browser feature message
 */
function showUnsupported(text: string): void {
    Alpine.store('component', text);
    Alpine.store('state', 'unsupported');
}

/**
 * Prompt user to choose a video file. Uses the File System Access API on
 * browsers that support it (Chrome/Edge desktop), and falls back to a
 * standard <input type="file"> on every other browser and on mobile.
 */
async function chooseFile(e?: Event): Promise<void> {
    // Prefer the native picker where available
    if (window.showOpenFilePicker) {
        try {
            const [fileHandle] = await window.showOpenFilePicker({
                types: [{
                    description: 'Video Files',
                    accept: { 'video/mp4': ['.mp4'] }
                }],
                multiple: false
            });

            // Read it as a File now so the worker only ever deals with a File,
            // regardless of which picker produced it.
            const file = await fileHandle.getFile();
            await loadVideo(file);
            return;
        } catch (err) {
            // User cancelled or the picker failed — fall through to the
            // universal file input as a graceful fallback.
            console.log('File picker cancelled/failed, using fallback', err);
        }
    }

    // Universal fallback: works in every browser and on mobile
    (document.getElementById('file-input') as HTMLInputElement)?.click();
}

/**
 * Load video file (from either the native picker or the file-input fallback)
 */
async function loadVideo(file: File): Promise<void> {
    Alpine.store('state', 'loading');

    // Store the file for later processing
    inputFile = file;

    // Set up download name
    download_name = file.name.split(".")[0] + "-upscaled.mp4";
    Alpine.store('download_name', download_name);
    Alpine.store('filename', file.name);

    // Read file for preview setup
    const arrayBuffer = await file.arrayBuffer();
    await setupPreview(arrayBuffer);
}

/**
 * Set up the preview UI with before/after comparison
 */
async function setupPreview(data: ArrayBuffer): Promise<void> {
    video = document.createElement('video');

    const fileBlob = new Blob([data], { type: "video/mp4" });

    video.src = URL.createObjectURL(fileBlob);

    const imageCompare = document.getElementById('image-compare-outer') as HTMLElement;



    video.onloadeddata = async function (){



        const target = getTargetDimensions(resolutionPreset, video.videoWidth, video.videoHeight);
        outputWidth = target.width;
        outputHeight = target.height;
        Alpine.store('width', outputWidth);
        Alpine.store('height', outputHeight);
        upscaled_canvas.width = outputWidth;
        upscaled_canvas.height = outputHeight;
        original_canvas.width = outputWidth;
        original_canvas.height = outputHeight;


        imageCompare.style.height = '318px';
        imageCompare.style.width =  `${Math.round(video.videoWidth/video.videoHeight*318)}px`
        imageCompare.style.margin = 'auto';
        imageCompare.style.position = 'relative';

        // Keep the before/after preview within the viewport on small screens:
        // cap by width first (mobile landscape/portrait), else keep the
        // desktop height.
        const cap = () => {
            const padding = 40;
            const maxW = Math.max(260, window.innerWidth - padding);
            const aspect = video.videoWidth / video.videoHeight;
            let w = maxW;
            let h = w / aspect;
            if (h > window.innerHeight * 0.6) {
                h = window.innerHeight * 0.6;
                w = h * aspect;
            }
            imageCompare.style.width = `${Math.round(w)}px`;
            imageCompare.style.height = `${Math.round(h)}px`;
            position();
        };
        const position = () => {
            const containerWidth = Math.round(imageCompare.offsetWidth) || Math.round(video.videoWidth / video.videoHeight * 318);
            const containerHeight = Math.round(imageCompare.offsetHeight) || 318;
            const fullScreenButton = document.getElementById('full-screen');
            if (fullScreenButton) {
                fullScreenButton.style.left = `${imageCompare.offsetLeft + containerWidth - 20}px`;
                fullScreenButton.style.top = `${imageCompare.offsetTop + containerHeight - 20}px`;
            }
        };
        window.addEventListener('resize', cap);
        (window as any).__imageCompareCap = cap;
        (window as any).__imageComparePosition = position;


        new ImageCompare(document.getElementById('image-compare')).mount();
        video.currentTime = video.duration * 0.2 || 0;
        if(video.requestVideoFrameCallback)  video.requestVideoFrameCallback(showPreview);
        else requestAnimationFrame(showPreview);

        window.togglePause = function () {
            const currentState = Alpine.store('state');
            if (currentState === 'processing') {
                worker.postMessage({ cmd: 'pause' } satisfies WorkerRequestMessage);
            } else if (currentState === 'paused') {
                worker.postMessage({ cmd: 'resume' } satisfies WorkerRequestMessage);
            }
        };

    }




    async function showPreview(){

        window.initRecording = initRecording;
        window.fullScreenPreview = fullScreenPreview;

        const bitmap = await createImageBitmap(video);

        // Freeze the preview frame so the before/after comparison (and any
        // network or resolution switch) always stays on the same moment.
        video.pause();

        // Duration + estimated output size for every resolution preset
        const duration = video.duration || 0;
        Alpine.store('duration', duration);
        Alpine.store('previewTotal', formatTime(duration));
        Alpine.store('previewPos', formatTime(video.currentTime));
        Alpine.store('previewLeft', formatTime(Math.max(0, duration - video.currentTime)));
        Alpine.store('preview', video.currentTime);

        const sizeMap: Record<string, string> = {};
        const presets: ResolutionPreset[] = ['1080', '2k', '4k', '8k'];
        for (const p of presets) {
            const dims = getTargetDimensions(p, video.videoWidth, video.videoHeight);
            const rate = 5e6 * Math.sqrt((dims.width * dims.height) / (1280 * 720));
            const est = (rate / 8) * duration + (128 / 8) * duration;
            sizeMap[p] = humanFileSize(est, false, 0);
        }
        Alpine.store('resSizes', sizeMap);


        const upscaled = upscaled_canvas.transferControlToOffscreen();
        const original =    original_canvas.transferControlToOffscreen();


        worker.postMessage({cmd: "init", data: {
                bitmap,
                upscaled,
                original,
                resolution: {
                    width: outputWidth / 2,
                    height: outputHeight / 2
                }

            }}, [bitmap, upscaled, original]);


        // Default to 'rl' (real life) network
        content = 'rl';
        await updateNetwork();
        Alpine.store('style', 'rl');









        // Position full-screen button in the corner and re-run on layout
        const positionFullScreen = () => {
            const position = (window as any).__imageComparePosition;
            if (typeof position === 'function') position();
        };

        setTimeout(positionFullScreen, 20);
        setTimeout(positionFullScreen, 60);
        setTimeout(positionFullScreen, 200);





        imageCompare.addEventListener('fullscreenchange', function () {
            if(!document.fullscreenElement){
                // Reset canvas styles
                upscaled_canvas.style.width = ``;
                upscaled_canvas.style.height = ``;
                original_canvas.style.width = ``;
                original_canvas.style.height = ``;
                
                // Reset container styles to original preview dimensions
                const imageCompareOuter = document.getElementById('image-compare-outer');
                
                // Reset outer container
                imageCompareOuter.style.width = ``;
                imageCompareOuter.style.height = ``;
                imageCompareOuter.style.backgroundColor = ``;
                imageCompareOuter.style.display = ``;
                imageCompareOuter.style.justifyContent = ``;
                imageCompareOuter.style.alignItems = ``;

                // Re-apply responsive preview sizing on exit from fullscreen
                const cap = (window as any).__imageCompareCap;
                if (typeof cap === 'function') cap();
            }
        });

        updateEstimatedSize();

        const quota = (await navigator.storage.estimate()).quota;

        const estimated_size = (getBitrate()/8)*video.duration + (128/8)*video.duration; // Assume 128 kbps audio

        if(estimated_size > quota){
            return showError(`The video is too big. It would output a file of ${humanFileSize(estimated_size)} but the browser can only write files up to ${humanFileSize(quota)}`);
        }


        function canvasFullScreen(){
            // Calculate aspect ratios
            const videoAspectRatio = video.videoWidth / video.videoHeight;
            const screenAspectRatio = window.innerWidth / window.innerHeight;
            
            let displayWidth, displayHeight;

            const imageCompareOuter = document.getElementById('image-compare-outer');
            const imageCompareInner = document.getElementById('image-compare');
            
            // If video is wider than screen, fit to width (letterbox on top/bottom)
            if (videoAspectRatio > screenAspectRatio) {
                displayWidth = window.innerWidth;
                displayHeight = window.innerWidth / videoAspectRatio;
            } 
            // If video is taller than screen, fit to height (pillarbox on sides)
            else {
                displayWidth = window.innerHeight * videoAspectRatio;
                displayHeight = window.innerHeight;
            }
            
            // Style the outer container to fill screen with black background and center content
            imageCompareOuter.style.width = `${window.innerWidth}px`;
            imageCompareOuter.style.height = `${window.innerHeight}px`;
            imageCompareOuter.style.backgroundColor = 'black';
            imageCompareOuter.style.display = 'flex';
            imageCompareOuter.style.justifyContent = 'center';
            imageCompareOuter.style.alignItems = 'center';
            

            console.log("Image Compare Outer", imageCompareOuter);
            console.log("Image Compare Inner", imageCompareInner);
            // Size the inner container to maintain aspect ratio
            imageCompareInner.style.width = `${displayWidth}px`;
            imageCompareInner.style.height = `${displayHeight}px`;
            
            // Let the canvases fill their parent container
            upscaled_canvas.style.width = `${displayWidth}px`;
            upscaled_canvas.style.height = `${displayHeight}px`;
            original_canvas.style.width = `${displayWidth}px`;
            original_canvas.style.height = `${displayHeight}px`;
        }

        async function fullScreenPreview(e) {
            imageCompare.requestFullscreen();
            setTimeout(canvasFullScreen, 20);
            setTimeout(canvasFullScreen, 60);
            setTimeout(canvasFullScreen, 200);

        }


        Alpine.store('state', 'preview');




        window.switchNetworkSize = async function(el: HTMLInputElement){
            if(el.value !== size){
                size = el.value as NetworkSize;

                await updateNetwork();
            }
        }

        window.switchNetworkStyle = async function(el: HTMLInputElement){
            if(el.value !== content){
                content = el.value as ContentType;

                await updateNetwork();
            }
        }

        window.switchResolution = async function(el: HTMLInputElement){
            if(el.value !== resolutionPreset){
                resolutionPreset = el.value as ResolutionPreset;

                const target = getTargetDimensions(resolutionPreset, video.videoWidth, video.videoHeight);
                outputWidth = target.width;
                outputHeight = target.height;
                Alpine.store('width', outputWidth);
                Alpine.store('height', outputHeight);

                worker.postMessage({
                    cmd: 'resolution',
                    data: { width: outputWidth / 2, height: outputHeight / 2 }
                } satisfies WorkerRequestMessage);

                await updateNetwork();

                updateEstimatedSize();
            }
        }



    }

}


/**
 * Handle messages from the video processing worker
 */
worker.onmessage = function (event: MessageEvent<WorkerResponseMessage>) {
    if (event.data.cmd === 'isSupported') {
        const supported = event.data.data;

        if (!supported) return showUnsupported("WebGPU");

    } else if (event.data.cmd === 'progress') {
        Alpine.store('progress', event.data.data);
        if (Alpine.store('state') !== 'paused') {
            Alpine.store('state', 'processing');
        }

    } else if (event.data.cmd === 'process') {
        // Processing started

    } else if (event.data.cmd === 'error') {
        showError(event.data.data);

    } else if (event.data.cmd === 'eta') {
        Alpine.store('eta', event.data.data);

    } else if (event.data.cmd === 'finished') {
        Alpine.store('state', 'complete');
        Alpine.store('download_url', event.data.data ? window.URL.createObjectURL(event.data.data) : null);
    }
    else if (event.data.cmd === 'paused') {
        Alpine.store('state', 'paused');
    } else if (event.data.cmd === 'resumed') {
        Alpine.store('state', 'processing');
    } else if (event.data.cmd === 'cancelled') {
        Alpine.store('state', 'preview');
    }
};

/**
 * Cancel an in-progress upscaling job
 */
window.cancelUpscaling = function (): void {
    const currentState = Alpine.store('state');
    if (currentState === 'processing' || currentState === 'paused') {
        worker.postMessage({ cmd: 'cancel' } satisfies WorkerRequestMessage);
    }
};



/**
 * Format seconds as M:SS
 */
function formatTime(seconds: number): string {
    if (!isFinite(seconds) || seconds < 0) seconds = 0;
    const total = Math.round(seconds);
    const minutes = Math.floor(total / 60);
    const secs = total % 60;
    return `${minutes}:${secs < 10 ? '0' : ''}${secs}`;
}

/**
 * Wait until a pending video seek has finished
 */
function ensureSeeked(): Promise<void> {
    return new Promise<void>((resolve) => {
        if (video.seeking) {
            video.addEventListener('seeked', () => resolve(), { once: true });
        } else {
            resolve();
        }
    });
}

/**
 * Return to the app home screen (fresh page load)
 */
function goHome(): void {
    location.reload();
}

/**
 * Move the preview to a new position; re-renders the upscaled frame on commit
 */
async function seekPreview(el: HTMLInputElement, commit: boolean): Promise<void> {
    const t = parseFloat(el.value);
    if (!isFinite(t)) return;

    video.currentTime = t;
    Alpine.store('preview', t);
    Alpine.store('previewPos', formatTime(t));
    Alpine.store('previewLeft', formatTime(Math.max(0, (video.duration || 0) - t)));

    if (commit) {
        await ensureSeeked();
        await updateNetwork();
    }
}

/**
 * Switch to a different upscaling network
 */
async function updateNetwork(): Promise<void> {
    const bitmap = await createImageBitmap(video);

    worker.postMessage({
        cmd: 'network',
        data: {
            name: networks[size].name,
            bitmap,
            weights: weights[size][content]
        }
    } satisfies WorkerRequestMessage);
}

//===================  Process ===========================

/**
 * Start the video upscaling process
 */
async function initRecording(): Promise<void> {
    Alpine.store('state', 'loading');

    let bitrate = getBitrate();
    const estimated_size = (bitrate / 8) * video.duration + (128 / 8) * video.duration; // Assume 128 kbps audio

    outputHandle = undefined;

    // Max Blob size - large results streamed to disk require the File System
    // Access API (Chrome/Edge desktop). On browsers without it, fall back to
    // an in-memory blob and refuse jobs that would exceed the blob limit.
    if (estimated_size > MAX_FILE_BLOB_SIZE) {
        if (!window.showSaveFilePicker) {
            return showError(`The video is too big. The output would be about ${humanFileSize(estimated_size)}, which needs a browser with File System Access support. Please use Chrome or Edge on desktop, or choose a shorter video.`);
        }
        try {
            outputHandle = await showFilePicker();
        } catch (e) {
            console.warn("User aborted request");
            return Alpine.store('state', 'preview');
        }
    }

    worker.postMessage({
        cmd: "process",
        file: inputFile,
        outputHandle
    } satisfies WorkerRequestMessage);
}

/**
 * Display error message to user
 */
function showError(message: string): void {
    Alpine.store('state', 'error');
    Alpine.store('error', message);
}

/**
 * Calculate target output dimensions for the selected resolution preset.
 * The upscaler model outputs 2x its input, so the model input ("staging")
 * is half of the target and gets 2x'd onto a target-sized canvas.
 */
function getTargetDimensions(preset: ResolutionPreset, sourceWidth: number, sourceHeight: number): Resolution {
    const targetHeight = RESOLUTION_HEIGHTS[preset];
    const height = targetHeight + (targetHeight % 2);
    let width = Math.round((sourceWidth / sourceHeight) * targetHeight);
    if (width % 2) {
        width += 1;
    }

    return { width, height };
}

/**
 * Calculate target bitrate based on the output resolution
 */
function getBitrate(): number {
    return 5e6 * Math.sqrt((outputWidth * outputHeight) / (1280 * 720));
}

/**
 * Refresh the estimated output size and target (writer vs blob)
 */
function updateEstimatedSize(): void {
    const bitrate = getBitrate();
    const estimated_size = (bitrate / 8) * video.duration + (128 / 8) * video.duration; // Assume 128 kbps audio

    Alpine.store('size', humanFileSize(estimated_size));
    Alpine.store('target', estimated_size > MAX_FILE_BLOB_SIZE ? 'writer' : 'blob');
}

/**
 * Format bytes into human-readable file size
 */
function humanFileSize(bytes: number, si: boolean = false, dp: number = 1): string {
    const thresh = si ? 1000 : 1024;

    if (Math.abs(bytes) < thresh) {
        return bytes + ' B';
    }

    const units = si
        ? ['kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
        : ['KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB'];
    let u = -1;
    const r = 10 ** dp;

    do {
        bytes /= thresh;
        ++u;
    } while (Math.round(Math.abs(bytes) * r) / r >= thresh && u < units.length - 1);

    return bytes.toFixed(dp) + ' ' + units[u];
}

/**
 * Show native file picker for saving output video
 */
async function showFilePicker(): Promise<FileSystemFileHandle> {
    const handle = await window.showSaveFilePicker({
        startIn: 'downloads',
        suggestedName: download_name,
        types: [{
            description: 'Video File',
            accept: { 'video/mp4': ['.mp4'] }
        }],
    });

    return handle;
}












