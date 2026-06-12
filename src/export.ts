import { ArrayBufferTarget, Muxer } from 'mp4-muxer';

export interface ExportOptions {
  video: HTMLVideoElement;
  fps: number;
  outW: number;
  outH: number;
  /** Draws the output image for time t — same renderer as the preview. */
  render: (ctx: CanvasRenderingContext2D, outW: number, outH: number, t: number) => void;
  seekTo: (t: number) => Promise<void>;
  onProgress: (frac: number) => void;
}

/**
 * Deterministic frame-by-frame export: seek to every frame (no drops, no
 * realtime pressure), draw through the shared renderer, encode with
 * hardware H.264 via WebCodecs, and mux to MP4 in memory.
 */
export async function exportVideo(opts: ExportOptions): Promise<Blob> {
  const { video, fps, outW, outH } = opts;
  if (typeof VideoEncoder === 'undefined') {
    throw new Error('WebCodecs not supported in this browser');
  }

  const canvas = document.createElement('canvas');
  canvas.width = outW;
  canvas.height = outH;
  const ctx = canvas.getContext('2d')!;

  const muxer = new Muxer({
    target: new ArrayBufferTarget(),
    video: { codec: 'avc', width: outW, height: outH },
    fastStart: 'in-memory',
  });

  let encError: unknown = null;
  const encoder = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: (e) => {
      encError = e;
    },
  });
  encoder.configure({
    codec: 'avc1.64002a', // High@4.2 — covers 1080p60
    width: outW,
    height: outH,
    bitrate: 16_000_000,
    framerate: fps,
  });

  const frameDurUs = Math.round(1e6 / fps);
  const keyEvery = Math.max(1, Math.round(fps * 2));
  const total = Math.max(1, Math.floor(video.duration * fps));

  for (let k = 0; k < total; k++) {
    if (encError) throw encError;
    // Mid-frame target so the seek lands unambiguously inside frame k.
    const t = Math.min((k + 0.5) / fps, Math.max(0, video.duration - 0.001));
    await opts.seekTo(t);
    opts.render(ctx, outW, outH, video.currentTime);
    const frame = new VideoFrame(canvas, {
      timestamp: k * frameDurUs,
      duration: frameDurUs,
    });
    encoder.encode(frame, { keyFrame: k % keyEvery === 0 });
    frame.close();
    while (encoder.encodeQueueSize > 4) {
      await new Promise((r) => setTimeout(r, 2));
    }
    opts.onProgress((k + 1) / total);
  }

  await encoder.flush();
  muxer.finalize();
  if (encError) throw encError;
  return new Blob([muxer.target.buffer], { type: 'video/mp4' });
}
