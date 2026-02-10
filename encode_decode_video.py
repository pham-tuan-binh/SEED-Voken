#!/usr/bin/env python3
"""
Load a video, encode it with the video VQ model, decode, and save the reconstructed video.
Uses the checkpoint "Video 128 262144.ckpt" and processes in temporal windows of 17 frames.
"""
import os
import sys
import argparse
import time

sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from omegaconf import OmegaConf

from src.Open_MAGVIT2.models.video_lfqgan import VQModel
import src.Open_MAGVIT2.data.video_transforms as video_transforms
import src.Open_MAGVIT2.data.volume_transforms as volume_transforms

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None

def _get_device():
    # Video model: NPU > CUDA > CPU only (no MPS — encode is much slower on MPS fallback).
    if hasattr(torch, "npu"):
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                return torch.device("npu:0")
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

DEVICE = _get_device()

RESOLUTION = 128
TEMPORAL_WINDOW = 17


def load_model(config_path: str, ckpt_path: str):
    config = OmegaConf.load(config_path)
    # Only model params; drop training-only keys if present
    init_args = dict(config.model.init_args)
    for key in ("image_pretrain_path", "sche_type", "wpe", "wp", "wp0", "max_iter", "wp_iter", "resume_lr"):
        init_args.pop(key, None)
    model = VQModel(**init_args)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.eval().to(DEVICE)


def load_video_frames(path: str, max_frames: int = None):
    if VideoReader is None:
        raise ImportError("decord is required. Install with: uv add eva-decord")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    try:
        vr = VideoReader(path, ctx=cpu(0))
    except Exception as e:
        raise RuntimeError(f"decord could not open video {path}. Try another format (e.g. mp4). Error: {e}") from e
    n = len(vr)
    if n == 0:
        raise RuntimeError(f"Video has no frames: {path}")
    fps = vr.get_avg_fps()
    if max_frames is not None and n > max_frames:
        indices = np.linspace(0, n - 1, max_frames, dtype=int)
    else:
        indices = np.arange(n)
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8
    return frames, fps


def preprocess_frames(frames: np.ndarray, resolution: int = RESOLUTION) -> torch.Tensor:
    """(T, H, W, C) uint8 -> (1, C, T, H, W) float in [-1, 1], resized and center-cropped."""
    transforms = video_transforms.Compose([
        video_transforms.Resize(resolution, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(resolution, resolution)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    out = transforms(frames)  # (C, T, H, W)
    return out.unsqueeze(0).to(DEVICE)  # (1, C, T, H, W)


def tensor_to_frames(x: torch.Tensor) -> np.ndarray:
    """(1, C, T, H, W) in [-1, 1] -> (T, H, W, C) uint8."""
    x = x.detach().cpu().squeeze(0)  # (C, T, H, W)
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
    return (255 * x).astype(np.uint8)


def write_video_av(path: str, frames: np.ndarray, fps: float = 24):
    """(T, H, W, C) uint8 RGB -> mp4 via av."""
    import av
    T, H, W, C = frames.shape
    container = av.open(path, "w")
    stream = container.add_stream("mpeg4", rate=int(round(fps)))
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"
    for t in range(T):
        frame = av.VideoFrame.from_ndarray(frames[t], format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main():
    parser = argparse.ArgumentParser(description="Encode a video, decode it, save reconstructed video")
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path for decoded video (default: input_reconstructed.mp4)")
    parser.add_argument("--config", type=str,
                        default="configs/Open-MAGVIT2/gpu/ucf101_lfqfan_128_L.yaml",
                        help="Video model config YAML")
    parser.add_argument("--ckpt", type=str, default="checkpoints/Video 128 262144.ckpt",
                        help="Video checkpoint path")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process (default: all)")
    parser.add_argument("--fps", type=float, default=None,
                        help="FPS for output video (default: same as input)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"Video not found: {args.video}")
    if not os.path.isfile(args.config):
        sys.exit(f"Config not found: {args.config}")
    if not os.path.isfile(args.ckpt):
        sys.exit(f"Checkpoint not found: {args.ckpt}")

    out_path = args.output
    if out_path is None:
        base, _ = os.path.splitext(args.video)
        out_path = base + "_reconstructed.mp4"

    print("Loading model...", flush=True)
    model = load_model(args.config, args.ckpt)
    print(f"Using device: {DEVICE}", flush=True)
    print("Loading video...", flush=True)
    try:
        frames, source_fps = load_video_frames(args.video, max_frames=args.max_frames)
    except Exception as e:
        print(f"Error loading video: {e}", flush=True)
        raise
    T = frames.shape[0]
    fps = args.fps if args.fps is not None else source_fps
    print(f"Frames: {T}  Source FPS: {source_fps:.2f}  Output FPS: {fps:.2f}", flush=True)
    if T == 0:
        sys.exit("No frames in video.")

    video_tensor = preprocess_frames(frames, RESOLUTION)
    _, _, t, h, w = video_tensor.shape

    # Process in temporal windows of 17
    output_list = []
    num_windows = (t + TEMPORAL_WINDOW - 1) // TEMPORAL_WINDOW

    # Warmup (MPS/CUDA): run one encode+decode so first window isn't cold
    if DEVICE.type in ("cuda", "mps"):
        warmup_start = 0
        warmup_end = min(TEMPORAL_WINDOW, t)
        warmup_chunk = video_tensor[:, :, warmup_start:warmup_end, :, :].contiguous()
        if warmup_chunk.shape[2] < TEMPORAL_WINDOW:
            pad = TEMPORAL_WINDOW - warmup_chunk.shape[2]
            warmup_chunk = torch.cat([warmup_chunk, warmup_chunk[:, :, -1:, :, :].repeat(1, 1, pad, 1, 1)], dim=2)
        with torch.no_grad():
            if model.use_ema:
                with model.ema_scope():
                    q, _, _, _ = model.encode(warmup_chunk)
                    _ = model.decode(q)
            else:
                q, _, _, _ = model.encode(warmup_chunk)
                _ = model.decode(q)

    print(f"Encoding/decoding {num_windows} window(s)...", flush=True)
    t0 = time.perf_counter()
    for idx in range(num_windows):
        start = idx * TEMPORAL_WINDOW
        end = min(start + TEMPORAL_WINDOW, t)
        chunk = video_tensor[:, :, start:end, :, :].contiguous()
        if chunk.shape[2] < TEMPORAL_WINDOW:
            # Pad last chunk by repeating last frame
            pad = TEMPORAL_WINDOW - chunk.shape[2]
            chunk = torch.cat([chunk, chunk[:, :, -1:, :, :].repeat(1, 1, pad, 1, 1)], dim=2)
        with torch.no_grad():
            if model.use_ema:
                with model.ema_scope():
                    t_encode = time.perf_counter()
                    quant, _, _, _ = model.encode(chunk)
                    encode_s = time.perf_counter() - t_encode
                    t_decode = time.perf_counter()
                    rec = model.decode(quant)
                    decode_s = time.perf_counter() - t_decode
            else:
                t_encode = time.perf_counter()
                quant, _, _, _ = model.encode(chunk)
                encode_s = time.perf_counter() - t_encode
                t_decode = time.perf_counter()
                rec = model.decode(quant)
                decode_s = time.perf_counter() - t_decode
        rec = rec.clamp(-1.0, 1.0)
        if end - start < TEMPORAL_WINDOW:
            rec = rec[:, :, : end - start, :, :]
        output_list.append(rec)
        print(f"  window {idx + 1}/{num_windows}  encode: {encode_s:.3f} s  decode: {decode_s:.3f} s", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"Encode+decode: {elapsed:.2f} s ({elapsed / num_windows:.3f} s per window)", flush=True)

    reconstructed = torch.cat(output_list, dim=2)
    out_frames = tensor_to_frames(reconstructed)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    t_save = time.perf_counter()
    write_video_av(out_path, out_frames, fps=fps)
    print(f"Saved reconstructed video to {out_path} (write: {time.perf_counter() - t_save:.2f} s)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
