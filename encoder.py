#!/usr/bin/env python3
"""
Encoder client: load a video file, encode chunks with the SEED-Voken VQ model,
and send quant tensors as LiveKit data packets.

Usage:
    uv run python encoder.py path/to/video.mp4 [--config ...] [--ckpt ...] [--fps 24]
"""
import os
import sys
import argparse
import asyncio
import csv
import logging
import struct
import time

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from dotenv import load_dotenv
from livekit import api, rtc

from encode_decode_video import (
    DEVICE,
    RESOLUTION,
    TEMPORAL_WINDOW,
    load_model,
    load_video_frames,
    preprocess_frames,
)

load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("encoder")

# ── Payload helpers ──────────────────────────────────────────────────────────

# Header layout (all little-endian):
#   chunk_index  uint32
#   total_chunks uint32
#   fps          float32
#   num_frames   uint32   (actual frames before padding)
#   send_ts      float64
#   quant_t      uint32   (temporal dim of latent)
#   quant_h      uint32   (spatial height of latent)
#   quant_w      uint32   (spatial width of latent)
HEADER = struct.Struct("<IIfIdIII")  # 4+4+4+4+8+4+4+4 = 36 bytes


def _pack_18bit(indices: np.ndarray) -> bytes:
    """Bit-pack an array of 18-bit indices into a compact byte string.

    Every 4 indices (4×18 = 72 bits = 9 bytes) are packed together.
    If len(indices) is not a multiple of 4, it is zero-padded on the right.
    """
    vals = indices.astype(np.uint32)
    # Pad to a multiple of 4
    rem = len(vals) % 4
    if rem:
        vals = np.concatenate([vals, np.zeros(4 - rem, dtype=np.uint32)])
    out = bytearray()
    for i in range(0, len(vals), 4):
        a, b, c, d = int(vals[i]), int(vals[i+1]), int(vals[i+2]), int(vals[i+3])
        # 4 × 18 = 72 bits = 9 bytes, little-endian
        bits = a | (b << 18) | (c << 36) | (d << 54)
        out.extend(bits.to_bytes(9, "little"))
    return bytes(out)


def pack_payload(chunk_index: int, total_chunks: int, fps: float,
                 num_frames: int, send_ts: float,
                 quant_t: int, quant_h: int, quant_w: int,
                 indices: np.ndarray) -> bytes:
    """Serialize header + 18-bit packed codebook indices into a single bytes payload.

    Each index is a codebook entry in [0, 2^18).  Four 18-bit values are
    packed into 9 bytes (72 bits), giving the theoretical minimum byte count
    (ceil(n_indices * 18 / 8)).
    """
    header = HEADER.pack(chunk_index, total_chunks, fps, num_frames, send_ts,
                         quant_t, quant_h, quant_w)
    return header + _pack_18bit(indices)


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Encode video and stream quant tensors via LiveKit")
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("--config", type=str,
                        default="configs/Open-MAGVIT2/gpu/ucf101_lfqfan_128_L.yaml",
                        help="Video model config YAML")
    parser.add_argument("--ckpt", type=str, default="checkpoints/Video 128 262144.ckpt",
                        help="Video checkpoint path")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process (default: all)")
    parser.add_argument("--fps", type=float, default=24.0, help="Source video FPS (for throttle + header)")
    parser.add_argument("--csv", type=str, default="encoder_stats.csv", help="CSV output path")
    args = parser.parse_args()

    # Validate paths
    for label, path in [("Video", args.video), ("Config", args.config), ("Checkpoint", args.ckpt)]:
        if not os.path.isfile(path):
            sys.exit(f"{label} not found: {path}")

    # LiveKit credentials
    url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    room_name = os.environ.get("LIVEKIT_ROOM", "edge-cv")

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity("encoder-client")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    # ── Load model ───────────────────────────────────────────────────────
    logger.info("Loading model...")
    model = load_model(args.config, args.ckpt)
    logger.info("Using device: %s", DEVICE)

    # ── Load & preprocess video ──────────────────────────────────────────
    logger.info("Loading video: %s", args.video)
    frames = load_video_frames(args.video, max_frames=args.max_frames)
    total_frames = frames.shape[0]
    logger.info("Loaded %d frames", total_frames)
    if total_frames == 0:
        sys.exit("No frames in video.")

    video_tensor = preprocess_frames(frames, RESOLUTION)
    _, _, t, h, w = video_tensor.shape
    num_windows = (t + TEMPORAL_WINDOW - 1) // TEMPORAL_WINDOW
    chunk_duration = TEMPORAL_WINDOW / args.fps  # seconds per chunk at source FPS

    # ── Warmup ───────────────────────────────────────────────────────────
    if DEVICE.type in ("cuda", "npu"):
        warmup_chunk = video_tensor[:, :, :min(TEMPORAL_WINDOW, t), :, :].contiguous()
        if warmup_chunk.shape[2] < TEMPORAL_WINDOW:
            pad = TEMPORAL_WINDOW - warmup_chunk.shape[2]
            warmup_chunk = torch.cat([warmup_chunk, warmup_chunk[:, :, -1:, :, :].repeat(1, 1, pad, 1, 1)], dim=2)
        with torch.no_grad():
            if model.use_ema:
                with model.ema_scope():
                    q, _, _, _ = model.encode(warmup_chunk)
            else:
                q, _, _, _ = model.encode(warmup_chunk)
        del q, warmup_chunk
        logger.info("Warmup done")

    # ── Connect to LiveKit ───────────────────────────────────────────────
    room = rtc.Room()
    logger.info("Connecting to %s  room=%s", url, room_name)
    await room.connect(url, token)
    logger.info("Connected")

    # ── Encode & send loop ───────────────────────────────────────────────
    stats: list[dict] = []
    try:
        logger.info("Encoding %d window(s)...", num_windows)
        for idx in range(num_windows):
            start = idx * TEMPORAL_WINDOW
            end = min(start + TEMPORAL_WINDOW, t)
            actual_frames = end - start
            chunk = video_tensor[:, :, start:end, :, :].contiguous()

            # Pad last chunk
            if chunk.shape[2] < TEMPORAL_WINDOW:
                pad = TEMPORAL_WINDOW - chunk.shape[2]
                chunk = torch.cat([chunk, chunk[:, :, -1:, :, :].repeat(1, 1, pad, 1, 1)], dim=2)

            # Encode
            t_enc = time.perf_counter()
            with torch.no_grad():
                if model.use_ema:
                    with model.ema_scope():
                        quant, _, indices, _ = model.encode(chunk)
                else:
                    quant, _, indices, _ = model.encode(chunk)
            encode_s = time.perf_counter() - t_enc

            # Extract latent shape and codebook indices
            _, _, qt, qh, qw = quant.shape  # (B, C, T', H', W')
            indices_np = indices.detach().cpu().numpy().astype(np.uint32)

            # Serialize
            t_ser = time.perf_counter()
            send_ts = time.time()
            payload = pack_payload(idx, num_windows, args.fps, actual_frames, send_ts,
                                   qt, qh, qw, indices_np)
            serialize_s = time.perf_counter() - t_ser

            # Send
            await room.local_participant.publish_data(
                payload=payload,
                reliable=True,
                topic="video_embeddings",
            )

            logger.info(
                "chunk %d/%d  encode=%.3fs  serialize=%.3fs  send_ts=%.3f  payload=%d bytes",
                idx + 1, num_windows, encode_s, serialize_s, send_ts, len(payload),
            )
            stats.append({
                "chunk_index": idx,
                "encode_time_s": round(encode_s, 6),
                "serialize_time_s": round(serialize_s, 6),
                "send_ts": send_ts,
            })

            # Throttle to mimic real-time playback rate
            await asyncio.sleep(chunk_duration)

        logger.info("All %d chunks sent", num_windows)
    finally:
        # Write CSV
        if stats:
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["chunk_index", "encode_time_s", "serialize_time_s", "send_ts"])
                writer.writeheader()
                writer.writerows(stats)
            logger.info("Wrote %s (%d rows)", args.csv, len(stats))

        await room.disconnect()
        logger.info("Disconnected")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
