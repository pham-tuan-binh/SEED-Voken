#!/usr/bin/env python3
"""
Decoder client: join a LiveKit room, receive encoded quant tensors,
decode with the SEED-Voken VQ model, and play back as a video track in the room.

Usage:
    uv run python decoder.py [--config ...] [--ckpt ...] [--fps 24]
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
    tensor_to_frames,
)

load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("decoder")

# ── Payload helpers (must match encoder.py) ──────────────────────────────────

HEADER = struct.Struct("<IIfIdIII")  # 36 bytes


MASK18 = (1 << 18) - 1  # 0x3FFFF


def _unpack_18bit(buf: bytes, n: int) -> np.ndarray:
    """Unpack *n* 18-bit indices from a packed byte string (little-endian).

    Every 9 bytes encode 4 indices.
    """
    out = np.empty(n, dtype=np.uint32)
    idx = 0
    off = 0
    while idx < n:
        bits = int.from_bytes(buf[off:off+9], "little")
        off += 9
        for _ in range(4):
            if idx >= n:
                break
            out[idx] = bits & MASK18
            bits >>= 18
            idx += 1
    return out


def unpack_payload(data: bytes):
    """Deserialize header + 18-bit packed codebook indices from encoder payload."""
    offset = HEADER.size
    (chunk_index, total_chunks, fps, num_frames, send_ts,
     qt, qh, qw) = HEADER.unpack_from(data, 0)

    n_indices = qt * qh * qw
    indices = _unpack_18bit(data[offset:], n_indices)

    return chunk_index, total_chunks, fps, num_frames, send_ts, qt, qh, qw, indices


def indices_to_quant(model, indices: np.ndarray, qt: int, qh: int, qw: int) -> torch.Tensor:
    """Reconstruct the quantized embedding tensor from codebook indices.

    indices:  flat uint32 array of length qt*qh*qw
    returns:  (1, embed_dim, qt, qh, qw) float tensor on DEVICE
    """
    idx = torch.from_numpy(indices.copy()).long().to(DEVICE)
    # LFQ.decode expects (..., num_codebooks) -> (..., num_codebooks * codebook_dim)
    idx = idx.reshape(1, qt, qh, qw, 1)          # (B, T, H, W, num_codebooks=1)
    emb = model.quantize.decode(idx)               # (B, T, H, W, embed_dim=18)
    return emb.permute(0, 4, 1, 2, 3).contiguous() # (B, embed_dim, T, H, W)


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Decode quant tensors from LiveKit and play video in room")
    parser.add_argument("--config", type=str,
                        default="configs/Open-MAGVIT2/gpu/ucf101_lfqfan_128_L.yaml",
                        help="Video model config YAML")
    parser.add_argument("--ckpt", type=str, default="checkpoints/Video 128 262144.ckpt",
                        help="Video checkpoint path")
    parser.add_argument("--fps", type=float, default=24.0, help="Playback FPS")
    parser.add_argument("--csv", type=str, default="decoder_stats.csv", help="CSV output path")
    args = parser.parse_args()

    for label, path in [("Config", args.config), ("Checkpoint", args.ckpt)]:
        if not os.path.isfile(path):
            sys.exit(f"{label} not found: {path}")

    # LiveKit credentials
    url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    room_name = os.environ.get("LIVEKIT_ROOM", "edge-cv")

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity("decoder")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    # ── Load model ───────────────────────────────────────────────────────
    logger.info("Loading model...")
    model = load_model(args.config, args.ckpt)
    logger.info("Using device: %s", DEVICE)

    # ── Warmup ───────────────────────────────────────────────────────────
    # Latent shape for 17 frames @ 128px: (1, 18, 5, 16, 16)
    #   spatial = RESOLUTION / 8,  temporal = 5 (from 17 via two (2,2,2) strides)
    if DEVICE.type in ("cuda", "npu"):
        latent_h = RESOLUTION // 8
        dummy_indices = torch.zeros(1, 5, latent_h, latent_h, 1, dtype=torch.long, device=DEVICE)
        dummy_quant = model.quantize.decode(dummy_indices)            # (1, 5, H', W', 18)
        dummy_quant = dummy_quant.permute(0, 4, 1, 2, 3).contiguous() # (1, 18, 5, H', W')
        with torch.no_grad():
            if model.use_ema:
                with model.ema_scope():
                    _ = model.decode(dummy_quant)
            else:
                _ = model.decode(dummy_quant)
        del dummy_indices, dummy_quant
        logger.info("Warmup done")

    # ── Connect to LiveKit ───────────────────────────────────────────────
    room = rtc.Room()

    # Publish a video track so the decoded frames appear in the room
    source = rtc.VideoSource(RESOLUTION, RESOLUTION)
    track = rtc.LocalVideoTrack.create_video_track("decoded-video", source)

    frame_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=512)
    stats: list[dict] = []
    total_expected = None
    playback_fps = args.fps

    # ── Data received handler ────────────────────────────────────────────
    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        nonlocal total_expected, playback_fps
        if data.topic != "video_embeddings":
            return

        recv_ts = time.time()
        try:
            (chunk_index, total_chunks, fps, num_frames, send_ts,
             qt, qh, qw, indices) = unpack_payload(data.data)
        except Exception as e:
            logger.error("Failed to unpack payload: %s", e)
            return

        if total_expected is None:
            total_expected = total_chunks
            playback_fps = fps
            logger.info("Stream started: %d total chunks, fps=%.1f", total_chunks, fps)

        latency_s = recv_ts - send_ts

        # Reconstruct quant from codebook indices, then decode
        t_dec = time.perf_counter()
        quant = indices_to_quant(model, indices, qt, qh, qw)
        with torch.no_grad():
            if model.use_ema:
                with model.ema_scope():
                    rec = model.decode(quant)
            else:
                rec = model.decode(quant)
        rec = rec.clamp(-1.0, 1.0)
        # Trim padding
        if num_frames < TEMPORAL_WINDOW:
            rec = rec[:, :, :num_frames, :, :]
        decode_s = time.perf_counter() - t_dec

        # Convert to RGB frames and push to buffer
        rgb_frames = tensor_to_frames(rec)
        for frame in rgb_frames:
            try:
                frame_queue.put_nowait(frame)
            except asyncio.QueueFull:
                logger.warning("Frame queue full, dropping frame")

        logger.info(
            "chunk %d/%d  latency=%.3fs  decode=%.3fs  frames=%d",
            chunk_index + 1, total_chunks, latency_s, decode_s, num_frames,
        )
        stats.append({
            "chunk_index": chunk_index,
            "send_ts": send_ts,
            "recv_ts": recv_ts,
            "latency_s": round(latency_s, 6),
            "decode_time_s": round(decode_s, 6),
        })

    # ── Playback worker ──────────────────────────────────────────────────
    async def playback():
        while True:
            frame = await frame_queue.get()
            h, w, c = frame.shape
            video_frame = rtc.VideoFrame(w, h, rtc.VideoBufferType.RGB24, frame.tobytes())
            source.capture_frame(video_frame)
            await asyncio.sleep(1.0 / playback_fps)

    # ── Run ──────────────────────────────────────────────────────────────
    logger.info("Connecting to %s  room=%s", url, room_name)
    await room.connect(url, token)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    )
    logger.info("Connected, publishing decoded video track. Waiting for data...")

    playback_task = asyncio.create_task(playback())

    try:
        await asyncio.Future()  # run forever until Ctrl+C
    except asyncio.CancelledError:
        pass
    finally:
        playback_task.cancel()

        # Latency summary
        if stats:
            latencies = [s["latency_s"] for s in stats]
            logger.info(
                "Latency summary: min=%.3fs  max=%.3fs  avg=%.3fs  (%d chunks)",
                min(latencies), max(latencies), sum(latencies) / len(latencies), len(latencies),
            )

        # Write CSV
        if stats:
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["chunk_index", "send_ts", "recv_ts", "latency_s", "decode_time_s"])
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
