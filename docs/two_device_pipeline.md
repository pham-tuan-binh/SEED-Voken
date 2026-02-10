# Encode on one device, decode on another

## Signal: what to send

The **encoder** turns 17 frames (1, 3, 17, 128, 128) into:
- **quant**: float tensor ~(1, 18, T', 16, 16) — the quantized latent the decoder needs.
- **indices**: int tensor, one code per spatiotemporal position (compact).

**Send one of:**
1. **quant** — ~90 KB per window, easy: decoder runs `decode(quant)`. No shape logic.
2. **indices** — ~5–10 KB per window (e.g. 1280 ints). Decoder must reshape and call `decode_code(indices)` (needs correct shape / pack info).

For simplicity, sending **quant** is enough; for minimum bandwidth, send **indices** and reshape on the decoder side.

## Timing (example: encode 0.15 s per window)

- **Device A (encoder):** every **0.15 s** finishes one window → send quant (or indices) to B.
- **Device B (decoder):** decode ~**0.05 s** → 17 frames out; then wait for next chunk.

So the pipeline is **encode-bound**; B is idle part of the time. Throughput = 17 frames every 0.15 s ≈ **113 fps** of input.

```
Device A:  [encode W1][encode W2][encode W3]...
             ↓ 0.15s   ↓ 0.15s   ↓ 0.15s
           send W1   send W2   send W3
             ↓          ↓          ↓
Device B:  [recv][decode W1][recv][decode W2][recv][decode W3]...
               ~0.05s      ~0.05s      ~0.05s
             → 17 frames → 17 frames → 17 frames
```

## Implementation options

1. **Files:** A writes `quant.pt` (or `indices.pt`) per window; B reads from a shared mount or copies (e.g. rsync), then decodes. Simple, no daemon.
2. **Queue (Redis/RabbitMQ/etc.):** A pushes bytes; B pops and decodes. Good for many workers.
3. **Socket:** A sends quant bytes over TCP; B listens and decodes. Single producer, single consumer.
4. **ZMQ:** Same idea, with ready-made patterns (PUSH/PULL or PUB/SUB).

## Steps

**Device A (encoder):**
1. Load video (or stream), preprocess 17-frame windows.
2. Run `quant, _, _, _ = model.encode(chunk)`.
3. Serialize `quant` (e.g. `torch.save(quant.cpu(), buf)` or `.numpy().tobytes()`) and send to B (file, socket, or queue).

**Device B (decoder):**
1. Load same model (decoder + quantizer only if using indices; full model for `decode(quant)`).
2. Receive quant (or indices) for one window.
3. If quant: `recon = model.decode(quant.to(device))`. If indices: reshape then `recon = model.decode_code(indices.to(device))`.
4. Append recon to output video or stream.

## Minimal scripts

- `encode_only_video.py`: reads video, encodes each window, writes one file per window (e.g. `out/window_000.pt`) or streams over a socket.
- `decode_only_video.py`: reads those files (or socket), decodes, writes reconstructed video.

Share storage (NFS, S3, etc.) or a small message bus between A and B.

---

## Streaming: buffers and latency

For **live** streams (camera, capture, etc.) you have to think in terms of buffers and end-to-end latency.

### 1. Encoder side (Device A)

- **Frame buffer (input):** You need **17 frames** before the first encode. So you always have at least **17 × (1/fps)** seconds of input latency (e.g. 17/24 ≈ **0.71 s** at 24 fps).
- **Windowing:** Non-overlapping: frames 0–16 → window 0, 17–33 → window 1, … Overlapping would reduce latency but duplicate work and change semantics.
- **Send queue:** After each encode you push one quant (~90 KB) onto a queue (or write to socket). If the network or the decoder is slow, this queue grows → you need either **backpressure** (block encoder until queue drains) or **drop** (skip windows). For real-time, a small bounded queue (e.g. 2–4 windows) + block when full keeps memory bounded and applies backpressure.

### 2. Decoder side (Device B)

- **Receive queue:** Incoming quants (from socket or queue). Decoder thread pops one quant, decodes (~0.05 s), pushes 17 frames into the **output frame buffer**.
- **Output frame buffer:** Decode produces 17 frames per window in a burst; display or recorder usually wants frames at a **steady rate** (e.g. 24 fps). So you need a **ring buffer** or queue of decoded frames: producer = decode thread, consumer = output thread that reads one frame every 1/24 s. Size this so you don’t underrun (buffer empty when output needs a frame) or grow unbounded. Example: 2–3 windows (34–51 frames) at 24 fps ≈ 1.4–2.1 s of buffer.

### 3. Latency (ballpark)

| Stage | Time |
|-------|------|
| Input frame buffer (17 frames) | 17/fps (e.g. 0.71 s @ 24 fps) |
| Encode one window | ~0.15 s |
| Transfer (quant) | ~negligible for 90 KB on LAN |
| Decode | ~0.05 s |
| Output buffer drain (first frame of window) | 0 (if you output as soon as decoded) |

**End-to-end (first frame in → first frame out):** ≈ **17/fps + 0.15 + 0.05** (+ network RTT if any). At 24 fps that’s ~**0.9–1.0 s** minimum.

### 4. Suggested buffer sizes (streaming)

- **Encoder send queue:** 2–4 windows. If full → block encoder (backpressure) so you don’t run OOM or fall too far behind.
- **Decoder receive queue:** Same as above if you share a single queue; else 2–4 quants.
- **Decoder output frame buffer:** 34–51 frames (2–3 windows). Lets you absorb decode jitter and feed a steady 24 fps output.

### 5. Backpressure

- **Encode > network or decode:** Encoder produces quants faster than B can receive/decode. Without backpressure the encoder’s send queue grows. **Option A:** Block encoder when send queue is full (keeps latency bounded, may drop input frames if capture isn’t blocking). **Option B:** Drop windows (encoder skips encoding some windows when queue is full) → lower quality, no backpressure on capture.
- **Decode > display/recorder:** Decoder produces 17 frames every ~0.05 s; output consumes at 24 fps. So decoder is faster; the output frame buffer will fill. Cap the buffer size and, when full, either block the decoder (backpressure to receive queue and then to encoder) or drop decoded frames (not ideal).

### 6. Simple streaming pipeline sketch

```
[Capture] → frame_buf(17) → [Encode] → send_queue(2–4) → [Network] →
  → recv_queue(2–4) → [Decode] → frame_buf(34–51) → [Output @ 24 fps]
```

- **frame_buf(17):** collect 17 frames, then pass to encoder; refill for next window.
- **send_queue / recv_queue:** bounded so encoder blocks when network/B is slow.
- **frame_buf(34–51):** decoded frames; output thread drains at fixed fps.
