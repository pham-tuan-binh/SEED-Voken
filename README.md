# SEED-Voken

Visual tokenizer experiment using the [SEED-Voken](https://github.com/ARC-TencentPCG/SEED-Voken) VQ model to stream encoded video over [LiveKit](https://livekit.io/).

## LiveKit Video Streaming (encoder / decoder)

Stream video through the SEED-Voken VQ model over LiveKit: the encoder reads a video file, encodes each 17-frame chunk into a quantized latent, and sends it as a data packet; the decoder receives packets, decodes them back to RGB frames, and plays the result as a video track in the LiveKit room.

**Setup:**
```bash
# Install deps (includes livekit, livekit-api, python-dotenv)
uv sync

# Configure LiveKit credentials
cp .env.local.example .env.local
# Edit .env.local with your LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
```

**Run:**
```bash
# Terminal 1: start the decoder (waits for incoming data)
uv run python decoder.py --config configs/Open-MAGVIT2/gpu/ucf101_lfqfan_128_L.yaml \
                         --ckpt "checkpoints/Video 128 262144.ckpt"

# Terminal 2: start the encoder (loads video, encodes, streams)
uv run python encoder.py path/to/video.mp4 \
                         --config configs/Open-MAGVIT2/gpu/ucf101_lfqfan_128_L.yaml \
                         --ckpt "checkpoints/Video 128 262144.ckpt" \
                         --fps 24
```

On exit (or Ctrl+C), each process writes a CSV with timing stats:
- `encoder_stats.csv` — chunk_index, encode_time_s, serialize_time_s, send_ts
- `decoder_stats.csv` — chunk_index, send_ts, recv_ts, latency_s, decode_time_s

## Compression

A 128×128 image has 3 channels per pixel, each channel is 1 byte:

> 128 × 128 × 3 = **49,152 bytes**

MAGVIT downsamples this to 16×16 tokens where each token can be represented by 18 bits (codebook size = 2^18 = 262,144):

> 16 × 16 × 18 / 8 = **576 bytes**
>
> 49,152 / 576 = **85.33× compression**

Furthermore, MAGVIT also does temporal downsampling — a 17-frame chunk goes down to 5 frames — so the compression rate becomes:

> 85.33 × 17 / 5 = **290× compression**

Per chunk payload over the wire: **36 bytes header + 2,880 bytes data = 2,916 bytes** for 17 frames of 128×128 video.

## Credits

Based on [SEED-Voken](https://github.com/ARC-TencentPCG/SEED-Voken) by ARC Lab Tencent PCG, Tsinghua University, Nanjing University.

> [Open-MAGVIT2: An Open-source Project Toward Democratizing Auto-Regressive Visual Generation](https://arxiv.org/abs/2409.04410)
> Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, Ying Shan

> [IBQ: Scalable Image Tokenization with Index Backpropagation Quantization](https://arxiv.org/abs/2412.02692)
> Fengyuan Shi, Zhuoyan Luo, Yixiao Ge, Yujiu Yang, Ying Shan, Limin Wang

We thank [Lijun Yu](https://me.lj-y.com/) for his encouraging discussions. We refer a lot from [VQGAN](https://github.com/CompVis/taming-transformers) and [MAGVIT](https://github.com/google-research/magvit). We also refer to [LlamaGen](https://github.com/FoundationVision/LlamaGen), [VAR](https://github.com/FoundationVision/VAR), [RQVAE](https://github.com/kakaobrain/rq-vae-transformer) and [VideoGPT](https://github.com/wilson1yan/VideoGPT), [OmniTokenizer](https://github.com/FoundationVision/OmniTokenizer). We also thank Bowen Zheng for pointing out that the IBQ is mathematically equivalent to the deterministic hard-gumbel without temperature implementation available in the [lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) library. Thanks for their wonderful work.
