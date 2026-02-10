#!/usr/bin/env python3
"""
Encode an image with the VQ model, decode it, and show original vs reconstructed.
No dataset or evaluation metrics — just load one image and run encode/decode.
"""
import os
import sys
import argparse
import time

sys.path.insert(0, os.getcwd())

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from src.Open_MAGVIT2.models.lfqgan import VQModel

def _get_device():
    if hasattr(torch, "npu"):
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                return torch.device("npu:0")
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _get_device()


def load_model(config_path: str, ckpt_path: str, size: int = 256):
    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.init_args)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    model = model.eval().to(DEVICE)
    return model


def load_and_preprocess_image(path: str, size: int = 256) -> torch.Tensor:
    """Load image, resize/crop to size x size, normalize to [-1, 1], return (1, 3, H, W)."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = np.array(img)
    # Same as ImagePaths: resize then center crop
    resize = T.Resize(size)
    crop = T.CenterCrop((size, size))
    # T expects PIL or tensor; we have numpy HWC
    img = Image.fromarray(img)
    img = resize(img)
    img = crop(img)
    img = np.array(img)
    img = (img / 127.5 - 1.0).astype(np.float32)
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
    return x


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """(1, 3, H, W) in [-1, 1] -> PIL RGB."""
    x = x.detach().cpu().squeeze(0)
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    return Image.fromarray(x, mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="Encode/decode one image with the VQ model")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--config", type=str, default="configs/Open-MAGVIT2/gpu/imagenet_lfqgan_256_L.yaml",
                        help="Model config YAML")
    parser.add_argument("--ckpt", type=str, default="checkpoints/imagenet_256_L.ckpt",
                        help="Checkpoint path")
    parser.add_argument("--size", type=int, default=256, help="Resize/crop size (default 256)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save comparison image here (default: show with matplotlib)")
    parser.add_argument("--no-show", action="store_true", help="Do not open a window (only save if --output)")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"Not a file: {args.image}")
    if not os.path.isfile(args.config):
        sys.exit(f"Config not found: {args.config}")
    if not os.path.isfile(args.ckpt):
        sys.exit(f"Checkpoint not found: {args.ckpt}")

    print("Loading model...")
    model = load_model(args.config, args.ckpt, args.size)
    print("Loading image...")
    x = load_and_preprocess_image(args.image, args.size)

    with torch.no_grad():
        if model.use_ema:
            with model.ema_scope():
                t_encode = time.perf_counter()
                quant, _emb_loss, _info, _ = model.encode(x)
                encode_s = time.perf_counter() - t_encode
                t_decode = time.perf_counter()
                recon = model.decode(quant)
                decode_s = time.perf_counter() - t_decode
        else:
            t_encode = time.perf_counter()
            quant, _emb_loss, _info, _ = model.encode(x)
            encode_s = time.perf_counter() - t_encode
            t_decode = time.perf_counter()
            recon = model.decode(quant)
            decode_s = time.perf_counter() - t_decode
    recon = torch.clamp(recon, -1.0, 1.0)
    print(f"encode: {encode_s:.3f} s  decode: {decode_s:.3f} s")

    orig_pil = tensor_to_pil(x)
    recon_pil = tensor_to_pil(recon)

    # Side-by-side
    w, h = orig_pil.size
    out = Image.new("RGB", (w * 2, h))
    out.paste(orig_pil, (0, 0))
    out.paste(recon_pil, (w, 0))

    if args.output:
        out.save(args.output)
        print(f"Saved to {args.output}")

    if not args.no_show:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(orig_pil)
            ax[0].set_title("Original")
            ax[0].axis("off")
            ax[1].imshow(recon_pil)
            ax[1].set_title("Reconstructed")
            ax[1].axis("off")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("Could not show plot:", e)
            if not args.output:
                out.save("encode_decode_comparison.png")
                print("Saved encode_decode_comparison.png")


if __name__ == "__main__":
    main()
