"""Combine the 500K decoder with an H3 encoder for ComfyUI's native VAE loader.

Converts names/layout and casts to FP16. Output is for T=1 decoding only.
"""
import argparse
import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DECODER_SHA256 = "6c5ff2caa8fade6769f4dd53ee244f77a06652c8cb66b2fedc93f75046d9f001"


def pack_qkv(parts):
    # ComfyUI groups Q/K/V within each 64-channel head.
    tensor = torch.stack([part.reshape(32, 64, *part.shape[1:]) for part in parts], dim=1)
    return tensor.reshape(3 * parts[0].shape[0], *parts[0].shape[1:])


def pack_swiglu(tensor):
    # Diffusers stores [value, gate]; ComfyUI applies SiLU to the first half.
    value, gate = tensor.chunk(2, dim=0)
    return torch.cat((gate, value), dim=0)


def convert(base_path, decoder_path, output_path):
    if output_path.exists():
        raise FileExistsError(output_path)
    checksum = hashlib.sha256()
    with decoder_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    digest = checksum.hexdigest()
    if digest != DECODER_SHA256:
        raise ValueError("Decoder differs from the author's audited 500K release.")
    converted = {}
    consumed = set()
    with safe_open(base_path, framework="pt") as base, safe_open(decoder_path, framework="pt") as decoder:
        for key in base.keys():
            if key.startswith(("decoder.", "post_quant_conv.")) and key != "decoder.mask_token":
                source = key.replace("decoder.x_embedder.", "decoder.proj_in.")
                source = source.replace(".attn.to_out.", ".attn.to_out.0.")
                source = source.replace(".ff.w1.", ".ff.net.0.proj.").replace(".ff.w2.", ".ff.net.2.")
                if ".attn.to_qkv." in source:
                    keys = [source.replace(".to_qkv.", f".to_{part}.") for part in ("q", "k", "v")]
                    parts = [decoder.get_tensor(k).to(torch.float16) for k in keys]
                    tensor = pack_qkv(parts)
                    consumed.update(keys)
                else:
                    tensor = decoder.get_tensor(source).to(torch.float16)
                    if ".ff.w1." in key:
                        tensor = pack_swiglu(tensor)
                    consumed.add(source)
            else:
                tensor = base.get_tensor(key).clone()
            if list(tensor.shape) != base.get_slice(key).get_shape():
                raise ValueError(f"Shape mismatch for {key}")
            converted[key] = tensor.contiguous()
        if consumed != set(decoder.keys()):
            raise ValueError(f"Unmapped decoder tensors: {set(decoder.keys()) - consumed}")
    save_file(converted, output_path, metadata={
        "source": "iamkaikai/MiniMax-H3-Single-Frame-VAE-500K",
        "decoder_sha256": digest, "usage": "Experimental T=1 image decode only; not a video VAE",
        "encoder_source": base_path.name,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.base, args.decoder, args.output)
    print(args.output)
