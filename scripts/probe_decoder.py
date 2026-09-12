"""Compare the author's decoder against a saved H3 latent, outside ComfyUI."""
import argparse
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file


def main():
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoViTDecoder3d, MiniMaxH3VideoRotaryPosEmbed
    parser = argparse.ArgumentParser()
    parser.add_argument("latent", type=Path)
    parser.add_argument("decoder", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    state = load_file(str(args.decoder))
    with torch.device("meta"):
        decoder = MiniMaxH3VideoViTDecoder3d()
        post = torch.nn.Conv3d(24, 24, 1)
    decoder.load_state_dict({k.removeprefix("decoder."): v for k, v in state.items() if k.startswith("decoder.")}, assign=True)
    post.load_state_dict({k.removeprefix("post_quant_conv."): v for k, v in state.items() if k.startswith("post_quant_conv.")}, assign=True)
    decoder.rope = MiniMaxH3VideoRotaryPosEmbed(48)
    decoder.to(device="cuda").eval()
    post.to(device="cuda").eval()
    from safetensors import safe_open
    with safe_open(args.base, framework="pt") as base:
        mean = base.get_tensor("latents_mean").to("cuda", torch.float16).reshape(1, -1, 1, 1, 1)
        std = base.get_tensor("latents_std").to("cuda", torch.float16).reshape(1, -1, 1, 1, 1)
    latent = load_file(str(args.latent))["latent_tensor"][:, :, :1].to("cuda", torch.float16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        decoded = decoder(post(latent * std + mean))[:, :, -1].float()
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").reshape(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], device="cuda").reshape(1, 3, 1, 1)
    rgb = ((decoded * pixel_std + pixel_mean).clamp(0, 1)[0] * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(rgb).save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
