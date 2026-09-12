"""Build inspection contact sheets without modifying benchmark originals."""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output-node", default="13")
    args = parser.parse_args()
    results = json.loads((args.directory / "results.json").read_text())
    items = []
    for key, result in results.items():
        if key.startswith(args.prefix):
            outputs = result.get("outputs", {})
            output_ids = [name for name in outputs if name != "90"] if args.output_node == "all" else [args.output_node]
            for output_id in output_ids:
                for item in outputs.get(output_id, {}).get("images", []):
                    label = f"{key}_out{output_id}" if args.output_node == "all" else key
                    items.append((label, args.directory / "images" / item.get("subfolder", "") / item["filename"], result.get("execution_seconds")))
    items.sort()
    for offset in range(0, len(items), 12):
        batch = items[offset:offset+12]
        sheet = Image.new("RGB", (480 * 4, 540 * math.ceil(len(batch)/4)), "#202020")
        draw = ImageDraw.Draw(sheet)
        for index, (key, image_path, seconds) in enumerate(batch):
            x, y = index % 4 * 480, index // 4 * 540
            with Image.open(image_path) as original:
                preview = ImageOps.contain(original.convert("RGB"), (474, 480))
            sheet.paste(preview, (x + (480-preview.width)//2, y + 40 + (480-preview.height)//2))
            draw.text((x+8, y+4), key.replace(args.prefix, "", 1), fill="white")
            draw.text((x+8, y+18), f"{seconds}s", fill="white")
        path = args.directory / f"sheet_{args.prefix}_{args.output_node}_{offset//12}.jpg"
        sheet.save(path, quality=94)
        print(path)


if __name__ == "__main__":
    main()
