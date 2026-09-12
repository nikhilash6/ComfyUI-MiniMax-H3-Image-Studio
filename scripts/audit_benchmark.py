"""Export compact, non-private measurements from a completed local benchmark."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageStat


def audit(directory):
    results = json.loads((directory / "results.json").read_text())
    records = []
    for key, result in results.items():
        output_id = "16" if key.startswith("refiner_") else "13"
        item = result.get("outputs", {}).get(output_id, {}).get("images", [])
        record = {"case": key, "status": result["status"]["status_str"],
                  "execution_seconds": result.get("execution_seconds"), "selected_frame": None}
        if "previous_attempt" in result:
            record["previous_status"] = result["previous_attempt"]["status"]["status_str"]
        expected = (result.get("width"), result.get("height"))
        if key.startswith("refiner_"):
            source_case = key.removeprefix("refiner_")
            source_result = results.get(f"t2i_{source_case}_turbo8_104", {})
            expected = (source_result.get("width"), source_result.get("height"))
        if item:
            filename = directory / "images" / item[0].get("subfolder", "") / item[0]["filename"]
            with Image.open(filename) as source:
                image = source.convert("RGB")
                record.update(width=image.width, height=image.height,
                              mean_rgb=[round(x, 2) for x in ImageStat.Stat(image).mean],
                              dimensions_match=image.size == expected if all(expected) else None)
                selected = image.tobytes()
            for index, candidate in enumerate(result.get("outputs", {}).get("90", {}).get("images", [])):
                with Image.open(directory / "images" / candidate.get("subfolder", "") / candidate["filename"]) as frame:
                    if frame.size == image.size and frame.convert("RGB").tobytes() == selected:
                        record["selected_frame"] = index
                        break
        record["output_dimensions"] = {}
        for node_id, output in result.get("outputs", {}).items():
            if node_id == "90":
                continue
            sizes = []
            for candidate in output.get("images", []):
                with Image.open(directory / "images" / candidate.get("subfolder", "") / candidate["filename"]) as frame:
                    sizes.append(list(frame.size))
            if sizes:
                record["output_dimensions"][node_id] = sizes
        records.append(record)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = audit(args.directory)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Exported {len(records)} records to {args.output}")
