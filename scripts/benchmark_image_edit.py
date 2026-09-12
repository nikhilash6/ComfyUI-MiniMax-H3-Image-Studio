"""Run a local H3 generation/edit smoke test; never uploads to an external service."""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(url, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode()) from exc


def run(url, graph):
    started = time.monotonic()
    prompt_id = request(url + "/prompt", {"prompt": graph})["prompt_id"]
    print(f"Queued {prompt_id}", flush=True)
    while time.monotonic() - started < 1800:
        result = request(url + "/history/" + prompt_id).get(prompt_id)
        if result:
            if result["status"]["status_str"] != "success":
                raise RuntimeError(json.dumps(result["status"]))
            return {"seconds": round(time.monotonic() - started, 2), "outputs": result["outputs"]}
        time.sleep(2)
    raise TimeoutError(f"Prompt {prompt_id} did not finish within 30 minutes; inspect the local queue.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8191")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", help="Existing local ComfyUI input filename; otherwise generate a source first.")
    parser.add_argument("--resolution", default="fast preview | 0.40 MP")
    parser.add_argument("--instruction", default="Change the red mug in <Picture 1> to blue. Add a yellow lemon on the table to the right of the mug. Keep the mug shape, handle, background and camera angle unchanged.")
    parser.add_argument("--prefix", default="H3_v22_test")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    results = {}
    source = args.source
    if not source:
        graph = json.loads((repo / "examples/api/H3_IMAGE_GENERATE_API.json").read_text())
        graph["4"]["inputs"].update(aspect_ratio="1:1 square", resolution_profile="fast preview | 0.40 MP")
        graph["5"]["inputs"]["prompt"] = "Product photograph of one red ceramic mug with a curved handle on a pale wooden table. Plain light gray background, soft daylight, front view."
        graph["13"]["inputs"]["filename_prefix"] = "H3_v22_test_source"
        results["generate_cold"] = run(args.url, graph)
        image = results["generate_cold"]["outputs"]["13"]["images"][0]
        source = "/".join(part for part in (image.get("subfolder"), image["filename"]) if part) + " [output]"
        args.output.write_text(json.dumps(results, indent=2))
    for transport in ("native", "semantic (experimental)"):
        for seed in (42, 43):
            graph = json.loads((repo / "examples/api/H3_IMAGE_EDIT_API.json").read_text())
            graph["0"]["inputs"]["image"] = source
            graph["4"]["inputs"]["resolution_profile"] = args.resolution
            graph["5"]["inputs"].update(
                edit_instruction=args.instruction,
                reference_transport=transport,
            )
            graph["6"]["inputs"]["noise_seed"] = seed
            graph["13"]["inputs"]["filename_prefix"] = f"{args.prefix}_{transport.split()[0]}_{seed}"
            key = f"{transport}_{seed}"
            results[key] = run(args.url, graph)
            args.output.write_text(json.dumps(results, indent=2))
            print(key, results[key]["seconds"], flush=True)


if __name__ == "__main__":
    main()
