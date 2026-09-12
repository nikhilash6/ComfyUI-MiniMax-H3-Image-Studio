"""Reproducible local H3 comparison. Keeps failures, prompts and every decoded frame."""
import argparse
import json
import time
from pathlib import Path

from benchmark_image_edit import request

REPO = Path(__file__).resolve().parents[1]
PROMPTS = {
    "portrait": "Full-body studio photograph of an adult woman with short curly black hair and round glasses, wearing a mustard yellow jacket, dark blue trousers and white sneakers. She sits on a wooden chair, both hands resting separately on her knees, both feet visible on the floor. Plain dark gray backdrop, soft light from the left, realistic skin and woven fabric texture.",
    "street": 'Wide architectural photograph of a small bookshop on a rainy evening. The sign above its door reads "NORD" in four clear white capital letters. Warm light inside, dark blue exterior, wet cobblestones with reflections. A red bicycle leans against the left wall. Detailed books in the window. No people.',
    "pose": "Full-body studio photograph of an adult woman with long straight brown hair wearing a plain teal tracksuit, standing with both arms stretched horizontally sideways in a T-pose. Both hands and feet fully visible. Plain light gray backdrop, even lighting, front view.",
}
LORAS = {
    "turbo8": "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    "turbo4": "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors",
    "dare6": "dareties_pruned/minimax_h3_fl2v_lightx2v_v0.1_dareties_v4_step600_comfy_fro_rev2_pruned.safetensors",
    "dare6_rev6": "dareties_pruned/minimax_h3_fl2v_lightx2v_v0.1_dareties_v4_step600_comfy_fro_rev6_pruned.safetensors",
    "ref8": "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    "ref4": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
}


def template(slug):
    return json.loads((REPO / "examples/api" / f"{slug}_API.json").read_text())


def size(graph, case, mp):
    width, height = {1: (1024, 1024), 2: (1440, 1440), 4: (2048, 2048)}[mp]
    if case == "street":
        width, height = {1: (1344, 768), 2: (1920, 1088), 4: (2688, 1536)}[mp]
    graph.pop("4", None)
    graph["5"]["inputs"].update(width=width, height=height)
    return width, height


def recipe(graph, name):
    graph.pop("15", None)
    sampler = graph["8"]["inputs"]
    sampler.update(model=["1", 0], sampling_profile="base quality | RES 20 steps")
    if name == "base20":
        return
    graph["15"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["1", 0], "lora_name": LORAS[name], "strength_model": 0.8 if name.startswith("dare6") else 1.0,
    }}
    sampler.update(model=["15", 0], sampling_profile="custom | use controls below",
                   custom_sampler="euler", custom_scheduler="simple",
                   custom_steps=6 if name.startswith("dare6") else 4 if name in ("turbo4", "ref4") else 8,
                   custom_shift_video=8.0 if name.startswith("dare6") else 12.0 if name in ("ref4", "ref8") else 6.0,
                   custom_shift_audio=3.0, custom_denoise=1.0,
                   custom_beta_alpha=0.5, custom_beta_beta=0.5)


def source_filename(results, key):
    image = results[key]["outputs"]["13"]["images"][0]
    return "/".join(p for p in (image.get("subfolder"), image["filename"]) if p) + " [output]"


def execute(url, graph, deadline):
    started = time.monotonic()
    response = request(url + "/prompt", {"prompt": graph})
    prompt_id = response["prompt_id"]
    while time.monotonic() - started < deadline:
        result = request(url + "/history/" + prompt_id).get(prompt_id)
        if result:
            events = result["status"]["messages"]
            start = next((event[1]["timestamp"] for event in events if event[0] == "execution_start"), None)
            end = next((event[1]["timestamp"] for event in events if event[0] == "execution_success"), None)
            return {"prompt_id": prompt_id, "wall_seconds": round(time.monotonic() - started, 3),
                    "execution_seconds": round((end-start)/1000, 3) if start is not None and end is not None else None,
                    "status": result["status"], "outputs": result["outputs"]}
        time.sleep(0.5)
    raise TimeoutError(f"Timed out on {prompt_id}; stop and inspect the queue before resuming.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["t2i", "edit", "edit_stress", "single", "stress", "refiner", "refiner_stress", "decoder", "decoder_edit", "context"])
    parser.add_argument("--url", default="http://127.0.0.1:8191")
    parser.add_argument("--directory", type=Path, required=True, help="Benchmark root; ComfyUI must save into its images/ subfolder.")
    parser.add_argument("--recipes", nargs="+")
    parser.add_argument("--cases", nargs="+", help="Run only these exact case IDs.")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Keep previous attempts and rerun selected cases.")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    result_path = args.directory / "results.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else {}
    lora_names = request(args.url + "/object_info/LoraLoaderModelOnly")["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0]
    plan = []
    if args.stage in ("t2i", "stress"):
        for name in args.recipes or (["base20", "turbo8", "turbo4", "dare6"] if args.stage == "t2i" else ["base20", "turbo8"]):
            for case in ("portrait", "street", "pose") if args.stage == "t2i" else ("portrait", "street"):
                for mp in (1, 2) if args.stage == "t2i" else (4,):
                    for seed in (104, 105) if args.stage == "t2i" else (104,):
                        graph = template("H3_IMAGE_GENERATE")
                        recipe(graph, name)
                        graph["5"]["inputs"]["prompt"] = PROMPTS[case]
                        plan.append((f"t2i_{case}_{mp}mp_{name}_{seed}", graph, case, mp, seed))
    elif args.stage in ("edit", "edit_stress"):
        for name in args.recipes or (("ref8", "ref4") if args.stage == "edit_stress" else ("base20", "ref8", "ref4", "semantic8", "legacy8")):
            for case in ("wardrobe", "pose_transfer"):
                for mp in (4,) if args.stage == "edit_stress" else (1, 2):
                    for seed in (204,) if args.stage == "edit_stress" else (204, 205):
                        graph = template("H3_I2I_TURBO" if name == "legacy8" else "H3_IMAGE_EDIT")
                        recipe(graph, "ref8" if name == "semantic8" else "turbo8" if name == "legacy8" else name)
                        graph["0"]["inputs"]["image"] = source_filename(results, "t2i_portrait_1mp_base20_104")
                        inputs = graph["5"]["inputs"]
                        inputs["edit_instruction"] = (
                            "Replace the mustard yellow jacket with a bright red leather jacket. Keep the same woman, face, glasses, seated pose, trousers, shoes, chair, camera and dark background from <Picture 1>."
                            if case == "wardrobe" else
                            "The woman from <Picture 1> is now standing with both arms stretched horizontally sideways in a T-pose, matching the body pose from <Picture 2>. Keep her curly black hair, face, glasses, mustard jacket, blue trousers, white shoes and dark studio background from <Picture 1>. Remove the chair completely."
                        )
                        if name != "legacy8":
                            inputs["reference_transport"] = "semantic (experimental)" if name == "semantic8" else "native"
                        if case == "pose_transfer" and name != "legacy8":
                            graph["14"] = {"class_type": "LoadImage", "inputs": {"image": source_filename(results, "t2i_pose_1mp_base20_104")}}
                            inputs["reference_image_2"] = ["14", 0]
                        if case == "pose_transfer" and name == "legacy8":
                            inputs["edit_instruction"] = inputs["edit_instruction"].replace(
                                ", matching the body pose from <Picture 2>", "")
                        plan.append((f"edit_{case}_{mp}mp_{name}_{seed}", graph, "portrait", mp, seed))
    elif args.stage == "single":
        for case, slug in (("portrait", "H3_T2I_SINGLE"), ("street", "H3_T2I_SINGLE"), ("wardrobe", "H3_I2I_SINGLE"), ("pose_transfer", "H3_REFERENCE_SINGLE")):
            for mp in (1, 2):
                for seed in (104, 105):
                    graph = template(slug)
                    if case in PROMPTS:
                        graph["5"]["inputs"]["prompt"] = PROMPTS[case]
                    else:
                        graph["0"]["inputs"]["image"] = source_filename(results, "t2i_portrait_1mp_base20_104")
                        graph["5"]["inputs"]["edit_instruction"] = "Change the jacket to red leather. Keep the woman, face, glasses, pose and background unchanged." if case == "wardrobe" else "Keep the woman and clothing from <Picture 1>, but use the standing T-pose from <Picture 2>. Remove the chair and keep the dark background."
                        if "14" in graph:
                            graph["14"]["inputs"]["image"] = source_filename(results, "t2i_pose_1mp_base20_104")
                    plan.append((f"single_{case}_{mp}mp_{seed}", graph, "street" if case == "street" else "portrait", mp, seed))
    elif args.stage in ("refiner", "refiner_stress"):
        mp = 4 if args.stage == "refiner_stress" else 2
        for case in ("portrait", "street"):
            graph = template("H3_DETAIL_REFINER")
            graph["0"]["inputs"]["image"] = source_filename(results, f"t2i_{case}_{mp}mp_turbo8_104")
            graph["91"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"refiner_{case}_raw"}}
            plan.append((f"refiner_{case}_{mp}mp", graph, case, None, None))
    elif args.stage in ("decoder", "decoder_edit"):
        editing = args.stage == "decoder_edit"
        for case in ("wardrobe", "pose_transfer") if editing else ("portrait", "street"):
            for mp in (1, 2):
                for seed in (204, 205) if editing else (104, 105):
                    graph = template("H3_IMAGE_EDIT" if editing else "H3_IMAGE_GENERATE")
                    if editing:
                        graph["0"]["inputs"]["image"] = source_filename(results, "t2i_portrait_1mp_base20_104")
                        graph["5"]["inputs"]["edit_instruction"] = (
                            "Replace the mustard yellow jacket with a bright red leather jacket. Keep the same woman, face, glasses, seated pose, trousers, shoes, chair, camera and dark background from <Picture 1>."
                            if case == "wardrobe" else
                            "The woman from <Picture 1> is now standing with both arms stretched horizontally sideways in a T-pose, matching the body pose from <Picture 2>. Keep her curly black hair, face, glasses, mustard jacket, blue trousers, white shoes and dark studio background from <Picture 1>. Remove the chair completely."
                        )
                        if case == "pose_transfer":
                            graph["14"] = {"class_type": "LoadImage", "inputs": {"image": source_filename(results, "t2i_pose_1mp_base20_104")}}
                            graph["5"]["inputs"]["reference_image_2"] = ["14", 0]
                    else:
                        graph["5"]["inputs"]["prompt"] = PROMPTS[case]
                    for number, filename in ((21, "minimax_h3_t1_image_vae_step1597.safetensors"), (22, "minimax_h3_image_vae_500k_comfy_fp16.safetensors")):
                        graph[str(number)] = {"class_type": "VAELoader", "inputs": {"vae_name": filename}}
                        for index in (0, 1):
                            decode_id, save_id = str(number * 10 + index), str(number * 10 + index + 100)
                            graph[decode_id] = {"class_type": "H3ImageDecode", "inputs": {
                                "samples": ["10", 0], "vae": [str(number), 0], "decode_mode": "single_latent_slice", "latent_index": index,
                            }}
                            graph[save_id] = {"class_type": "SaveImage", "inputs": {
                                "images": [decode_id, 0], "filename_prefix": f"decoder_{case}_{mp}mp_{seed}_{number}_slice{index}",
                            }}
                    plan.append((f"decoder_{case}_{mp}mp_{seed}", graph, case, mp, seed))
    elif args.stage == "context":
        for frames, profile in ((9, "extended quality | 9 frames"), (20, "maximum quality | 20 frames (slow)")):
            for case in ("portrait", "street", "pose_transfer"):
                graph = template("H3_IMAGE_EDIT" if case == "pose_transfer" else "H3_IMAGE_GENERATE")
                if case == "pose_transfer":
                    graph["0"]["inputs"]["image"] = source_filename(results, "t2i_portrait_1mp_base20_104")
                    graph["14"] = {"class_type": "LoadImage", "inputs": {"image": source_filename(results, "t2i_pose_1mp_base20_104")}}
                    graph["5"]["inputs"].update(reference_image_2=["14", 0], edit_instruction="The woman from <Picture 1> is now standing with both arms stretched horizontally sideways in a T-pose, matching the body pose from <Picture 2>. Keep her curly black hair, face, glasses, mustard jacket, blue trousers, white shoes and dark studio background from <Picture 1>. Remove the chair completely.")
                else:
                    graph["5"]["inputs"]["prompt"] = PROMPTS[case]
                graph["5"]["inputs"]["quality_profile"] = profile
                plan.append((f"context_{case}_2mp_{frames}frames_104", graph, "street" if case == "street" else "portrait", 2, 104))
    if args.cases:
        missing = set(args.cases) - {item[0] for item in plan}
        if missing:
            raise ValueError(f"Cases not present in this stage: {sorted(missing)}")
        plan = [item for item in plan if item[0] in args.cases]
    for key, graph, case, mp, seed in plan:
        if key in results and not args.rerun and (not args.retry_failed or results[key]["status"]["status_str"] == "success"):
            continue
        previous = results.get(key)
        for node in graph.values():
            if node["class_type"] == "LoraLoaderModelOnly":
                wanted = node["inputs"]["lora_name"].replace("\\", "/")
                node["inputs"]["lora_name"] = next((name for name in lora_names if name.replace("\\", "/") == wanted), wanted)
        if mp is not None:
            width, height = size(graph, case, mp)
            graph["6"]["inputs"]["noise_seed"] = seed
            graph["90"] = {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": key + "_frames"}}
            graph["13"]["inputs"]["filename_prefix"] = key
        else:
            width = height = None
            for node in graph.values():
                if node["class_type"] == "SaveImage":
                    node["inputs"]["filename_prefix"] = key
        (args.directory / f"{key}.json").write_text(json.dumps(graph, indent=2))
        print(f"RUN {key}", flush=True)
        try:
            result = execute(args.url, graph, 1200)
        except RuntimeError as exc:
            result = {"status": {"status_str": "validation_error", "error": str(exc)}}
        result.update(width=width, height=height)
        if previous:
            result["previous_attempt"] = previous
        results[key] = result
        temporary = result_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=2))
        temporary.replace(result_path)
        print(f"DONE {key}: {result['status']['status_str']} {result.get('execution_seconds')}s", flush=True)


if __name__ == "__main__":
    main()
