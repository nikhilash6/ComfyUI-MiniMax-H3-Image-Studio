"""Build the two eight-step entry points from the base API workflows."""
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
for source, target, family in (
    ("H3_T2I", "H3_IMAGE_GENERATE", "fl2v"),
    ("H3_REFERENCE_EDIT", "H3_IMAGE_EDIT", "ref2v"),
):
    workflow = json.loads((repo / "examples/api" / f"{source}_API.json").read_text())
    workflow["15"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["1", 0],
        "lora_name": f"minimax_h3_{family}_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
        "strength_model": 1.0,
    }}
    workflow["8"]["inputs"].update(
        model=["15", 0],
        sampling_profile=f"{'FL2VA' if family == 'fl2v' else 'REF2VA'} Turbo v1.0 768p | 8 steps",
    )
    workflow["4"]["inputs"]["resolution_profile"] = "balanced | 0.70 MP"
    workflow["13"]["inputs"]["filename_prefix"] = target
    if family == "ref2v":
        del workflow["14"]
        inputs = workflow["5"]["inputs"]
        del inputs["reference_image_2"]
        inputs.update(
            edit_instruction="Change the main subject to blue. Keep the background and framing unchanged.",
            reference_detail="match_generation_area", reference_transport="native",
        )
    (repo / "examples/api" / f"{target}_API.json").write_text(json.dumps(workflow, indent=2) + "\n")
