"""Build paired four- and eight-step image workflows from the base graphs."""
import copy
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
    workflow["4"]["inputs"]["resolution_profile"] = "native detail | 0.98 MP"
    workflow["13"]["inputs"]["filename_prefix"] = target
    if family == "ref2v":
        del workflow["14"]
        inputs = workflow["5"]["inputs"]
        del inputs["reference_image_2"]
        inputs.update(
            edit_instruction="Change the main subject to blue. Keep the background and framing unchanged.",
            reference_detail="match_generation_area", reference_transport="native",
        )
    else:
        workflow["4"]["inputs"]["aspect_ratio"] = "1:1 square"
        workflow["5"]["inputs"]["prompt"] = (
            "Studio photograph of an adult woman wearing a green wool jacket, "
            "natural skin texture, soft light from the left, plain dark gray background."
        )
    (repo / "examples/api" / f"{target}_API.json").write_text(json.dumps(workflow, indent=2) + "\n")
    draft = copy.deepcopy(workflow)
    draft_target = "H3_IMAGE_DRAFT" if family == "fl2v" else "H3_EDIT_DRAFT"
    draft["15"]["inputs"]["lora_name"] = (
        "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"
        if family == "fl2v" else "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
    )
    draft["8"]["inputs"]["sampling_profile"] = (
        "FL2VA Turbo v1.2 768p | 4 steps" if family == "fl2v" else "REF2VA Turbo v0.1 | 4 steps"
    )
    draft["13"]["inputs"]["filename_prefix"] = draft_target
    (repo / "examples/api" / f"{draft_target}_API.json").write_text(json.dumps(draft, indent=2) + "\n")
