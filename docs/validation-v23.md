# v23 validation — 12 September 2026

RTX 4090 24 GB, 64 GB RAM, Windows, PyTorch 2.13.0+cu130, ComfyUI 0.34.0 (`12d52794`). Default PyTorch attention unless stated otherwise. Only Image Studio enabled; no cache accelerator or ComfyUI core patch. Models and results were stored on the separate model drive.

This is a bounded regression comparison, not a human preference study. Successful execution means a graph completed and wrote images, not that every instruction was followed. Prompts/seeds were fixed before review, without best-seed selection. Comparison sheets retain all outputs in each group, including defects.

## Coverage

138 final cases completed successfully. This count includes 16 decoder comparisons with five outputs each; it is not 138 independent prompts. The measurement export records dimensions for every saved comparison output.

- Generation: portrait, rainy bookshop with “NORD” sign, standing T-pose; two seeds at approximately 1 and 2 MP. Base 20 steps, FL Turbo 8 steps, FL Turbo v1.2 4 steps, DARE rev2 at 6 steps.
- Editing: yellow jacket to red leather; seated subject to standing T-pose from a second reference. Two seeds at 1 and 2 MP. Base REF2VA, REF Turbo 8/4 steps, semantic-only references, legacy FL2VA. Legacy receives a text pose instruction, not image 2; its reference capability is not equivalent.
- High resolution: portrait/bookshop generation at 4 MP with base/8 steps; jacket/pose editing at 4 MP with REF 8/4 steps.
- Single-frame examples: T2I, I2I and two-image reference editing at 1/2 MP, two seeds.
- Context: 9 and 20 frames at 2 MP for portrait, bookshop and pose transfer.
- Decoders: identical sampled latents, official temporal VAE versus independent slices 0/1 with Mamad8 and converted 500K; portrait, bookshop, jacket edit and pose transfer at 1/2 MP with two seeds.
- Qwen 2511 refiner: portrait/bookshop at 2/4 MP, raw and tone-locked output.

Square sizes: 1024², 1440², 2048². Landscape: 1344×768, 1920×1088, 2688×1536. MP labels are approximate, using ComfyUI's 1024² convention.

## Recipes

Official pruned INT8 ConvRot FL2VA/REF2VA models, NVFP4 AWQ Qwen encoder, FP16 video VAE. Adapter strength 1 unless specified.

| Recipe | Sampler / scheduler | Steps | Video/audio shift |
|---|---|---:|---:|
| Base | RES multistep / simple | 20 | 12/3 |
| FL Turbo v1.0 768p | Euler / simple | 8 | 6/3 |
| FL Turbo v1.2 768p | Euler / simple | 4 | 6/3 |
| REF Turbo v1.0 768p | Euler / simple | 8 | 12/3 |
| REF Turbo v0.1 | Euler / simple | 4 | 12/3 |
| DARE rev2, strength 0.8 | Euler / simple | 6 | 8/3 |

`SINGLE` examples retain their hybrid checkpoint and ER-SDE/sgm_uniform eight-step recipe. They differ in model stack, not just frame count.

## Measured examples

Server execution timestamps include encoding, model loading when needed, decoding and saving all frames. Repeated seeds can reuse conditioning; first runs can include cold loading. Order, OS cache, background CPU work and model switching affect results. These are individual observations, not isolated sampler speedups or guaranteed latency.

| Task | Canvas | Base 20 | Turbo 8 | Turbo 4 |
|---|---|---:|---:|---:|
| Portrait, seed 105 | 1024² | 10.87 s | 7.31 s | 4.41 s |
| Portrait, seed 105 | 1440² | 18.23 s | 13.13 s | 8.68 s |
| Jacket edit, seed 205 | 1440² | 25.04 s | 13.88 s | 13.26 s |
| Pose transfer, seed 205 | 1440² | 34.13 s | 18.34 s | 11.97 s |
| Portrait, seed 104 | 2048² | 50.82 s | 25.13 s | Not run |
| Bookshop, seed 104 | 2688×1536 | 40.31 s | 24.06 s | Not run |
| Jacket edit, seed 204 | 2048² | Not run | 32.46 s | 24.32 s |
| Pose transfer, seed 204 | 2048² | Not run | 40.85 s | 29.74 s |

DARE completed 12 cases. Initial requests were rejected for a Windows LoRA path-separator mismatch; resolving names against `object_info` fixed the benchmark client. Previous failures remain in local attempt history and are not model-quality failures. This small comparison does not establish DARE as superior to matched official adapters.

## Visual findings

- REF2VA visibly applied the jacket/pose changes, including at 4 MP. Faces, folds, framing and lighting were not perfectly preserved. Some outputs have narrow dark edge bars. Arbitrary identity/pose reliability is not established.
- Semantic-only references changed requested traits but sometimes altered appearance/framing more. Native remains default.
- Four-step recipes gave useful drafts at lower cost. Separate workflows pair the correct adapters and schedules.
- Larger canvases changed composition at the same seed. Some portraits missed hands-on-knees placement; small fingers and book lettering remain imperfect. The four-letter shop sign was legible in the reviewed comparisons, not evidence of general typography accuracy.
- Nine frames introduced side bars in one portrait. Twenty frames altered its expression/pose and took 45.25 s; 20-frame pose transfer took 63.22 s. More context was not consistently better. Five frames remain the starting point.
- True single-frame generation was fast after loading (1 MP portrait second seed: 2.31 s), but one output had large white side borders. Single-frame edits visibly changed the jacket and pose. They remain experimental.
- 500K is not promoted as a quality upgrade. Conversion required QKV head interleaving and SwiGLU half swapping. After correction, patch/geometry defects remained in a generated portrait, including whole-image decoding with the author's pinned Diffusers code. Native temporal VAE remains default. Whole-image decoding is an experimental option; Mamad8 slice decoding retains native tiling by default.
- Qwen refinement preserved exact original dimensions: 1440², 1920×1088, 2048², 2688×1536. Times were 19.27/9.45 s at 2 MP and 20.52/9.43 s at 4 MP (portrait/bookshop, including switching). This does not meet a universal 2–4 second target or establish better fine detail/identity in every image.

[Measurements](../assets/benchmarks/v23/measurements.json) and [all comparison sheets](../assets/benchmarks/v23/README.md) contain generated non-personal material only. Full PNGs and attempt history stay local. [v22 smoke tests](validation-v22.md) remain historical evidence.

## Reproduction

Use a Git checkout: developer benchmark clients are excluded from the Registry runtime package. Start a separate ComfyUI instance on localhost:8191 with `--output-directory <benchmark-root>/images`, then run stages in order:

```bash
python scripts/benchmark_matrix.py t2i --directory <benchmark-root>
python scripts/benchmark_matrix.py edit --directory <benchmark-root>
python scripts/benchmark_matrix.py single --directory <benchmark-root>
python scripts/benchmark_matrix.py stress --directory <benchmark-root>
python scripts/benchmark_matrix.py edit_stress --directory <benchmark-root>
python scripts/benchmark_matrix.py context --directory <benchmark-root>
python scripts/benchmark_matrix.py decoder --directory <benchmark-root>
python scripts/benchmark_matrix.py decoder_edit --directory <benchmark-root>
python scripts/benchmark_matrix.py refiner --directory <benchmark-root>
python scripts/benchmark_matrix.py refiner_stress --directory <benchmark-root>
python scripts/benchmark_gallery.py t2i --directory <benchmark-root>
python scripts/audit_benchmark.py <benchmark-root> --output measurements.json
```

`--recipes` / `--cases` select cases. Completed cases are skipped; `--retry-failed` repeats failures; `--rerun` preserves prior results and repeats selected cases. Use one writer per directory. Inspect the queue before resuming a timeout. Decoder tests need the converted experimental weights.

## Release validation

All 26 automated tests passed. All 12 API/UI/PNG workflow sets passed release validation. All explicit API input values are compared after frontend import, save, reopening and conversion back to a prompt, not just node counts. PNGs embed matching UI and API JSON. Tests cover reference ordering, preservation wording, sampling recipes, one-frame latents, batch decoding, conversion math, tone-lock resolution and artifacts. CI installs CPU PyTorch so runtime tests run rather than being skipped.

Registry v22 was flagged for the developer HTTP client `scripts/benchmark_image_edit.py`; GitHub's upload action still reported success. Developer clients are excluded from v23's Registry package, while remaining on GitHub. Publishing now checks for active Registry status and fails on a flagged release or unconfirmed activation. Upload success alone does not prove Manager availability.

## Sources and untested approaches

- [FL recipes](https://github.com/ModelTC/Minimax-H3-Turbo), [REF eight-step release](https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/51): adapter-specific shifts.
- [FL four-step v1.2](https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/52): announcement emphasizes audio; still-image behavior tested separately here.
- [500K decoder](https://huggingface.co/iamkaikai/MiniMax-H3-Single-Frame-VAE-500K): independent T=1, not a video replacement. Reference probe used pinned Diffusers `9284607295a09f759aadd65ed08f48b35feea6d9`.
- [DARE experiments](https://huggingface.co/silveroxides/MiniMax-H3_tests/discussions/2): exploratory community recipe, not an official still-image ranking.
- [Alibaba PDD](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs) needs a separate parallel-decoding implementation. PDD, FastH3, SageAttention, SLA and DARE rev6 were not benchmarked in this release.
