# v23 comparison sheets

[Test conditions, prompts, recipes and limitations](../../../docs/validation-v23.md). [Measurements](measurements.json) cover all 138 final cases. These are generated test subjects, not personal photos. Sheets are resized JPEG previews; they do not establish full-resolution detail quality.

All final saved comparison images are included, without best-seed selection. Labels identify canvas, recipe, seed and server execution time. Timing includes model switching and cached conditioning where applicable. It is not isolated sampler time.

| Group | Pages |
|---|---|
| Portrait generation | [1](sheet_t2i_portrait_13_0.jpg), [2](sheet_t2i_portrait_13_1.jpg) |
| Bookshop generation | [1](sheet_t2i_street_13_0.jpg), [2](sheet_t2i_street_13_1.jpg) |
| T-pose generation | [1](sheet_t2i_pose_13_0.jpg), [2](sheet_t2i_pose_13_1.jpg) |
| Jacket editing | [1](sheet_edit_wardrobe_13_0.jpg), [2](sheet_edit_wardrobe_13_1.jpg) |
| Two-reference pose editing | [1](sheet_edit_pose_transfer_13_0.jpg), [2](sheet_edit_pose_transfer_13_1.jpg) |
| True single-frame workflows | [1](sheet_single_13_0.jpg), [2](sheet_single_13_1.jpg) |
| Nine/twenty-frame context | [1](sheet_context_13_0.jpg) |
| Qwen refinement, raw and tone-locked | [1](sheet_refiner_all_0.jpg) |
| Decoder comparisons | [1](sheet_decoder_all_0.jpg), [2](sheet_decoder_all_1.jpg), [3](sheet_decoder_all_2.jpg), [4](sheet_decoder_all_3.jpg), [5](sheet_decoder_all_4.jpg), [6](sheet_decoder_all_5.jpg), [7](sheet_decoder_all_6.jpg) |

Editing uses the base portrait at 1 MP, seed 104; the second reference is the corresponding generated T-pose. REF and semantic recipes receive both images; legacy FL receives only the source and text pose instruction.

Decoder output IDs: 13 = official temporal VAE, 310/311 = Mamad8 slices 0/1, 320/321 = converted 500K slices 0/1. Sampling is shared within each five-output comparison. The displayed runtime is for the entire comparison, not an individual decoder. The corrected 500K still produces patch artifacts in some images; it is not the recommended default.

Refiner output 13 is raw Qwen, output 16 is tone-locked. Originals are the Turbo 8 generation examples at 2/4 MP, seed 104. Both outputs preserve the source dimensions; no universal detail improvement or 2–4 second latency is claimed.
