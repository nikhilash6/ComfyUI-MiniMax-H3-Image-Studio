# Example workflows

- `ui/`: canvas workflow JSON
- `png/`: workflow preview with embedded UI and API metadata
- `api/`: prompt JSON for API clients

Open files from `ui/` or `png/` in ComfyUI. Files from `api/` do not contain a canvas layout.

Use MiniMax H3 Image Studio v23 or newer for this set. Restart ComfyUI and reopen the workflow after updating the node package. Saved canvases do not update automatically.

| File stem | Workflow |
|---|---|
| `H3_IMAGE_GENERATE` | Start here: eight-step FL2VA generation, about 1 MP |
| `H3_IMAGE_EDIT` | Start here: eight-step REF2VA editing, about 1 MP, source aspect ratio |
| `H3_IMAGE_DRAFT` | Four-step FL2VA v1.2 generation |
| `H3_EDIT_DRAFT` | Four-step REF2VA editing |
| `H3_T2I` | FL2VA text-to-image |
| `H3_T2I_SINGLE` | Experimental one-frame text-to-image with the H3 image VAE |
| `H3_I2I` | FL2VA image-to-image |
| `H3_I2I_SINGLE` | Experimental one-frame image-to-image through reference conditioning |
| `H3_REFERENCE_EDIT` | REF2VA reference editing |
| `H3_REFERENCE_SINGLE` | Experimental true one-frame reference generation |
| `H3_I2I_TURBO` | FL2VA image-to-image with the official Turbo v1.0 eight-step adapter |
| `H3_DETAIL_REFINER` | Optional four-step Qwen Image Edit 2511 generative detail pass for any finished H3 image |

Image-to-image and reference-edit workflows require an image in every `Load Image` node.
The refiner requires a finished image. Its `Load Image` can be replaced with the output of any H3 `Single Image Output` node. The Qwen pass works on a two-megapixel copy, while `Detail Tone Lock` restores the exact source dimensions.
