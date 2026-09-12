import importlib.util
from pathlib import Path
import unittest

try:
    import torch
    import safetensors
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "Conversion needs ComfyUI's torch and safetensors")
class DecoderConversionTests(unittest.TestCase):
    def test_swiglu_conversion_preserves_forward_math(self):
        path = Path(__file__).resolve().parents[1] / "scripts/convert_single_frame_decoder.py"
        spec = importlib.util.spec_from_file_location("decoder_conversion", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        x, weight, bias = torch.randn(2, 8), torch.randn(32, 8), torch.randn(32)
        value, gate = torch.nn.functional.linear(x, weight, bias).chunk(2, dim=-1)
        expected = value * torch.nn.functional.silu(gate)
        native_gate, native_value = torch.nn.functional.linear(
            x, module.pack_swiglu(weight), module.pack_swiglu(bias)).chunk(2, dim=-1)
        self.assertTrue(torch.equal(expected, torch.nn.functional.silu(native_gate) * native_value))

    def test_qkv_layout_matches_native_attention_for_weights_and_bias(self):
        path = Path(__file__).resolve().parents[1] / "scripts/convert_single_frame_decoder.py"
        spec = importlib.util.spec_from_file_location("decoder_conversion", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for shape in ((2048,), (2048, 8)):
            parts = [torch.randn(shape) for _ in range(3)]
            fused = module.pack_qkv(parts).reshape(32, 3 * 64, *shape[1:])
            for index, recovered in enumerate(torch.chunk(fused, 3, dim=1)):
                self.assertTrue(torch.equal(recovered.reshape(shape), parts[index]))
