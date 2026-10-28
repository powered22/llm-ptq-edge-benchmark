"""Helper: build llm-compressor scheme kwargs, optionally with a custom group size.

Presets such as W4A16 use group_size=128, which fails for models whose hidden size is not a multiple of 128
(for example SmolLM2-360M, hidden size 960). Passing group_size=64 builds the same symmetric INT4 weight scheme
with 64-wide groups instead.
"""
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme


def scheme_kwargs(scheme: str, group_size: int | None = None) -> dict:
    if group_size is None:
        return {"scheme": scheme}
    bits = {"W4A16": 4, "W8A16": 8}[scheme]
    weights = QuantizationArgs(
        num_bits=bits, type="int", symmetric=True, strategy="group", group_size=group_size
    )
    return {"config_groups": {"group_0": QuantizationScheme(targets=["Linear"], weights=weights)}}
