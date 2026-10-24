"""Shared names for unqualified GOZ1 rendering inputs."""

from __future__ import annotations

from combine_for_ai.goz_import import SCHEMA_MULTIBLOCK_V1, SCHEMA_ROUTE_PRESERVATION_V1


SUPPORTED_VERSIONS = (1, "1", "v1", "1.0")
GOZ_SCHEMAS = {SCHEMA_MULTIBLOCK_V1, SCHEMA_ROUTE_PRESERVATION_V1}
ARMS = ("expert_only", "fp16_control")
# Use declared GOZ1 units. The route schema's undeclared JS log base is rewritten
# when a record is built. Legacy speed/VRAM presets are not in this allowlist.
GOZ_METRICS = {
    "route_top1_agreement": ("fraction", "Top-1 router agreement with FP16", "router"),
    "route_top2_agreement": (
        "fraction",
        "Top-2 router set agreement with FP16",
        "router",
    ),
    "block_output_cosine": (
        "cosine",
        "Block output cosine similarity with FP16",
        "block",
    ),
    "resid_in_drift": (
        "relative norm",
        "Residual stream drift relative to FP16",
        "block",
    ),
    "expert_load_js": ("bits", "Expert-load Jensen-Shannon divergence", "router"),
}
RAW_METRIC_KEYS = (
    "router_top1_agreement",
    "router_top2_set_agreement",
    "router_topk_set_agreement",
    "block_output_cosine",
    "residual_drift_relative_norm",
    "expert_load_js_bits",
)
GATE_ALIASES = {
    "route_top1_agreement": "router_top1_agreement",
    "route_top2_agreement": "router_top2_set_agreement",
    "resid_in_drift": "residual_stream_drift",
    "expert_load_js": "expert_load_js_divergence",
}
