# Architecture

## Ecosystem boundaries

```text
xai-dissect ──manifests──► grok-ozempic ──GOZ1 packs──► combine-for-AI (eval/report)
                                 │                            ▲
                                 ▼                            │
                          myelin-accelerator            magere-brug handoff
                          (CUDA kernels)                corinth-canal telemetry
```

| Component | Responsibility |
|-----------|----------------|
| Manifest ingestion | Validate magere-style handoff JSON/YAML; dispatch by artifact format |
| GOZ1 header sniff | Magic/version/tensor_count only — no dequant (format SoT in grok-ozempic) |
| Benchmark runner | Mock (CI) + future import adapters for grok-ozempic experiment JSON |
| Telemetry | Local GPU snapshot + optional corinth/myelin overlay |
| Reports | JSON/CSV (+ markdown generators); MoE/SNN fields nullable |

**combine-for-AI does not** pack GOZ1 files, run Grok-1 residual experiments, or own CUDA kernels.

## Manifest ingestion

### Manifest format (JSON or YAML)

- `manifest_version`: fixed `"1.0.0"`
- `model_name`: display name
- `model_family`: optional (e.g. `grok`)
- `source_artifact`:
  - `format`: `gguf`, `safetensors`, `hf`, `pytorch`, `onnx`, `myelin`, **`goz1`**
  - `path`, `hf_repo_id` / `hf_revision`, `url`, `checksum_sha256`, `parameter_count`, `moe_layout`
- `generated_artifacts[]`:
  - `format` includes **`goz1`**, `awq`, `gptq`, `myelin`, …
  - `status`: `success`, `failed`, `partial`, `planned`, `skipped`
  - optional nested **`goz1`**: `container_version`, `packing_scheme`, `gif_threshold`, tensor counts, `scale_source`
- `backend_compatibility`: flags including `goz1`, `myelin_accelerator`, …
- `saaq_metadata`: routing / route-agreement fields for SAAQ research
- `benchmark_linkage`: combine run id + optional `grok_ozempic_report_path`

### GOZ1 notes

- Format SoT: `rmems/grok-ozempic/docs/goz1-format.md`
- Supported header versions: **1, 2, 3** (v3 current write path)
- `scale_source`: `pack_v3` | `pack_v2` | `legacy_oracle` | `unknown`
- `sniff_goz1_header(path)` validates magic `GOZ1` and version only

### Dispatch logic

`dispatch_artifact(manifest)`:

1. First generated artifact with status `success` | `partial` | `planned` → `generated_<format>` (e.g. `generated_goz1`)
2. Else source `gguf` → `gguf`
3. Else source `safetensors` / `hf` → `safetensors_hf`
4. Else source `goz1` → `goz1`
5. Else `unknown`

### Example manifests

- `configs/manifests/goz1.sample.json` — GOZ1 success artifact + nested metadata
- `configs/manifests/grok_planning.sample.json` — planned GOZ1 + myelin
- `configs/manifests/gguf.sample.json` / `safetensors_hf.sample.json`
- YAML samples for human-edited configs (`gguf.sample.yaml`)

## Artifact smoke

Entrypoints:

- `scripts/run_smoke_benchmark.sh <manifest> [args…]`
- `scripts/run_artifact_smoke.py --manifest <path>`

Selection priority for **existing** generated artifacts: **GOZ1 → AWQ → GPTQ**, then source GGUF / HF / GOZ1.

GOZ1 success path uses quantization profile **`saaq`** and attaches header fields to the report row.

## Metrics

### LLM baseline

`accuracy`, `perplexity`, `throughput`, `latency_ms`, `vram_gb`, `routing_entropy`, `spike_density`

### MoE / SNN (nullable)

| Field | Meaning |
|-------|---------|
| `route_top1_agreement` | Router top-1 match vs FP ref |
| `route_top2_agreement` | Router top-2 match vs FP ref |
| `block_output_cosine` | Block output cosine vs FP ref |
| `resid_in_drift` | Residual-input drift (multi-block coupling) |
| `block_index` | Block id in a chain |
| `expert_load_js` | Expert-load JS divergence (secondary) |
| `scale_source` | How ternary α was obtained |
| `goz1_version` | Container version |
| `sparsity` | Ternary zero fraction when known |

Science note (from grok-ozempic #61/#64/#68): expert-only ternary is routing-safe **within** a block; multi-block residual drift can collapse later routing. Cosine-only gates are insufficient.

## grok-ozempic experiment import

`combine_for_ai.goz_import` normalizes upstream experiment JSON without re-running quant or residual harnesses.

| Kind | Detection | Schema id |
|------|-----------|-----------|
| Multiblock metrics | `chain.per_block` | `grok_ozempic.multiblock_metrics.v1` |
| Route preservation | `pilot` + `summary` | `grok_ozempic.route_preservation.v1` |

CLI: `scripts/import_goz_experiment.py`. Output rows map:

- `router_top1_agreement` → `route_top1_agreement`
- `router_top2_set_agreement` → `route_top2_agreement`
- `block_output_cosine` → `block_output_cosine`
- `residual_drift_relative_norm` / `residual_stream_drift` → `resid_in_drift`
- `expert_load_js_bits` / `expert_load_js_divergence` → `expert_load_js`
- pack provenance → `scale_source`, `goz1_version`, `sparsity`

Payload includes `benchmark_linkage.grok_ozempic_report_path` and optional decision/provenance.

## Telemetry

See `benchmarks/telemetry.py`: `SystemSnapshot`, `GPUMetrics` (optional `pynvml`), `RoutingMetrics`, upstream merge for corinth-canal / myelin-accelerator.

## Quantization registry

Supported profiles today: `fp16`, `awq`, `gptq`, `gguf`, **`ternary`**, **`saaq`** (GOZ1 path).

`saaq` is for evaluation/import of Spiking Adaptive Activity Quantization artifacts — kernels live upstream.

## CI

- `ci.yml` — pytest + smoke help
- `manifest-ingestion.yml` — example manifests + dispatch
- `mini-eval-smoke.yml` — tiny mock benchmark
- `benchmark-smoke.yml` — manual small benchmark

## Tracking

Epic: GitHub **#20** — GOZ1 / MoE-SNN evaluation readiness for grok-ozempic.
