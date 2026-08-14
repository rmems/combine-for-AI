# combine-for-AI

Neutral benchmark harness for **hybrid MoE/SNN quantization** experiments.

Public slug: **`combine-for-AI`**. Python package name remains `nfl-combine-for-ai` for now.

## Role

Owns:

- evaluation scripts and dataset loaders
- artifact **manifest** ingestion (magere-brug handoff JSON/YAML)
- metric collection and CSV/JSON/Markdown reports
- GOZ1 / SAAQ evaluation path for **`rmems/grok-ozempic`** packs

Does **not** own:

- quantization kernels or GOZ1 packing (`grok-ozempic`, `myelin-accelerator`)
- precision-tier inventory (`xai-dissect`)
- SAAQ calibration loops (`corinth-canal`, `magere-brug`)

**SAAQ** = Spiking Adaptive Activity Quantization.

## Ecosystem

| Repo | Owns |
|------|------|
| [xai-dissect](https://github.com/rmems/xai-dissect) | Tensor inventory; tiers `preserve` / `fp16` / `ternary_snn` |
| [grok-ozempic](https://github.com/rmems/grok-ozempic) | Streaming quant → **GOZ1** container; route-preservation experiments |
| [magere-brug](https://github.com/rmems/magere-brug) | SAAQ recipes + handoff manifests |
| [corinth-canal](https://github.com/rmems/corinth-canal) | SAAQ latent telemetry runs |
| [myelin-accelerator](https://github.com/Limen-Neural/myelin-accelerator) | CUDA GEMV/GEMM for ternary/GOZ1 |
| **combine-for-AI** | Ingest, metrics, reports, comparison |

GOZ1 container format SoT: [`grok-ozempic/docs/goz1-format.md`](https://github.com/rmems/grok-ozempic/blob/main/docs/goz1-format.md) (v3 current: per-tensor `scale`, `gif_threshold`, `threshold_abs`).

## Quickstart

Mock sample benchmark:

```bash
uv sync
python scripts/benchmark.py --config configs/benchmark.sample.json
```

Manifest-driven artifact smoke (GGUF / HF / AWQ / GPTQ / **GOZ1**):

```bash
./scripts/run_smoke_benchmark.sh configs/manifests/safetensors_hf.sample.json
```

`configs/manifests/goz1.sample.json` validates the GOZ1 schema. Smoke on that file **fails closed** until `generated_artifacts[].path` points at a real `.goz1` pack (it will not silently fall back to HF).

Reports land under `reports/json` and `reports/csv` by default.

## GOZ1 / MoE-SNN metrics

Beyond accuracy/perplexity, the report schema supports (nullable) fields used by grok-ozempic science:

- `route_top1_agreement` / `route_top2_agreement`
- `block_output_cosine`, `resid_in_drift`, `block_index`
- `expert_load_js` (secondary — do not gate alone)
- `scale_source` (`pack_v3` / `pack_v2` / `legacy_oracle`)
- `goz1_version`, `sparsity`

Smoke runs on GOZ1 packs attach header fields: `goz1_version`, `goz1_tensor_count`, `goz1_scale_source`.

Import grok-ozempic experiment JSON (multiblock `metrics.json` or route-preservation reports) into combine rows:

```bash
python scripts/import_goz_experiment.py \
  --input /path/to/grok-ozempic/reports/.../metrics.json \
  --output-dir reports \
  --arms expert_only,fp16_control
```

Writes `reports/json/<run_id>.goz-import.json` and `reports/csv/<run_id>.goz-import.csv` with route/residual fields populated (issue **#22**).

## Tracking

Primary board: [GitHub issues](https://github.com/rmems/combine-for-AI/issues) — epic **#20** (GOZ1 / MoE-SNN evaluation readiness for grok-ozempic).

## Docs

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
