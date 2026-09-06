# Gemma-4 NVFP4 KV on consumer Blackwell: repro and verification

Companion to vllm-project/vllm#46329 (NVFP4 KV cache on sm120/sm121 via FlashInfer FA2) and
vllm-project/vllm#55559 (KV-sharing layers inherit the target layer's quantized-KV scales).

Gemma-4 models served with `--kv-cache-dtype nvfp4` on RTX 5090 (sm120) and GB10 / DGX Spark (sm121)
produced garbage under the default compilation config, even for a 34-token prompt, while `--enforce-eager`
was correct. This repo has the two root causes, the probes that isolate them, the numbers, and wheels of
the fixed branch so anyone with a 5090 or a GB10 can check without building vLLM.

## The two bugs

**1. FULL cudagraph decode capture around the VO-split prefill wrapper** (fixed in #46329, commit 8bc0d3b10).
Gemma-4's global-attention layers have `head_dim` 512. On the FA2 NVFP4 path vLLM runs them as two passes with
`head_dim_vo` 256 (the "VO split") and, because FlashInfer's decode wrapper has no `head_dim_vo`, sets
`reorder_batch_threshold = 0`: every request, decode included, goes through the per-step-planned prefill wrapper,
which has no cudagraph buffers. `get_cudagraph_support()` still advertised single-token decode capture for sm12x
NVFP4, so the runner captured FULL decode graphs around that wrapper and replayed stale plan data. Every Gemma-4
(E2B, E4B, 12B, ...) hits it; head-256 models (Gemma 3, Qwen) use the real decode wrapper and do not.
Fix: refuse capture (`AttentionCGSupport.NEVER`) for sm12x NVFP4 groups with `head_size > 256`; the runner then
uses PIECEWISE graphs for the model.

**2. KV-sharing layers dequantize with scale 1.0** (fixed in #55559).
Gemma-4 E-series share KV across layers (`num_kv_shared_layers`, 20 of E2B's 35 layers). A sharing layer reads
the cache its target wrote with the target's `k_scale`/`v_scale`; checkpoints carry no scales for sharing
layers (they have no K/V projections), so they kept the 1.0 default. FlashInfer folds `k_scale` into the softmax
scale and applies `v_scale` to the output, so attention on those 20 layers ran about 5x too sharp with V about
0.4x. This is a quality degradation, not the collapse; it also applies to fp8 KV.
Fix: copy the target's scales onto the sharing layer when the shared cache is bound.

## What is NOT the cause (ruled out along the way)

- The FlashInfer kernel and its NVFP4 split-KV gate: the failure is identical with the gate on or off, with stock
  FlashInfer 0.6.18, with the merged #3684 head, and with older branches; TechPrototyper's 512/256 ladder is 72/72 clean.
- The vLLM store kernel's scale layout (linear on sm12x); its unit tests pass on GB10.
- The checkpoint: public "NVFP4" Gemma-4 checkpoints have no KV scales at all and are expected to produce garbage
  with NVFP4 KV (`Using KV cache scaling factor 1.0` in the log). You need a checkpoint with a calibrated
  `kv_cache_scheme: {num_bits: 4}`; `scripts/calibrate.py` makes one from the bf16 model in about two minutes.

## Reproduce

```bash
# 1. calibrated checkpoint (llm-compressor 0.12, compressed-tensors 0.17.1)
python scripts/calibrate.py --model google/gemma-4-E2B-it --out ./gemma4-e2b-nvfp4kv

# 2. serve with NVFP4 KV (default compilation config)
scripts/serve_gemma_nvfp4.sh ./gemma4-e2b-nvfp4kv

# 3. needle probe, chat and raw formats, 0..1600 filler words
python scripts/probe_len_format.py http://127.0.0.1:8001
```

On an unfixed build every line is `MISS` (answers like `2024`, `314159`, or `stonesstonesstones`). Add
`--enforce-eager` or `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` and it passes: that is bug 1.

`scripts/dump_scales.py MODEL nvfp4` prints each attention layer's effective `k`/`v` scale from a live engine
(bug 2 shows as `k=1.0000 v=1.0000` on layers 15-34 with `target=...layers.13/14...`).
`scripts/compare_layers.py MODEL OUT` captures per-layer attention outputs under bf16 KV and NVFP4 KV on the same
prompt and prints the relative error per layer (bug 2 shows as a jump at the first sharing layer).
Both need `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (they use `collective_rpc` with a function).

## Results

RESULTS_PLACEHOLDER

## Wheels

Release assets are vLLM built from branch `test/nvfp4-both-fixes` (jethac/vllm, commit a421d4a13 = #46329 head
8bc0d3b10 + #55559's c474f4722 cherry-picked), torch 2.13.0+cu130, FlashInfer 0.6.18:

- `vllm-*-cp312-*-linux_x86_64.whl`: sm_120 (RTX 5090), built on a RunPod 5090.
- `vllm-*-cp312-*-linux_aarch64.whl`: sm_120 + sm_121 (GB10 / DGX Spark), built on the Spark.

```bash
uv venv --python 3.12 && . .venv/bin/activate
uv pip install "torch==2.13.0" --torch-backend=cu130
uv pip install <wheel> --extra-index-url https://flashinfer.ai/whl/
```

## Layout

- `scripts/` the probes and the calibration script (`prefix_cache_needle.py` used in the thread is maxpla3's, see #46329).
- `results/` raw probe / dump / compare outputs per machine and build.
