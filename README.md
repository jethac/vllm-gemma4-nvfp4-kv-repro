# Gemma-4 NVFP4 KV on consumer Blackwell: repro and verification

Companion to vllm-project/vllm#46329 (NVFP4 KV cache on sm120/sm121 via FlashInfer FA2),
vllm-project/vllm#55559 (KV-sharing layers inherit the target layer's quantized-KV scales) and
vllm-project/vllm#55976 (the KV-cache writer lays V block scales out linearly on SM120/SM121).

> **2026-09-09.** The writer fix has been split out of #46329 into its own bugfix PR, #55976, because
> upstream's #55908 landed the matching test and consumer Blackwell now fails
> `test_reshape_and_cache_flash[nvfp4]` on `main` without it (24 failed on sm_120 and sm_121; 24 passed
> with the fix). The wheels below are rebuilt from all three PRs on top of current `main`.

Gemma-4 models served with `--kv-cache-dtype nvfp4` on RTX 5090 (sm120) and GB10 / DGX Spark (sm121)
produced garbage under the default compilation config, even for a 34-token prompt, while `--enforce-eager`
was correct. This repo has the root causes, the probes that isolate them, the numbers, and wheels of
the fixed branch so anyone with a 5090 or a GB10 can check without building vLLM.

## The bugs

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

**3. The KV-cache writer lays V block scales out for the wrong reader on SM120/SM121** (fixed in #55976).
`reshape_and_cache_nvfp4_kernel` wrote K block scales linearly and V block scales in the SM100 trtllm-gen
4-token swizzle, unconditionally. Consumer Blackwell reads them through the FlashInfer FA2 paged reader,
which takes scale strides from the SF tensor and reads V linearly, so V dequantized to garbage.
Fix: make the V swizzle conditional on the cache's device (`major < 12`); K and the page layout are unchanged,
so SM100 is bit-identical. Since upstream #55908 landed the matching test, `main` without this fix fails
`test_reshape_and_cache_flash[nvfp4]` on consumer Blackwell: 24 failed / 0 passed on both an RTX PRO 4500
(sm_120, measured by @TensorRaya) and a GB10 (sm_121, measured by @hclsys), 24 passed with the fix. The
B200 job attached to #55908 is SM100, where writer and test agree, so upstream CI does not surface it.

## What is NOT the cause (ruled out along the way)

- The FlashInfer kernel and its NVFP4 split-KV gate: the failure is identical with the gate on or off, with stock
  FlashInfer 0.6.18, with the merged #3684 head, and with older branches; TechPrototyper's 512/256 ladder is 72/72 clean.
- The vLLM store kernel's scale layout (linear on sm12x); its unit tests pass on GB10.
- The checkpoint: public "NVFP4" Gemma-4 checkpoints have no KV scales at all and are expected to produce garbage
  with NVFP4 KV (`Using KV cache scaling factor 1.0` in the log). You need a checkpoint with a calibrated
  `kv_cache_scheme: {num_bits: 4}`; `scripts/calibrate.py` makes one from the bf16 model in about two minutes.

## Reproduce

```bash
# 1. calibrated checkpoint: use the pre-built one, or make your own (llm-compressor 0.12, compressed-tensors 0.17.1, ~2 min on a 5090)
#    https://huggingface.co/jethachan/gemma-4-E2B-it-NVFP4KV-calib
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

Same compiled kernels on each machine; "before" swaps in the four affected Python files from the #46329 head
(b423d395d, no fixes), "after" uses the test branch (a421d4a13, both fixes). Probe: needle at 0/20/50/100/200/400/800/1600
filler words, chat-template and raw-completion prompts, temperature 0, default compilation config unless noted.
Checkpoints calibrated on the spot with `scripts/calibrate.py` (256 samples, seq 2048; 128 for 12B).

| Model / KV | config | RTX 5090 (sm120) | GB10 (sm121) |
|---|---|---|---|
| Gemma-4-E2B, nvfp4 | before, default | 0/8 chat, 0/8 raw | 0/8 chat, 0/8 raw |
| Gemma-4-E2B, nvfp4 | before, `--enforce-eager` | 8/8 chat, 5/8 raw | (8/8 chat on the older checkout) |
| Gemma-4-E2B, nvfp4 | **after, default** | **8/8 chat, 4/8 raw** | **8/8 chat, 4/8 raw** |
| Gemma-4-E2B, bf16 KV (reference) | default | 8/8 chat, 4/8 raw | 8/8 chat, 4/8 raw |
| Gemma-4-12B, nvfp4 | before, default | (0/8 on the older checkout) | 0/8 chat, 0/8 raw |
| Gemma-4-12B, nvfp4 | **after, default** | **8/8 chat**, 0/8 raw | **8/8 chat**, 0/8 raw |
| Gemma-3-1B, nvfp4 (head 256, control) | after, default | 7/8 chat, 8/8 raw | 7/8 chat, 8/8 raw |

Raw-completion misses on 12B are the base model continuing the "Q:/A:" prompt with more questions, same as bf16; the
chat column is the one that measures correctness.

Bug 2 in isolation, Gemma-4-E2B, eager, per-layer attention-output relative error vs bf16 KV (`compare_layers.py`):

| layer | 5090 before | 5090 after | GB10 before | GB10 after |
|---|---|---|---|---|
| 0 | 0.11 | 0.11 | 0.11 | 0.11 |
| 13 (last non-shared full-attn) | 0.21 | 0.21 | 0.22 | 0.22 |
| 15 (first sharing layer) | 0.41 | 0.21 | 0.40 | 0.22 |
| 17 | 0.69 | 0.20 | 0.66 | 0.21 |
| 29 | 0.92 | 0.24 | 0.91 | 0.25 |
| 34 | 0.85 | 0.20 | 0.86 | 0.21 |

`dump_scales.py` before: layers 0-14 carry calibrated scales (k 0.13-0.29, v 2.1-3.7), layers 15-34 are k=v=1.0 with
`target=layers.13/14`. After: 15-34 carry their target's values.

Rest of the Gemma 4 family, RTX PRO 6000 Blackwell (sm120, 96 GB), same released wheel, each model calibrated on
the spot, default compilation config (chat needles):

| model | NVFP4 KV | bf16 KV |
|---|---|---|
| Gemma-4-E4B-it | 8/8 | 8/8 |
| Gemma-4-31B-it | 8/8 | 8/8 |
| Gemma-4-26B-A4B-it (MoE) | 8/8 | 8/8 |

Raw outputs are under `results/rtx5090-sm120-a421d4a13/` and `results/gb10-sm121-a421d4a13/`;
`results/rtxpro6000-sm120-a421d4a13/` has the family re-run; `results/gb10-patched-ce2fece1e/` is the earlier diagnosis run on an older checkout (cudagraph-mode split etc.).

## sm121 (GB10 / DGX Spark) partial run, 2026-09-07

Same fixed build, checkpoints already calibrated on that box, default compilation config:

| model | NVFP4 KV | bf16 KV |
|---|---|---|
| Gemma-4-E4B-it | 8/8 chat | 8/8 chat |
| Gemma-4-31B-it | 8/8 chat | 8/8 chat |
| Gemma-4-26B-A4B-it | does not start on 119 GiB unified memory (see incident note) | — |

The run was cut short when `--gpu-memory-utilization 0.88` — carried over unchanged from a discrete
96 GiB card — wedged the unified-memory host. See
[INCIDENT-2026-09-07-gb10-host-wedge.md](INCIDENT-2026-09-07-gb10-host-wedge.md); it is field evidence
for vllm-project/vllm#46307 / #49760.

## Wheels

Release assets are vLLM built from branch `community/nvfp4-stack` (jethac/vllm), which is current
upstream `main` plus, in order, #55976 (writer), #46329 (NVFP4 KV on sm12x) and #55559 (KV-sharing
scales, cherry-picked since it is still open). torch 2.13.0+cu130, FlashInfer 0.6.18:

- `vllm-*-cp312-*-linux_x86_64.whl`: sm_120 (RTX 5090), built on a RunPod 5090.
- `vllm-*-cp312-*-linux_aarch64.whl`: sm_120 + sm_121 (GB10 / DGX Spark), built on the Spark.

The earlier `v2026.09.06-a421d4a13` release stays up for reproducing the numbers in the tables below,
which were measured against it. Prefer the newest release for actually running anything: it carries the
writer fix as upstream will take it, the NVFP4-only VO-split gating, and the DeviceGuard ordering fix.

```bash
uv venv --python 3.12 && . .venv/bin/activate
uv pip install "torch==2.13.0" --torch-backend=cu130
uv pip install <wheel> --extra-index-url https://flashinfer.ai/whl/
```

## Layout

- `scripts/test_gemma4_nvfp4_kv_sm12x.py` the vLLM e2e regression test (also in the PR). On a GB10 with one
  compiled build, swapping only the four affected Python files: fails on the pre-fix head b423d395d with all four
  needles missed, passes with the fixes.
- `scripts/` the probes and the calibration script (`prefix_cache_needle.py` used in the thread is maxpla3's, see #46329).
- `results/` raw probe / dump / compare outputs per machine and build.
