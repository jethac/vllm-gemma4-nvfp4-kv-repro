# Incident: `gpu_memory_utilization=0.88` wedged a GB10 host (unified memory)

Field evidence for vllm-project/vllm#46307 and the fix in vllm-project/vllm#49760.

## What happened

2026-09-07 ~03:32 UTC. Re-running the Gemma 4 family under the default compilation config on a
GB10 / DGX Spark (sm121, 119 GiB usable unified memory), serving with:

```
vllm serve <gemma-4-31B-it, calibrated NVFP4 KV> --kv-cache-dtype nvfp4 \
  --gpu-memory-utilization 0.88 --max-model-len 8192 --enable-prefix-caching --language-model-only
```

vLLM build: `jethac/vllm` a421d4a13 (= #46329 head + #55559). **#49760 was deliberately not in this
build.** Gemma-4-E4B (util 0.60) and 31B NVFP4 (util 0.88) had already completed successfully;
the host died during the run that followed.

The host stopped answering SSH and ICMP entirely. Tailscale still listed the node as `active` with
`tx … rx 0` — packets going out, nothing coming back. Recovery required a physical power cycle.

## Why

`gpu_memory_utilization` is expressed against *total device memory*. On an integrated GPU the device
total **is** the pool the operating system is using: 0.88 × 119 GiB ≈ 105 GiB claimed, leaving the OS
about 14 GiB, and that was not enough to keep the box alive under a 61 GiB weight load plus KV cache
and activations. On a discrete card the same number is unremarkable, which is exactly the trap — the
value was carried over unchanged from an RTX PRO 6000 (96 GiB discrete) run earlier the same night.

There is no user-visible warning before the host stops responding; the process does not OOM-kill
cleanly, the machine simply stops scheduling.

## What #49760 does about it

`cap_unified_memory_budget()` caps the profiling/KV budget by `available_memory - reserve` on
integrated GPUs only (a no-op on discrete ones), with the floor controlled by a new knob:

```
VLLM_UNIFIED_MEMORY_HOST_RESERVE_GB   # default 8.0
```

If the requested budget would fall below that reserve, startup fails cleanly instead of wedging the
host — which is the behaviour you want: a failed `vllm serve` is recoverable over SSH, a wedged Spark
is not.

## Follow-up planned

Run Gemma-4-26B-A4B-it (bf16 weights, calibrated NVFP4 KV) on the same GB10 twice at a utilization
that reproduces the wedge:

1. build **without** #49760 → expect host wedge (or, at minimum, the unbounded budget)
2. build **with** #49760 (`test/nvfp4-both-fixes-plus-49760`, ef03e286d — pure Python, no rebuild
   needed) → expect a clean cap, or a clean startup failure, and a live host either way

That pair is the evidence #46307 deserves; this incident is only the first half of it.

## Operational note for anyone testing on a Spark

Treat `gpu_memory_utilization` on unified memory as "fraction of the machine you are taking away from
the OS". Until #49760 lands, keep a large explicit reserve and size from *free* memory rather than
total. The failure mode is losing the box, not an exception.
