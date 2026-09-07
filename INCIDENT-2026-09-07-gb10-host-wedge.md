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
build.** Gemma-4-E4B (util 0.60) and Gemma-4-31B (util 0.88, both NVFP4 and bf16 KV, 8/8 needles each) had
already completed successfully; the host died during the 26B-A4B run that followed.

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

## Follow-up: what happened with #49760 applied

Three further attempts at Gemma-4-26B-A4B-it (49.44 GiB of bf16 weights, calibrated NVFP4 KV) on the
same GB10, this time on a build **with** #49760 (`test/nvfp4-both-fixes-plus-49760`, ef03e286d),
`VLLM_UNIFIED_MEMORY_HOST_RESERVE_GB=16`, and an external watchdog killing vLLM if `MemAvailable`
fell below 4-8 GiB:

| attempt | util | outcome |
|---|---|---|
| 1 | 0.88 | cap fired, then avail → 3 GiB after torch.compile; watchdog killed vLLM |
| 2 | 0.60 | avail → 5 GiB (contaminated: another service was loading concurrently) |
| 3 | 0.60, clean box | avail → 3 GiB; watchdog killed vLLM |

The cap works and is visible in the log:

```
WARNING [mem_utils.py:142] Integrated (unified-memory) GPU detected: capping the memory budget
from 105.28GiB to 96.93GiB to keep 16.0GiB free for the OS.
```

Two things worth noting for #46307 / #49760:

1. **The cap governs the profiled budget, not the resident footprint.** At util 0.60 the budget is
   ~71 GiB, well clear of the 49 GiB of weights, and the host *still* ran out. So the binding
   constraint is not the number the cap adjusts.
2. **The collapse is fast.** A 10-second memory sampler recorded a floor of 18 GiB while the
   2-second watchdog caught the same run hitting 3 GiB — available memory fell more than 20 GiB
   inside one sampling window, consistent with a single large allocation late in startup
   (KV-cache allocation or graph capture) rather than a gradual climb.

The practical read: 26B-A4B does not come up on a 119 GiB Spark today, with or without #49760, which
matches vllm-project/vllm#46329's own statement that this configuration is blocked in the MoE-weight
path rather than the KV path. #49760 remains necessary — it is what turns "0.88 wedges the host" into
"0.88 is capped to something survivable" — but it is not sufficient for this model on this box.

Every attempt after the first incident kept the host alive, because of the external watchdog rather
than anything in vLLM. That is the gap worth closing: the promised clean startup failure never fired.

## Operational note for anyone testing on a Spark

Treat `gpu_memory_utilization` on unified memory as "fraction of the machine you are taking away from
the OS". Until #49760 lands, keep a large explicit reserve and size from *free* memory rather than
total. The failure mode is losing the box, not an exception.
