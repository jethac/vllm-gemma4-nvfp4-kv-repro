"""Per-layer attention-output divergence between two engine configs on one fixed prompt.
usage: compare_layers.py MODEL OUT_PREFIX  (runs bf16/default backend, then nvfp4/flashinfer; writes per-layer stats)"""
import sys, json, random, torch, gc
from vllm import LLM, SamplingParams
model, outp = sys.argv[1], sys.argv[2]
WORDS = "the river runs quietly between old stones and the wind carries dust over the fields while travellers rest under tall trees".split()
rng = random.Random(7); body = " ".join(rng.choice(WORDS) for _ in range(300))
prompt = body + " The secret code is 48213. " + " ".join(rng.choice(WORDS) for _ in range(300)) + "\n\nQ: What is the secret code? Answer with just the number.\nA:"

def install(worker):
    ctx = worker.vllm_config.compilation_config.static_forward_context
    worker._cap = {}
    def mk(n):
        def hook(mod, inp, out):
            if n not in worker._cap:  # first call = prefill
                worker._cap[n] = out.detach().float().cpu()
        return hook
    for name, m in ctx.items():
        if hasattr(m, "kv_sharing_target_layer_name") and hasattr(m, "_k_scale_float"):
            m.register_forward_hook(mk(name))
    return len(worker._cap)
def fetch(worker):
    return {n: t for n, t in worker._cap.items()}

def run(tag, **kw):
    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.40, enforce_eager=True, language_model_only=True, enable_prefix_caching=False, **kw)
    llm.collective_rpc(install)
    out = llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0))[0].outputs[0].text
    caps = llm.collective_rpc(fetch)[0]
    print(f"[{tag}] answer={out!r} layers captured={len(caps)}", flush=True)
    del llm; gc.collect(); torch.cuda.empty_cache()
    return out, caps

ans_a, A = run("bf16-default", kv_cache_dtype="auto")
ans_b, B = run("nvfp4-flashinfer", kv_cache_dtype="nvfp4")
import re
def L(n): m = re.search(r"layers\.(\d+)\.", n); return int(m.group(1)) if m else -1
rows = []
for n in sorted(A, key=L):
    a, b = A[n], B.get(n)
    if b is None or a.shape != b.shape: rows.append((L(n), n, None)); continue
    rel = ((a - b).norm() / (a.norm() + 1e-6)).item(); cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rows.append((L(n), n, rel, cos))
    print(f"L{L(n):2d} rel_err={rel:.4f} cos={cos:.4f} {n}", flush=True)
json.dump({"answers": [ans_a, ans_b], "rows": rows}, open(outp + ".json", "w"), indent=1)
print("COMPARE_DONE")
