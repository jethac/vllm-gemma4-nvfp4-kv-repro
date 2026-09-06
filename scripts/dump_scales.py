"""Start an engine and dump every attention layer's effective KV scales (ground truth, not checkpoint values)."""
import sys, json
from vllm import LLM
model = sys.argv[1]; kv = sys.argv[2] if len(sys.argv) > 2 else "nvfp4"
llm = LLM(model=model, kv_cache_dtype=kv, max_model_len=2048, gpu_memory_utilization=float(__import__("os").environ.get("GPU_UTIL", "0.8")), enforce_eager=True,
          language_model_only=True, enable_prefix_caching=False)
def dump(worker):
    ctx = worker.vllm_config.compilation_config.static_forward_context
    out = {}
    for name, m in ctx.items():
        if hasattr(m, "_k_scale_float"):
            out[name] = dict(k=float(m._k_scale_float), v=float(m._v_scale_float), q=float(getattr(m, "_q_scale_float", -1)),
                             target=getattr(m, "kv_sharing_target_layer_name", None), has_param=hasattr(m, "k_scale"),
                             qm=type(getattr(m, "quant_method", None)).__name__, dtype=getattr(m, "kv_cache_dtype", None))
    return out
res = llm.collective_rpc(dump)[0]
def L(n):
    import re; m = re.search(r"layers\.(\d+)\.", n); return int(m.group(1)) if m else -1
for n in sorted(res, key=L):
    r = res[n]; print(f"L{L(n):2d} k={r['k']:.4f} v={r['v']:.4f} q={r['q']:.4f} param={r['has_param']} qm={r['qm']} dtype={r['dtype']} target={r['target']}")
print("SCALES_DUMP_DONE")
