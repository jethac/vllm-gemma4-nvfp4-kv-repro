"""Which offline LLM configuration reproduces the server-side Gemma-4 NVFP4 cudagraph failure?
usage: e2e_variants.py MODEL VARIANT   (VARIANT in: batch, seq, seq_defaultcap, seq_maxseqs256, seq_nopc)"""
import random, sys
from vllm import LLM, SamplingParams
model, variant = sys.argv[1], sys.argv[2]
WORDS = "the river runs quietly between old stones and the wind carries dust over the fields while travellers rest under tall trees".split()
def case(n, seed):
    rng = random.Random(seed); needle = str(rng.randint(10000, 99999))
    f = lambda k: " ".join(rng.choice(WORDS) for _ in range(k))
    return f"{f(n//2)} The secret code is {needle}. {f(n-n//2)}\n\nWhat is the secret code? Answer with just the number.", needle
kw = dict(kv_cache_dtype="nvfp4", max_model_len=4096, gpu_memory_utilization=0.6, enforce_eager=False, enable_prefix_caching=True, max_num_seqs=2,
          compilation_config={"cudagraph_capture_sizes": [1, 2]})
if variant == "seq_defaultcap": kw.pop("compilation_config")
if variant == "seq_maxseqs256": kw.pop("compilation_config"); kw["max_num_seqs"] = 256
if variant == "seq_nopc": kw["enable_prefix_caching"] = False
llm = LLM(model=model, **kw); tok = llm.get_tokenizer(); sp = SamplingParams(temperature=0, max_tokens=16)
cases = [case(n, n + 1) for n in (0, 50, 200, 800)]
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q, _ in cases]
if variant == "batch":
    outs = [o.outputs[0].text for o in llm.generate(prompts, sp)]
else:
    outs = [llm.generate([p], sp)[0].outputs[0].text for p in prompts]
hits = sum(needle in t for (_, needle), t in zip(cases, outs))
for (_, needle), t in zip(cases, outs): print(f"  needle={needle} got={t.strip()[:40]!r}")
print(f"VARIANT {variant}: {hits}/{len(cases)} hits")
