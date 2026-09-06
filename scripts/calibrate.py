import sys, argparse, subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None
from datasets import load_dataset
from llmcompressor import oneshot

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--nsamples", type=int, default=256)
ap.add_argument("--seqlen", type=int, default=2048)
a = ap.parse_args()

try:
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype="auto", trust_remote_code=True)
except Exception as e:
    print("CausalLM load failed (%s); trying ImageTextToText" % type(e).__name__)
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype="auto", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
_cfg = model.config
_tc = getattr(_cfg, "text_config", None)
if _tc is not None:
    for _attr in ("num_attention_heads", "num_key_value_heads", "head_dim", "hidden_size"):
        _v = getattr(_tc, _attr, None)
        if _v is not None and getattr(_cfg, _attr, None) is None:
            setattr(_cfg, _attr, _v)

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:%d]" % a.nsamples).shuffle(seed=42)
def prep(ex):
    return tok(tok.apply_chat_template(ex["messages"], tokenize=False), max_length=a.seqlen,
               truncation=True, add_special_tokens=False)
ds = ds.map(prep, remove_columns=ds.column_names)
recipe = """
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      ignore: ["lm_head", "re:.*vision.*", "re:.*visual.*", "re:.*multi_modal.*", "re:.*mm_.*", "re:.*audio.*", "re:.*vision_tower.*"]
      kv_cache_scheme:
        num_bits: 4
        type: float
        strategy: tensor
        dynamic: false
        symmetric: true
"""
oneshot(model=model, processor=tok, dataset=ds, recipe=recipe, max_seq_length=a.seqlen,
        num_calibration_samples=a.nsamples, pipeline="basic")
model.save_pretrained(a.out, save_compressed=True)
# Preserve the ORIGINAL tokenizer/processor files VERBATIM. Re-saving them via
# the pinned transformers (tok.save_pretrained / processor.save_pretrained)
# rewrites tokenizer_config.json (adds is_local/local_files_only/
# model_specific_special_tokens), which makes the Gemma-4 audio processor
# truncate audio to ~12 mel frames -> 3 audio tokens vs hundreds of
# placeholders -> _merge_multimodal_embeddings crash. Copy bytes, don't re-save.
import os, shutil
_PROC_FILES = ["tokenizer_config.json", "tokenizer.json", "tokenizer.model",
               "special_tokens_map.json", "processor_config.json",
               "preprocessor_config.json", "chat_template.jinja",
               "generation_config.json"]
def _copy_original(name, out):
    if os.path.isdir(name):
        for fn in _PROC_FILES:
            src = os.path.join(name, fn)
            if os.path.exists(src): shutil.copy(src, os.path.join(out, fn))
    else:
        from huggingface_hub import hf_hub_download
        for fn in _PROC_FILES:
            try: shutil.copy(hf_hub_download(name, fn), os.path.join(out, fn))
            except Exception: pass
_copy_original(a.model, a.out)
# strip the audio/vision KV scales the global kv_cache_scheme adds (vLLM's
# Gemma4AudioAttention/vision attn have no k_scale/v_scale param -> load error).
# check=True so a strip failure is LOUD, not silent (subprocess defaulted to no
# return-code check and could no-op, shipping an unloadable ckpt).
_strip = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strip_nontext_scales.py")
_r = subprocess.run([sys.executable, _strip, a.out])
if _r.returncode != 0:
    raise SystemExit("strip_nontext_scales failed (rc=%d) — ckpt would not load" % _r.returncode)
print("CALIB DONE ->", a.out)
