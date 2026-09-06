import sys, json, glob, os, re
from safetensors.torch import load_file, save_file
ckpt = sys.argv[1]
BAD = re.compile(r"(audio_tower|vision_tower|vision_model|multi_modal|mm_)")
for sf in glob.glob(os.path.join(ckpt, "*.safetensors")):
    t = load_file(sf)
    drop = [k for k in t if ("k_scale" in k or "v_scale" in k) and BAD.search(k)]
    if drop:
        for k in drop: del t[k]
        save_file(t, sf, metadata={"format": "pt"})
        print("%s: stripped %d non-text scale params" % (os.path.basename(sf), len(drop)))
idx = os.path.join(ckpt, "model.safetensors.index.json")
if os.path.exists(idx):
    d = json.load(open(idx)); wm = d.get("weight_map", {})
    bad = [k for k in wm if ("k_scale" in k or "v_scale" in k) and BAD.search(k)]
    for k in bad: del wm[k]
    json.dump(d, open(idx, "w"))
    print("index: dropped %d" % len(bad))
