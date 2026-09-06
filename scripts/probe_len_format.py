"""Probe: does Gemma-4 output degrade with prompt LENGTH or with raw-completion FORMAT?
Same needle task via (a) chat template, (b) raw /v1/completions, at increasing filler lengths."""
import json, random, sys, requests
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
WORDS = "the river runs quietly between old stones and the wind carries dust over the fields while travellers rest under tall trees".split()
def filler(n, rng): return " ".join(rng.choice(WORDS) for _ in range(n))
def task(n, seed):
    rng = random.Random(seed); needle = str(rng.randint(10000, 99999))
    body = filler(n // 2, rng) + f" The secret code is {needle}. " + filler(n - n // 2, rng)
    q = "What is the secret code? Answer with just the number."
    return body, q, needle
for n in (0, 20, 50, 100, 200, 400, 800, 1600):
    body, q, needle = task(n, n + 1)
    chat = requests.post(f"{URL}/v1/chat/completions", json={"model": "local-llm", "messages": [{"role": "user", "content": f"{body}\n\n{q}"}], "max_tokens": 16, "temperature": 0}, timeout=600).json()
    ctext = chat["choices"][0]["message"]["content"].strip(); ptoks = chat.get("usage", {}).get("prompt_tokens")
    raw = requests.post(f"{URL}/v1/completions", json={"model": "local-llm", "prompt": f"{body}\n\nQ: {q}\nA:", "max_tokens": 16, "temperature": 0}, timeout=600).json()
    rtext = raw["choices"][0]["text"].strip()
    print(f"n={n:5d} words ({ptoks} chat tokens) needle={needle} | chat: {'OK  ' if needle in ctext else 'MISS'} {ctext[:40]!r} | raw: {'OK  ' if needle in rtext else 'MISS'} {rtext[:40]!r}", flush=True)
