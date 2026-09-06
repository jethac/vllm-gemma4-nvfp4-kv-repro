#!/bin/bash
# serve_gemma_nvfp4.sh MODEL_DIR [extra vllm args...]  -- Gemma-4 with NVFP4 KV cache on sm120/sm121 (FlashInfer FA2 VO-split path)
# Needs a checkpoint with a calibrated 4-bit KV scheme (see calibrate.py); public NVFP4 Gemma-4 checkpoints carry no KV scales.
MODEL=$1; shift
exec vllm serve "$MODEL" --served-model-name local-llm --port 8001 --kv-cache-dtype nvfp4 \
  --gpu-memory-utilization ${GPU_UTIL:-0.85} --max-num-seqs 2 --max-model-len ${MAX_LEN:-32768} \
  --enable-prefix-caching --language-model-only "$@"
