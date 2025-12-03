#!/usr/bin/env python3
# vllm_bench.py — TTFT / throughput / energy on A100 with vLLM
import os
import time
import argparse
import threading
from dataclasses import dataclass
from typing import Optional, List, Iterable

# Optional torch (TF32 + GPU name)
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# -------------------------- CLI --------------------------
def parse_args():
    p = argparse.ArgumentParser("vLLM single-request benchmark (TTFT, tok/s, energy)")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    p.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"))
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--gpu-mem-util", type=float, default=0.95)
    p.add_argument("--prompt-tokens", type=int, default=1024)
    p.add_argument("--new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--nvml-interval", type=float, default=0.01)
    p.add_argument("--no-tf32", action="store_true")
    return p.parse_args()

# ---------------------- NVML power meter ----------------------
try:
    import pynvml as nvml
    NVML_OK = True
except Exception:
    try:
        import nvidia_ml_py3 as nvml  # alt package name
        NVML_OK = True
    except Exception:
        NVML_OK = False

@dataclass
class PowerSample:
    ts: float
    watts: float

class PowerMeter:
    def __init__(self, gpu_index: int = 0, interval_s: float = 0.01):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._samples: List[PowerSample] = []
        self._t: Optional[threading.Thread] = None
        self._handle = None
        self._inited = False

    def start(self):
        if not NVML_OK:
            return
        try:
            nvml.nvmlInit()
            self._handle = nvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._inited = True
        except Exception:
            self._inited = False
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            watts = 0.0
            if self._inited:
                try:
                    p_mw = nvml.nvmlDeviceGetPowerUsage(self._handle)
                    watts = p_mw / 1000.0
                except Exception:
                    watts = 0.0
            self._samples.append(PowerSample(time.perf_counter(), watts))
            time.sleep(self.interval_s)

    def stop(self):
        if not NVML_OK:
            return
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2)
        try:
            if self._inited:
                nvml.nvmlShutdown()
        except Exception:
            pass

    def energy_joules(self) -> float:
        s = self._samples
        if len(s) < 2:
            return 0.0
        e = 0.0
        for i in range(1, len(s)):
            dt = s[i].ts - s[i-1].ts
            p_avg = 0.5 * (s[i].watts + s[i-1].watts)
            e += p_avg * dt
        return e

    def avg_power(self) -> float:
        if not self._samples:
            return 0.0
        return sum(x.watts for x in self._samples) / len(self._samples)

def read_nvml_mem_gb(idx: int = 0) -> float:
    if not NVML_OK:
        return float("nan")
    try:
        nvml.nvmlInit()
        h = nvml.nvmlDeviceGetHandleByIndex(idx)
        info = nvml.nvmlDeviceGetMemoryInfo(h)
        used_gb = info.used / (1024**3)
        nvml.nvmlShutdown()
        return used_gb
    except Exception:
        return float("nan")

# ---------------------- Prompt builder ----------------------
def build_exact_prompt(tokenizer: AutoTokenizer, target_len: int) -> str:
    base = ("You are a helpful assistant. This paragraph is neutral filler used "
            "to reach a target token length for benchmarking purposes. ")
    text = base
    while True:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= target_len:
            ids = ids[:target_len]
            return tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        text += base * 4

# ---------------------- Streaming adapter ----------------------
def stream_outputs(llm: LLM, prompts: List[str], params: SamplingParams, use_tqdm: bool=False) -> Iterable:
    """
    Yields streaming RequestOutput chunks regardless of vLLM version.
    - If llm.generate_stream exists -> use it.
    - Else -> use llm.generate(..., stream=True).
    """
    if hasattr(llm, "generate_stream"):
        for chunk in llm.generate_stream(prompts, params):
            yield chunk
    else:
        # Older API: generate(..., stream=True)
        for chunk in llm.generate(prompts, params, use_tqdm=use_tqdm):
            yield chunk

# --------------------------- Main ---------------------------
def main():
    args = parse_args()

    if TORCH_OK and not args.no_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=args.hf_token)
    prompt = build_exact_prompt(tok, args.prompt_tokens)
    enc = tok(prompt, add_special_tokens=False, return_tensors="pt")
    assert enc["input_ids"].shape[1] == args.prompt_tokens, f"Prompt tokens={enc['input_ids'].shape[1]}"

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=False,
        hf_token=args.hf_token,
    )

    # Warmup (non-stream; older vLLM may not accept stream=False)
    warm_params = SamplingParams(
        max_tokens=max(8, min(args.new_tokens, 64)),
        temperature=args.temperature,
        top_p=args.top_p,
    )
    for _ in range(args.warmup_rounds):
        for _ in llm.generate([prompt], warm_params, use_tqdm=False):
            pass

    mem_before = read_nvml_mem_gb(0)

    params = SamplingParams(
        max_tokens=args.new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    pm = PowerMeter(gpu_index=0, interval_s=args.nvml_interval)
    if NVML_OK:
        pm.start()

    t0 = time.perf_counter()
    first_token_time = None
    total_generated_tok = 0

    # True streaming (API auto-detected)
    for chunk in stream_outputs(llm, [prompt], params, use_tqdm=False):
        out = chunk.outputs[0]
        # Prefer token_ids if present
        if hasattr(out, "token_ids") and out.token_ids is not None:
            cur_cnt = len(out.token_ids)
            if cur_cnt > total_generated_tok:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                total_generated_tok = cur_cnt
        else:
            # Fallback trigger on first non-empty text
            if out.text and first_token_time is None:
                first_token_time = time.perf_counter()

    t1 = time.perf_counter()
    if NVML_OK:
        pm.stop()

    mem_after = read_nvml_mem_gb(0)

    total_time = t1 - t0
    if first_token_time is None:
        ttft = float("nan")
        decode_time = float("nan")
    else:
        ttft = first_token_time - t0
        decode_time = max(1e-9, total_time - ttft)

    toks = total_generated_tok
    tokps_total = toks / total_time if total_time > 0 else float("nan")
    tokps_decode = toks / decode_time if decode_time > 0 else float("nan")

    energy_j = pm.energy_joules() if NVML_OK else float("nan")
    avg_w = pm.avg_power() if NVML_OK else float("nan")

    try:
        gpu_name = torch.cuda.get_device_name(0) if TORCH_OK else "CUDA device"
    except Exception:
        gpu_name = "CUDA device"

    print("\n===== System =====")
    print(f"GPU: {gpu_name}")
    print(f"vLLM engine dtype: {args.dtype}")
    print(f"TP size: {args.tp_size} | max_model_len: {args.max_model_len}")

    print("\n===== Workload =====")
    print(f"Prompt tokens: {args.prompt_tokens}")
    print(f"Requested new tokens: {args.new_tokens}")
    print(f"Generated tokens (counted): {toks}")

    print("\n===== Metrics (single sequence) =====")
    print(f"TTFT (s):                 {ttft:.4f}")
    print(f"Total time (s):           {total_time:.4f}")
    print(f"Decode-only time (s):     {decode_time:.4f}")
    print(f"Throughput total (tok/s): {tokps_total:.2f}")
    print(f"Throughput decode (tok/s):{tokps_decode:.2f}")

    print(f"GPU mem used before (GB): {mem_before:.2f}")
    print(f"GPU mem used after  (GB): {mem_after:.2f}")

    if NVML_OK:
        print("\n===== Energy =====")
        print(f"Energy (J):               {energy_j:.1f}")
        print(f"Average power (W):        {avg_w:.1f}")
        print(f"Energy per token (J/tok): {energy_j / max(toks,1):.4f}")

if __name__ == "__main__":
    main()



# python llama32_1b_vllm_latency_bench.py --model meta-llama/Llama-3.2-1B --prompt-tokens 1024 --new-tokens 2048