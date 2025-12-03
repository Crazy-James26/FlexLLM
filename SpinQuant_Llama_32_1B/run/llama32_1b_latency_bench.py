#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure TTFT/TTFB and generation throughput for Llama 3.2 1B,
and (optionally) sample GPU power during warmup/prefill/generate phases.

Example:
  python llama32_1b_latency_bench.py \
      --model meta-llama/Llama-3.2-1B \
      --input-tokens 1024 --output-tokens 1024 \
      --dtype bf16 --attn sdpa --power-sample-ms 100
"""

import argparse
import time
import math
import threading
import statistics
import shutil
import subprocess
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


# -----------------------------
# Power monitor
# -----------------------------

class PowerMonitor:
    """
    Sample GPU power draw (Watts) in a background thread.

    Backend:
      - NVML (preferred, via pynvml)
      - fallback: call `nvidia-smi` periodically

    Phases:
      Use set_phase("warmup"/"prefill"/"generate"/"idle") to tag samples.
    """

    def __init__(self, cuda_device_index: int, interval_s: float = 0.1, enable: bool = True):
        self.enable = enable and torch.cuda.is_available()
        self.idx = int(cuda_device_index)
        self.interval_s = float(interval_s)
        self.samples = []  # list of tuples (t, phase, power_w)
        self._phase = "idle"
        self._stop = threading.Event()
        self._thread = None
        self._backend = None
        self._nvml_dev = None

        if not self.enable:
            return

        # Try NVML
        try:
            import pynvml as _nvml
            _nvml.nvmlInit()
            self._nvml = _nvml
            # Map CUDA device -> NVML index (assumes 1-1 mapping; works on typical setups)
            self._nvml_dev = _nvml.nvmlDeviceGetHandleByIndex(self.idx)
            # Will raise if inaccessible
            _ = _nvml.nvmlDeviceGetPowerUsage(self._nvml_dev)
            self._backend = "nvml"
        except Exception:
            # Fallback: nvidia-smi polling
            if shutil.which("nvidia-smi") is not None:
                self._backend = "smi"
            else:
                self.enable = False  # no backend available

    def set_phase(self, name: str):
        self._phase = name

    def _poll_power_nvml(self):
        try:
            mw = self._nvml.nvmlDeviceGetPowerUsage(self._nvml_dev)  # milliwatts
            return mw / 1000.0
        except Exception:
            return None

    def _poll_power_smi(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", str(self.idx)],
                text=True
            ).strip()
            return float(out.splitlines()[0])
        except Exception:
            return None

    def _run(self):
        poll = self._poll_power_nvml if self._backend == "nvml" else self._poll_power_smi
        while not self._stop.is_set():
            pw = poll()
            now = time.perf_counter()
            if pw is not None:
                self.samples.append((now, self._phase, float(pw)))
            # light sleep
            self._stop.wait(self.interval_s)

    def start(self):
        if not self.enable:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.enable:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summarize(self, phase: str, start_t: float, end_t: float):
        """Return (avg_W, peak_W, duration_s, est_energy_J) for samples in [start_t, end_t] with tag==phase."""
        if not self.enable:
            return None
        pts = [(t, p) for (t, ph, p) in self.samples if ph == phase and start_t <= t <= end_t]
        duration = max(0.0, end_t - start_t)
        if not pts:
            return (float("nan"), float("nan"), duration, float("nan"))
        powers = [p for (_, p) in pts]
        avg_w = statistics.fmean(powers)
        peak_w = max(powers)
        # Energy ≈ average power * duration
        energy_j = avg_w * duration
        return (avg_w, peak_w, duration, energy_j)


# -----------------------------
# Helpers for text-gen timing
# -----------------------------

@dataclass
class FirstTokenTimer(StoppingCriteria):
    """Record wall-clock time when first new token beyond the prompt appears."""
    prompt_len: int
    start_time: float
    first_token_s: float | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        cur_len = input_ids.shape[-1]
        if self.first_token_s is None and cur_len > self.prompt_len:
            self.first_token_s = time.perf_counter() - self.start_time
        return False


def make_prompt_of_length(tokenizer, target_len: int) -> torch.LongTensor:
    seed_sentence = "This is a latency measurement prompt for Llama models. "
    text = seed_sentence
    while True:
        toks = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        if toks.numel() >= target_len:
            return toks[:target_len]
        text += seed_sentence


def humanize_rate(tokens: int, seconds: float) -> str:
    if seconds <= 0:
        return "n/a"
    return f"{tokens/seconds:.2f} tok/s"


def pick_attn_impl(requested: str) -> str | None:
    requested = (requested or "auto").lower()
    if requested in ("fa2", "flash", "flash2", "flash_attention_2"):
        return "flash_attention_2"
    if requested in ("sdpa", "scaled_dot_product", "torch"):
        return "sdpa"
    if requested in ("eager", "py", "vanilla"):
        return "eager"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', 'cpu', or 'cuda:0' etc.")
    parser.add_argument("--attn", type=str, default="auto", help="'auto'|'fa2'|'sdpa'|'eager'")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (default).")
    parser.add_argument("--temperature", type=float, default=0.0, help="<=0 for greedy")
    parser.add_argument("--no-warmup", action="store_true", help="Disable warmup generate()")
    parser.add_argument("--no-prefill-diagnostic", action="store_true", help="Skip explicit prefill timing")
    parser.add_argument("--no-power", action="store_true", help="Disable power monitoring")
    parser.add_argument("--power-sample-ms", type=int, default=100, help="Power sample interval in ms")
    args = parser.parse_args()

    # DType mapping
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Device
    if args.device == "auto":
        device_map = "auto"
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_map = None
        device_str = args.device

    print(f"== Model: {args.model}")
    print(f"== Requested dtype: {args.dtype}  (torch dtype: {dtype})")
    print(f"== Device: {device_str}")
    if torch.cuda.is_available():
        print(f"== CUDA device name: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Model
    model_kwargs = dict(torch_dtype=dtype)
    attn_impl = pick_attn_impl(args.attn)
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    print(f"== Attention implementation: {attn_impl}")

    print("== Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map=device_map, **model_kwargs)
    except Exception as e:
        if attn_impl == "flash_attention_2":
            print(f"!! flash_attention_2 unavailable ({e.__class__.__name__}: {e}). Falling back to 'sdpa'.")
            model_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(args.model, device_map=device_map, **model_kwargs)
        else:
            raise
    if device_map is None:
        model = model.to(device_str)
    model.eval()

    # Build prompt
    prompt_ids = make_prompt_of_length(tokenizer, args.input_tokens).unsqueeze(0)
    attention_mask = torch.ones_like(prompt_ids)

    # Put inputs on same device as embeddings
    try:
        embed_dev = model.get_input_embeddings().weight.device
    except Exception:
        embed_dev = next(p for p in model.parameters() if p.device.type != 'meta').device
    prompt_ids = prompt_ids.to(embed_dev)
    attention_mask = attention_mask.to(embed_dev)
    print(f"== Sanity: inputs on {prompt_ids.device}, embed on {embed_dev}")

    input_len = prompt_ids.shape[-1]
    assert input_len == args.input_tokens, f"Prompt is {input_len} tokens, expected {args.input_tokens}"

    # Determine CUDA device index for power monitor
    monitor = None
    if not args.no_power and torch.cuda.is_available() and embed_dev.type == "cuda":
        gpu_index = embed_dev.index if embed_dev.index is not None else torch.cuda.current_device()
        monitor = PowerMonitor(cuda_device_index=gpu_index, interval_s=args.power_sample_ms / 1000.0, enable=True)
        monitor.start()
    else:
        monitor = PowerMonitor(0, enable=False)

    # Warmup (optional)
    if not args.no_warmup:
        print("== Warmup...")
        if monitor.enable:
            monitor.set_phase("warmup")
        with torch.no_grad():
            _ = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        warmup_end = time.perf_counter()
    else:
        warmup_end = time.perf_counter()

    # Prefill diagnostic (optional)
    prefill_s = float("nan")
    prefill_t0 = prefill_t1 = None
    if not args.no_prefill_diagnostic:
        print("== Prefill diagnostic (forward over prompt only)...")
        if monitor.enable:
            monitor.set_phase("prefill")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=prompt_ids, attention_mask=attention_mask, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_t1 = time.perf_counter()
        prefill_s = prefill_t1 - prefill_t0
        print(f"Prefill time for {input_len} tokens: {prefill_s:.4f} s")

    # Actual measurement (generate)
    print("== Measuring...")
    if monitor.enable:
        monitor.set_phase("generate")
    stopping = FirstTokenTimer(prompt_len=input_len, start_time=0.0)
    gen_kwargs = dict(max_new_tokens=args.output_tokens, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    if args.greedy or args.temperature <= 0:
        gen_kwargs.update(dict(do_sample=False))
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=args.temperature))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    stopping.start_time = t0
    with torch.no_grad():
        out = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            output_scores=False,
            stopping_criteria=StoppingCriteriaList([stopping]),
            **gen_kwargs,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_s = t1 - t0
    ttfb_s = stopping.first_token_s if stopping.first_token_s is not None else float("nan")

    # Stop power monitor
    if monitor.enable:
        monitor.set_phase("idle")
        monitor.stop()

    # Tokens and rates
    out_len = out.sequences.shape[-1]
    new_tokens = out_len - input_len
    decoded_prefix = tokenizer.decode(out.sequences[0, input_len:input_len+64], skip_special_tokens=True)
    overall_rate = humanize_rate(new_tokens, total_s)
    if math.isfinite(ttfb_s) and total_s > ttfb_s and new_tokens > 1:
        steady_rate = humanize_rate(new_tokens - 1, total_s - ttfb_s)
    else:
        steady_rate = "n/a"

    # Print results
    print("\n===== Results =====")
    print(f"Prompt tokens        : {input_len}")
    print(f"Generated tokens     : {new_tokens} (requested {args.output_tokens})")
    if math.isfinite(prefill_s):
        print(f"Prefill (diagnostic) : {prefill_s:.4f} s")
    print(f"Time to first token  : {ttfb_s:.4f} s  (measured from generate() start; includes prefill)")
    print(f"Total generation time: {total_s:.4f} s")
    print(f"Overall throughput   : {overall_rate}")
    print(f"Steady-state throughput (excl. TTFB): {steady_rate}")
    print("\nOutput preview:")
    print(decoded_prefix)

    # Power summaries
    if monitor.enable:
        print("\n===== Power (GPU {}) =====".format(monitor.idx))
        now = time.perf_counter()
        # Warmup summary
        if not args.no_warmup:
            # Approximate warmup start as (warmup_end - ~0.2s) if we didn't record start explicitly
            warmup_avg = monitor.summarize("warmup", start_t=warmup_end-10.0, end_t=warmup_end)
            if warmup_avg:
                avg, peak, dur, energy = warmup_avg
                print(f"Warmup:   avg {avg:.1f} W | peak {peak:.1f} W | dur ~{dur:.2f} s | energy ~{energy:.1f} J")
        # Prefill summary
        if not args.no_prefill_diagnostic and prefill_t0 is not None and prefill_t1 is not None:
            prefill_avg = monitor.summarize("prefill", start_t=prefill_t0, end_t=prefill_t1)
            if prefill_avg:
                avg, peak, dur, energy = prefill_avg
                print(f"Prefill:  avg {avg:.1f} W | peak {peak:.1f} W | dur {dur:.2f} s | energy ~{energy:.1f} J")
        # Generate summary
        gen_avg = monitor.summarize("generate", start_t=t0, end_t=t1)
        if gen_avg:
            avg, peak, dur, energy = gen_avg
            print(f"Generate: avg {avg:.1f} W | peak {peak:.1f} W | dur {dur:.2f} s | energy ~{energy:.1f} J")
    else:
        print("\n(POWER) Skipped: NVML and `nvidia-smi` unavailable or --no-power specified.")

    print("\nDone.")


if __name__ == "__main__":
    main()


# python llama32_1b_latency_bench.py --model meta-llama/Llama-3.2-1B --dtype bf16 --attn sdpa --power-sample-ms 100 --input-tokens 1024 --output-tokens 2048 

