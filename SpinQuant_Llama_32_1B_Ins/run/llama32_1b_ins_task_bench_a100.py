import argparse
import time
import threading
import subprocess
import json
from typing import Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# try NVIDIA NVML
try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False


def read_system_user(path: str) -> tuple[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if "#" not in raw:
        return "You are a helpful assistant.", raw
    sys_part, user_part = raw.split("#", 1)
    return sys_part.strip(), user_part.strip()


class PowerSampler:
    """Try NVIDIA (pynvml) first, then fall back to AMD rocm-smi."""
    def __init__(self, interval: float = 0.2, device_index: int = 0):
        self.interval = interval
        self.device_index = device_index
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[float] = []
        self._mode = None  # "nvidia" or "amd"

    def start(self):
        # try nvidia
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                self._mode = "nvidia"
            except Exception:
                self._mode = None
        # try amd
        if self._mode is None:
            try:
                out = subprocess.check_output(
                    ["rocm-smi", "--showpower", "--json"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                # if this works, we are on amd
                _ = json.loads(out)
                self._mode = "amd"
            except Exception:
                self._mode = None

        if self._mode is None:
            return

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            if self._mode == "nvidia":
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                    self.samples.append(power_mw / 1000.0)
                except Exception:
                    pass
            elif self._mode == "amd":
                try:
                    out = subprocess.check_output(
                        ["rocm-smi", "--showpower", "--json"],
                        stderr=subprocess.DEVNULL,
                        text=True,
                    )
                    data = json.loads(out)
                    key = f"card{self.device_index}"
                    if key in data:
                        # key name can vary a bit; adjust if needed
                        # common key:
                        pw = data[key].get("Average Graphics Package Power (W)")
                        if pw is None:
                            # try another common key
                            pw = data[key].get("Power (W)")
                        if pw is not None:
                            if isinstance(pw, str):
                                pw = pw.replace("W", "").strip()
                            self.samples.append(float(pw))
                except Exception:
                    pass
            time.sleep(self.interval)

    def stop(self):
        if self._mode == "nvidia":
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        elif self._mode == "amd":
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
        else:
            pass

    def avg_power(self) -> Optional[float]:
        if not self.samples:
            return None
        return sum(self.samples) / len(self.samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("txt_file", help="Path to a .txt file containing system#user.")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="we will force 1024 anyway")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--out", default="", help="Optional path to save the output text.")
    parser.add_argument("--power-gpu", type=int, default=0, help="GPU index for power sampling (NVIDIA or AMD)")
    args = parser.parse_args()

    # choose dtype
    if args.dtype == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print(f"[Info] Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )
    print("[Info] Model loaded. Enter 'q' to quit. Press Enter or any char to run.")
    force_tokens = 1024  # your fixed benchmark length

    while True:
        cmd = input(">>> ").strip()
        if cmd.lower() == "q":
            print("[Info] Quit.")
            break

        # read prompt file each time
        system_text, user_text = read_system_user(args.txt_file)
        if not user_text:
            print("[Error] User part is empty.")
            continue

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        # sync before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        sampler = PowerSampler(device_index=args.power_gpu)
        sampler.start()

        start_time = time.time()
        # IMPORTANT: force to generate exactly 1024 tokens:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=force_tokens,
                min_new_tokens=force_tokens,  # try to force the length
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                # do NOT pass eos_token_id here so it doesn't stop early
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        sampler.stop()

        # output ids
        output_ids = output.sequences
        # generated part
        gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        total_gen_tokens = gen_ids.shape[0]

        # decode full
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # find first eos in generated part (to estimate "real output" time)
        eos_id = tokenizer.eos_token_id
        real_tokens = total_gen_tokens
        for i, tok in enumerate(gen_ids.tolist()):
            if tok == eos_id:
                real_tokens = i + 1  # include eos
                break

        total_time = end_time - start_time
        total_throughput = total_gen_tokens / total_time if total_time > 0 else 0.0

        # estimate real output time proportionally
        real_time = total_time * (real_tokens / total_gen_tokens) if total_gen_tokens > 0 else 0.0
        real_throughput = real_tokens / real_time if real_time > 0 else 0.0

        avg_power = sampler.avg_power()

        print("\n=== System ===")
        print(system_text)
        print("\n=== User ===")
        print(user_text)
        print("\n=== Response ===")
        print(gen_text)

        print("\n=== Stats ===")
        print(f"Total generation time (1024): {total_time:.3f} s")
        print(f"Total generated tokens: {total_gen_tokens}")
        print(f"Total throughput: {total_throughput:.2f} tokens/s")
        print(f"Real output tokens (up to first EOS): {real_tokens}")
        print(f"Estimated real output time: {real_time:.3f} s")
        print(f"Estimated real output throughput: {real_throughput:.2f} tokens/s")
        if avg_power is not None:
            print(f"Average GPU power: {avg_power:.1f} W")
        else:
            print("Average GPU power: N/A")

        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(gen_text)
            print(f"[Info] Saved output to {args.out}")

        print("\n--- Ready for next request (q to quit) ---")


if __name__ == "__main__":
    main()




# python llama32_1b_ins_task_bench_a100.py my_prompt.txt --out gpu_answer.txt


# python llama32_1b_ins_task_bench_a100.py my_prompt.txt --max-new-tokens 256 --temperature 0.2 --out gpu_answer.txt


# python llama32_1b_ins_task_bench_a100.py my_prompt.txt --dtype bf16


# python llama32_1b_ins_task_bench_a100.py my_prompt.txt --model-id meta-llama/Llama-3.2-3B-Instruct