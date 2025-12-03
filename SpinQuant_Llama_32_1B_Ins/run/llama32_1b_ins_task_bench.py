import argparse
import time
import threading
import subprocess
import json
from typing import Optional, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# try NVIDIA NVML
try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False


def read_system_user(path: str) -> Tuple[str, str]:
    """
    Read whole file and split into system/user by the first '#'.
    If no '#', system = default, user = whole file.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if "#" not in raw:
        return "You are a helpful assistant.", raw
    sys_part, user_part = raw.split("#", 1)
    return sys_part.strip(), user_part.strip()


class PowerSampler:
    """
    Sample GPU power in background.
    1) try NVIDIA (pynvml)
    2) else try AMD (rocm-smi --showpower --json)
    """
    def __init__(self, interval: float = 0.2, device_index: int = 0):
        self.interval = interval
        self.device_index = device_index
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[float] = []
        self._mode = None  # "nvidia" or "amd"

    def start(self):
        # try NVIDIA first
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                self._mode = "nvidia"
            except Exception:
                self._mode = None

        # fallback to AMD rocm-smi
        if self._mode is None:
            try:
                out = subprocess.check_output(
                    ["rocm-smi", "--showpower", "--json"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
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
                        pw = (
                            data[key].get("Average Graphics Package Power (W)")
                            or data[key].get("Power (W)")
                        )
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

    def avg_power(self) -> Optional[float]:
        if not self.samples:
            return None
        return sum(self.samples) / len(self.samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("txt_file", help="Path to a .txt file containing system#user.")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Target tokens (we force 1024).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--out", default="", help="Optional path to save the output text.")
    parser.add_argument("--power-gpu", type=int, default=0, help="GPU index for power sampling")
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
        device_map="auto" if torch.cuda.is_available() else None,
    )
    print("[Info] Model loaded. Enter 'q' to quit. Press Enter / any key to run.")

    force_tokens = args.max_new_tokens if args.max_new_tokens > 0 else 1024

    while True:
        cmd = input(">>> ").strip()
        if cmd.lower() == "q":
            print("[Info] Quit.")
            break

        # 1) read system/user each iteration
        try:
            system_text, user_text = read_system_user(args.txt_file)
        except FileNotFoundError:
            print(f"[Error] Cannot read {args.txt_file}")
            continue

        if not user_text:
            print("[Error] User part is empty.")
            continue

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        # 2) prefill timing (prompt-only forward)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_prefill0 = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_prefill1 = time.time()
        prefill_time = t_prefill1 - t_prefill0

        # 3) start timing + power for generation
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        sampler = PowerSampler(device_index=args.power_gpu)
        sampler.start()
        start_time = time.time()

        # -------- PHASE 1: normal generation, allow EOS --------
        with torch.no_grad():
            out1 = model.generate(
                **inputs,
                max_new_tokens=force_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mid_time = time.time()

        # extract phase-1 generated ids
        gen_ids_1 = out1.sequences[0][inputs["input_ids"].shape[-1]:]
        len_1 = gen_ids_1.shape[0]

        # -------- PHASE 2: if < force_tokens, generate the rest --------
        gen_ids_2 = None
        len_2 = 0
        if len_1 < force_tokens:
            need = force_tokens - len_1
            more_inputs = {"input_ids": out1.sequences.to(model.device)}
            with torch.no_grad():
                out2 = model.generate(
                    **more_inputs,
                    min_new_tokens=need,
                    max_new_tokens=need,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            gen_ids_2 = out2.sequences[0][out1.sequences.shape[-1]:]
            len_2 = gen_ids_2.shape[0]
        else:
            end_time = mid_time

        sampler.stop()
        avg_power = sampler.avg_power()

        # 4) decode only phase-1 (the real answer)
        gen_text = tokenizer.decode(gen_ids_1, skip_special_tokens=True).strip()

        # 5) compute stats
        total_gen_tokens = len_1 + len_2  # should be force_tokens
        total_time = end_time - start_time
        total_throughput = total_gen_tokens / total_time if total_time > 0 else 0.0

        # find earliest "end" in phase 1 to get real length
        gen1_list = gen_ids_1.tolist()
        end_ids = set()
        if tokenizer.eos_token_id is not None:
            end_ids.add(tokenizer.eos_token_id)
        for tok_str in ("<|eot_id|>", "<|end_of_text|>"):
            try:
                tid = tokenizer.convert_tokens_to_ids(tok_str)
                if tid is not None:
                    end_ids.add(tid)
            except Exception:
                pass

        real_tokens = len_1
        for i, tok in enumerate(gen1_list):
            if tok in end_ids:
                real_tokens = i + 1
                break

        # estimate real-time proportionally
        if total_gen_tokens > 0:
            real_time = total_time * (real_tokens / total_gen_tokens)
        else:
            real_time = 0.0
        real_throughput = real_tokens / real_time if real_time > 0 else 0.0

        # 6) print
        print("\n=== System ===")
        print(system_text)
        print("\n=== User ===")
        print(user_text)
        print("\n=== Response ===")
        print(gen_text)

        print("\n=== Stats ===")
        print(f"MI210 GPU Prefill time: {prefill_time:.3f} s")
        print(f"MI210 GPU Prefill sequence length: {inputs['input_ids'].shape[-1]}")
        print(f"MI210 GPU Prefill throughput: {inputs['input_ids'].shape[-1]/prefill_time:.2f} tokens/s")
        print(f"MI210 GPU Decode time: {real_time:.3f} / {total_time:.3f} s")
        print(f"MI210 GPU Decode sequence length: {real_tokens} / {total_gen_tokens}")
        print(f"MI210 GPU Decode throughput: {real_throughput:.2f} tokens/s")
        if avg_power is not None:
            print(f"Average GPU power: {avg_power:.1f} W")
        else:
            print("Average GPU power: N/A")

        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(gen_text)
            # print(f"[Info] Saved output to {args.out}")

        print("\n--- Ready for next request (q to quit) ---")


if __name__ == "__main__":
    main()





# python llama32_1b_ins_task_bench.py my_prompt.txt --out gpu_answer.txt


# python llama32_1b_ins_task_bench.py my_prompt.txt --max-new-tokens 256 --temperature 0.2 --out gpu_answer.txt


# python llama32_1b_ins_task_bench.py my_prompt.txt --dtype bf16


# python llama32_1b_ins_task_bench.py my_prompt.txt --model-id meta-llama/Llama-3.2-3B-Instruct