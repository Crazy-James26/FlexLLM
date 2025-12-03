import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("txt_file", help="Path to a .txt file containing ONE (possibly multi-line) prompt.")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--out", default="", help="Optional path to save model output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # dtype
    if args.dtype == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[Info] Loading {args.model_id} ...")
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()

    # read the whole file as one prompt
    with open(args.txt_file, "r", encoding="utf-8") as f:
        prefix = f.read().rstrip()  # keep formatting, trim tail spaces
    if not prefix:
        print("[Error] Empty prompt file."); return

    inputs = tok(prefix, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max-new-tokens if hasattr(args, "max-new-tokens") else args.max_new_tokens,  # guard
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )

    gen = tok.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    print("=== Completion ===")
    print(gen)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(gen)
        print(f"[Info] Saved to {args.out}")

if __name__ == "__main__":
    main()

# python llama32_1b_task_bench.py my_prompt.txt --out gpu_answer.txt --dtype bf16
