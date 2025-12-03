import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("txt_file", help="Path to a .txt file containing one (possibly multi-line) prompt.")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--out", default="", help="Optional path to save the output text.")
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

    # read the whole file as a single string (keep line breaks)
    with open(args.txt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    if not prompt:
        print("[Error] Empty prompt file.")
        return

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    print("=== Prompt ===")
    print(prompt)
    print("\n=== Response ===")
    print(gen_text)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(gen_text)
        print(f"[Info] Saved output to {args.out}")

if __name__ == "__main__":
    main()


# python llama32_1b_ins_task_bench.py my_prompt.txt --out gpu_answer.txt


# python llama32_1b_ins_task_bench.py my_prompt.txt --max-new-tokens 256 --temperature 0.2 --out gpu_answer.txt


# python llama32_1b_ins_task_bench.py my_prompt.txt --dtype bf16


# python llama32_1b_ins_task_bench.py my_prompt.txt --model-id meta-llama/Llama-3.2-3B-Instruct