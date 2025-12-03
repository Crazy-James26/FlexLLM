import sys
from transformers import AutoTokenizer

def main():
    # 1. Check arguments
    if len(sys.argv) != 3:
        print("Usage: python decode_answer.py <output_id_file.txt> <output_text.txt>")
        sys.exit(1)

    token_file = sys.argv[1]
    output_file = sys.argv[2]

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # 3. Get stop token ID (you can customize this)
    stop_id = tokenizer.eos_token_id  # usually 128009 for Llama 3.2 models

    # 4. Read token IDs from file
    with open(token_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 5. Convert to list of integers
    token_ids = [int(x) for x in content.strip().split() if x.isdigit()]

    # 6. Remove padding zeros
    token_ids = [t for t in token_ids if t != 0]

    # 7. Stop decoding if stop_id is found
    if stop_id in token_ids:
        stop_index = token_ids.index(stop_id)
        token_ids = token_ids[:stop_index]

    # 8. Decode into text
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)

    # 9. Print result and Save decoded text
    print("==== Decoded Output ====")
    print(decoded_text)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(decoded_text)

    print(f"✅ Decoded text saved to '{output_file}'")

if __name__ == "__main__":
    main()

# python decode_answer.py my_sampled_token_idx_0.txt my_answer.txt