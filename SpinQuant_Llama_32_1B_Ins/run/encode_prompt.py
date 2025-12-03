import sys
from transformers import AutoTokenizer

def main():
    if len(sys.argv) != 3:
        print("Usage: python encode_prompt.py <input_text_file> <output_id_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # 2. Read input text
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print("==== Input Prompt ====") 
    print(text)

    # 3. Tokenize into IDs
    token_ids = tokenizer.encode(text, add_special_tokens=True)

    # 4. Save IDs to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, token_ids)))

    print(f"✅ Encoded {len(token_ids)} tokens from '{input_file}' and saved to '{output_file}'")

if __name__ == "__main__":
    main()


# python encode_prompt.py my_prompt.txt my_prompt_token_idx.txt



