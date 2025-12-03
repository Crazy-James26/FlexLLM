#include "llama.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// silent logger that matches llama.h signature
static void llama_silent_log(enum ggml_log_level level, const char *text, void *user_data) {
    (void) level;
    (void) text;
    (void) user_data;
    // do nothing
}

// read whole file into string
std::string read_file(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return {};
    return std::string((std::istreambuf_iterator<char>(ifs)),
                       std::istreambuf_iterator<char>());
}

// write string to file
bool write_file(const std::string &path, const std::string &content) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;
    ofs << content;
    return true;
}

// message struct
struct Message {
    std::string role;
    std::string content;
};

// Llama 3.x style chat template
std::string apply_chat_template(const std::vector<Message> &messages,
                                bool add_generation_prompt = true) {
    const std::string bos = "<|begin_of_text|>";
    const std::string start_hdr = "<|start_header_id|>";
    const std::string end_hdr = "<|end_header_id|>";
    const std::string eot = "<|eot_id|>";

    std::string out;
    out += bos;

    for (const auto &m : messages) {
        out += start_hdr;
        out += m.role;
        out += end_hdr;
        out += "\n";
        out += m.content;
        out += eot;
    }

    if (add_generation_prompt) {
        out += start_hdr;
        out += "assistant";
        out += end_hdr;
        out += "\n";
    }

    return out;
}

// text -> token ids
std::vector<llama_token> encode_text(const llama_vocab *vocab,
                                     const std::string &text,
                                     bool add_special = false,
                                     bool parse_special = true) {
    std::vector<llama_token> tokens(1024);

    int32_t n = llama_tokenize(
        vocab,
        text.c_str(),
        (int32_t)text.size(),
        tokens.data(),
        (int32_t)tokens.size(),
        add_special,
        parse_special
    );

    if (n < 0) {
        return {};
    }

    tokens.resize(n);
    return tokens;
}

// token ids -> full text (concat per token, for debugging)
std::string decode_tokens_concat(const llama_vocab *vocab,
                                 const std::vector<llama_token> &tokens,
                                 bool remove_special = false,
                                 bool parse_special = false) {
    std::string result;

    for (auto tok : tokens) {
        char buf[256];
        int32_t written = llama_detokenize(
            vocab,
            &tok,
            /*n_tokens=*/1,
            buf,
            /*text_len_max=*/(int32_t)sizeof(buf),
            remove_special,
            parse_special
        );
        if (written < 0) {
            continue;
        }
        result.append(buf, buf + written);
    }

    return result;
}

// int main(int argc, char **argv) {
//     // install silent logger
//     llama_log_set(llama_silent_log, nullptr);

//     // defaults
//     const char *MODEL_PATH = "llama-3.2-1b-f16.gguf";
//     std::string in_path = "my_prompt.txt";
//     std::string out_path = "my_answer.txt";

//     // usage: ./chat_template_test [model.gguf] [input.txt] [output.txt]
//     if (argc >= 2) MODEL_PATH = argv[1];
//     if (argc >= 3) in_path = argv[2];
//     if (argc >= 4) out_path = argv[3];

//     // 1) read user prompt
//     std::string user_prompt = read_file(in_path);
//     if (user_prompt.empty()) {
//         std::cerr << "Failed to read prompt file: " << in_path << "\n";
//         return 1;
//     }

//     std::cout << "User prompt (" << in_path << "):\n";
//     std::cout << "----------------------------------------\n";
//     std::cout << user_prompt << "\n";
//     std::cout << "----------------------------------------\n\n";

//     // 2) load model vocab-only
//     llama_model_params mparams = llama_model_default_params();
//     mparams.vocab_only = true;
//     llama_model *model = llama_model_load_from_file(MODEL_PATH, mparams);
//     if (!model) {
//         std::cerr << "Failed to load model: " << MODEL_PATH << "\n";
//         return 1;
//     }

//     const llama_vocab *vocab = llama_model_get_vocab(model);
//     if (!vocab) {
//         std::cerr << "Failed to get vocab\n";
//         llama_model_free(model);
//         return 1;
//     }

//     // 3) build messages (system + user), like Python
//     std::vector<Message> messages = {
//         {"system", "You are a helpful assistant."},
//         {"user", user_prompt}
//     };

//     // 4) apply chat template
//     std::string chat_text = apply_chat_template(messages, /*add_generation_prompt=*/true);

//     std::cout << "Chat text (templated):\n";
//     std::cout << "----------------------------------------\n";
//     std::cout << chat_text << "\n";
//     std::cout << "----------------------------------------\n\n";

//     // 5) encode the templated text (debug / verify tokenizer)
//     auto ids = encode_text(vocab, chat_text,
//                            /*add_special=*/false,   // we already added <|begin_of_text|>
//                            /*parse_special=*/true); // we used special token strings
//     if (ids.empty()) {
//         std::cerr << "Tokenization failed\n";
//         llama_model_free(model);
//         return 1;
//     }

//     std::cout << "Encoded token IDs (" << ids.size() << "):\n";
//     for (auto id : ids) std::cout << id << " ";
//     std::cout << "\n\n";

//     // 6) (optional) decode back to see what tokenizer sees
//     std::string decoded = decode_tokens_concat(
//         vocab,
//         ids,
//         /*remove_special=*/false,
//         /*parse_special=*/false
//     );

//     std::cout << "Decoded (concat per token):\n";
//     std::cout << "----------------------------------------\n";
//     std::cout << decoded << "\n";
//     std::cout << "----------------------------------------\n";

//     // 7) write templated chat text to file (this mimics HF apply_chat_template(..., tokenize=False))
//     if (!write_file(out_path, chat_text)) {
//         std::cerr << "Failed to write to: " << out_path << "\n";
//         llama_model_free(model);
//         return 1;
//     }

//     std::cout << "Templated chat written to: " << out_path << "\n";

//     llama_model_free(model);
//     return 0;
// }
