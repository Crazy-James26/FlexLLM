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

// text -> token ids
std::vector<llama_token> encode_text(const llama_vocab *vocab,
                                     const std::string &text,
                                     bool add_special = true,
                                     bool parse_special = false) {
    std::vector<llama_token> tokens(512);

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

// token ids -> full text (concat per token, safest)
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
            continue;  // skip failed
        }
        result.append(buf, buf + written);
    }

    return result;
}