#include <iostream>
#include <vector>
#include <random>
#include <gflags/gflags.h>

#include <fstream>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <iomanip>

DEFINE_string(bitstream_pref, "", ""/*path to bitstream file, run csim if empty*/);
DEFINE_string(bitstream_dec, "", ""/*path to bitstream file, run csim if empty*/);
DEFINE_string(model_gguf_path, "llama-3.2-1b-f16.gguf", ""/*path to bitstream file, run csim if empty*/);

#include "SpinQuant_Prefilling_mem_opt.h"
#include "SpinQuant_Decoding_mem_opt.h"
#include "llama_tokenizer.h"

template <int read_parallel, int weight_parallel, int input_dim, int output_dim>
void prefilling_read_int4_bin_as_int8_weight_mmap(
    const std::string& layer_name,
    int layer,
    std::vector<hls::vector<ap_int<8>, read_parallel/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, read_parallel/2>>>& weight_mmap,
    int mmap_offset
) {
    static_assert(read_parallel % 2 == 0, "read_parallel must be even");
    static_assert(weight_parallel % 2 == 0, "weight_parallel must be even");

    char name[256];
    std::snprintf(name, sizeof(name), "parameters/%s_L%02d.bin", layer_name.c_str(), layer);
    std::ifstream fin(name, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open file: " << name << "\n";
        return;
    }

    // Each iteration packs two output channels: (2*n, 2*n+1)
    for (int n = 0; n < output_dim / 2; ++n) {
        // read one whole row (input_dim bytes) for each of the two output channels
        std::vector<int8_t> data_0(input_dim);
        std::vector<int8_t> data_1(input_dim);

        const std::streamoff off0 = static_cast<std::streamoff>( (2*n)     ) * input_dim * sizeof(int8_t);
        const std::streamoff off1 = static_cast<std::streamoff>( (2*n + 1) ) * input_dim * sizeof(int8_t);

        fin.seekg(off0, std::ios::beg);
        fin.read(reinterpret_cast<char*>(data_0.data()), data_0.size() * sizeof(int8_t));
        if (!fin) { std::cerr << "Read error @ row " << (2*n) << " in " << name << "\n"; return; }

        fin.seekg(off1, std::ios::beg);
        fin.read(reinterpret_cast<char*>(data_1.data()), data_1.size() * sizeof(int8_t));
        if (!fin) { std::cerr << "Read error @ row " << (2*n+1) << " in " << name << "\n"; return; }

        
        // Pack and write to mmap
        // Tile index = n / (weight_parallel/2), lane = n % (weight_parallel/2)
        const int tile  = n / (weight_parallel / 2);
        const int lane  = n % (weight_parallel / 2);
        for (int m = 0; m < input_dim; ++m) {
            // Values were exported in [-8..7]. Truncate to 4-bit two's complement.
            ap_int<4> val_0 = ap_int<4>(data_0[m]);  // low nibble (out = 2*n)
            ap_int<4> val_1 = ap_int<4>(data_1[m]);  // high nibble (out = 2*n+1)

            ap_int<8> pack_val = 0;
            pack_val.range(3, 0) = val_0;  // low  nibble
            pack_val.range(7, 4) = val_1;  // high nibble

            // Flat mmap index: [mmap_offset + tile][m]
            weight_mmap[mmap_offset + tile * input_dim + m][lane] = pack_val;
        }
    }
}


template <int read_parallel, int weight_parallel, int input_dim, int output_dim, int block_num>
void prefilling_read_int4_bin_as_blocked_int8_weight_mmaps(
    const std::string& layer_name,
    int layer,
    std::vector<hls::vector<ap_int<8>, read_parallel/block_num/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, read_parallel/block_num/2>>> weight_mmaps[block_num],
    int mmap_offset
) {
    static_assert(read_parallel % 2 == 0, "read_parallel must be even");
    static_assert(weight_parallel % 2 == 0, "weight_parallel must be even");

    char name[256];
    std::snprintf(name, sizeof(name), "parameters/%s_L%02d.bin", layer_name.c_str(), layer);
    std::ifstream fin(name, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open file: " << name << "\n";
        return;
    }

    // Each iteration packs two output channels: (2*n, 2*n+1)
    for (int n = 0; n < output_dim / 2; ++n) {
        // read one whole row (input_dim bytes) for each of the two output channels
        std::vector<int8_t> data_0(input_dim);
        std::vector<int8_t> data_1(input_dim);

        const std::streamoff off0 = static_cast<std::streamoff>( (2*n)     ) * input_dim * sizeof(int8_t);
        const std::streamoff off1 = static_cast<std::streamoff>( (2*n + 1) ) * input_dim * sizeof(int8_t);

        fin.seekg(off0, std::ios::beg);
        fin.read(reinterpret_cast<char*>(data_0.data()), data_0.size() * sizeof(int8_t));
        if (!fin) { std::cerr << "Read error @ row " << (2*n) << " in " << name << "\n"; return; }

        fin.seekg(off1, std::ios::beg);
        fin.read(reinterpret_cast<char*>(data_1.data()), data_1.size() * sizeof(int8_t));
        if (!fin) { std::cerr << "Read error @ row " << (2*n+1) << " in " << name << "\n"; return; }
        
        // Pack and write to mmap
        const int block_id = n % block_num;
        const int tile  = (n/block_num) / (weight_parallel/block_num / 2);
        const int lane  = (n/block_num) % (weight_parallel/block_num / 2);
        for (int m = 0; m < input_dim; ++m) {
            // Values were exported in [-8..7]. Truncate to 4-bit two's complement.
            ap_int<4> val_0 = ap_int<4>(data_0[m]);  // low nibble (out = 2*n)
            ap_int<4> val_1 = ap_int<4>(data_1[m]);  // high nibble (out = 2*n+1)

            ap_int<8> pack_val = 0;
            pack_val.range(3, 0) = val_0;  // low  nibble
            pack_val.range(7, 4) = val_1;  // high nibble

            // Flat mmap index: [mmap_offset + tile][m]
            weight_mmaps[block_id][mmap_offset + tile * input_dim + m][lane] = pack_val;
        }
    }
}


template <int token_parallel, int hidden_dim, int max_seq_len = MAX_PRE_SEQ_LEN>
void prefilling_read_embedding_from_bin(
    const std::string& binfile_name,
    std::vector<int> token_idx,
    std::vector<hls::vector<float, token_parallel>, tapa::aligned_allocator<hls::vector<float, token_parallel>>>& embed_mmap,
    int mmap_offset,
    bool read_last_one = true
) {
    int token_num = read_last_one ? token_idx.size() : token_idx.size() - 1;
    if (token_num > max_seq_len) {
        token_num = max_seq_len;
        cout << "Warning: token_num exceeds max_seq_len, truncated to " << max_seq_len << "\n";
    }

    // Zero out the vectors
    for (size_t idx = 0; idx < embed_mmap.size(); ++idx) {
        embed_mmap[idx] = 0.0f;
    }

    char name[256];
    std::snprintf(name, sizeof(name), "parameters/%s.bin", binfile_name.c_str());
    std::ifstream fin(name, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open file: " << name << "\n";
        return;
    }

    for (int n = 0; n < token_num; ++n) {
        std::vector<float> data(hidden_dim);

        const std::streamoff off = static_cast<std::streamoff>(token_idx[n]) * hidden_dim * sizeof(float);

        fin.seekg(off, std::ios::beg);
        fin.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
        if (!fin) { std::cerr << "Read error @ token_idx " << (token_idx[n]) << " in " << name << "\n"; return; }

        // Pack and write to mmap
        const int tile  = n / token_parallel;
        const int lane  = n % token_parallel;
        for (int m = 0; m < hidden_dim; ++m) {
            embed_mmap[mmap_offset + tile * hidden_dim + m][lane] = data[m];
        }
    }
}


template <int block_parallel, int weight_parallel, int input_dim, int output_dim, int unroll_num = 1>
void decoding_read_int4_bin_as_int8_weight_mmap(
    const std::string& layer_name,
    int layer,
    std::vector<hls::vector<ap_int<8>, weight_parallel/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, weight_parallel/2>>> weight_mmaps[block_parallel/unroll_num],
    int mmap_offset,
    int unroll_id = 0,
    int out_dim_offset = 0
) {
    static_assert(weight_parallel % 2 == 0, "weight_parallel must be even");

    char name[256];
    if(layer_name == "lm_head"){
        std::snprintf(name, sizeof(name), "parameters/%s.bin", layer_name.c_str());
    }
    else{
        std::snprintf(name, sizeof(name), "parameters/%s_L%02d.bin", layer_name.c_str(), layer);
    }
    std::ifstream fin(name, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open file: " << name << "\n";
        return;
    }

    // Each iteration packs two output channels: (2*n, 2*n+1)
    for (int block_id = 0; block_id < block_parallel/unroll_num; ++block_id) {
        int actual_block_id = unroll_id * (block_parallel/unroll_num) + block_id;
        for (int n = 0; n < (output_dim/block_parallel) / 2; ++n) {
            // read one whole row (input_dim bytes) for each of the two output channels
            std::vector<int8_t> data_0(input_dim);
            std::vector<int8_t> data_1(input_dim);

            const std::streamoff off0 = static_cast<std::streamoff>( actual_block_id * (output_dim/block_parallel) + (2*n)     ) * input_dim * sizeof(int8_t);
            const std::streamoff off1 = static_cast<std::streamoff>( actual_block_id * (output_dim/block_parallel) + (2*n + 1) ) * input_dim * sizeof(int8_t);

            fin.seekg(off0, std::ios::beg);
            fin.read(reinterpret_cast<char*>(data_0.data()), data_0.size() * sizeof(int8_t));
            if (!fin) { std::cerr << "Read error @ row " << actual_block_id * (output_dim/block_parallel) + (2*n) << " in " << name << "\n"; return; }

            fin.seekg(off1, std::ios::beg);
            fin.read(reinterpret_cast<char*>(data_1.data()), data_1.size() * sizeof(int8_t));
            if (!fin) { std::cerr << "Read error @ row " << actual_block_id * (output_dim/block_parallel) + (2*n+1) << " in " << name << "\n"; return; }

            
            // Pack and write to mmap
            const int tile  = (n + out_dim_offset/2)  / (weight_parallel / 2);
            const int lane  = (n + out_dim_offset/2)  % (weight_parallel / 2);
            for (int m = 0; m < input_dim; ++m) {
                // Values were exported in [-8..7]. Truncate to 4-bit two's complement.
                ap_int<4> val_0 = ap_int<4>(data_0[m]);  // low nibble (out = 2*n)
                ap_int<4> val_1 = ap_int<4>(data_1[m]);  // high nibble (out = 2*n+1)

                ap_int<8> pack_val = 0;
                pack_val.range(3, 0) = val_0;  // low  nibble
                pack_val.range(7, 4) = val_1;  // high nibble

                // Flat mmap index: [mmap_offset + tile][m]
                weight_mmaps[block_id][mmap_offset + tile * input_dim + m][lane] = pack_val;
            }
        }
    }
}


template <int block_parallel, int hidden_dim, int vocab_size>
void decoding_read_embedding_lib_from_bin(
    const std::string& binfile_name,
    std::vector<hls::vector<float, block_parallel>, tapa::aligned_allocator<hls::vector<float, block_parallel>>>& embed_lib
) {
    char name[256];
    std::snprintf(name, sizeof(name), "parameters/%s.bin", binfile_name.c_str());
    std::ifstream fin(name, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open file: " << name << "\n";
        return;
    }

    for (int n = 0; n < vocab_size; ++n) {
        std::vector<float> data(hidden_dim);

        const std::streamoff off = static_cast<std::streamoff>(n) * hidden_dim * sizeof(float);

        fin.seekg(off, std::ios::beg);
        fin.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
        if (!fin) { std::cerr << "Read error @ token_idx " << (n) << " in " << name << "\n"; return; }

        // Pack and write to mmap
        for (int m = 0; m < hidden_dim; ++m) {
            embed_lib[n * (hidden_dim/block_parallel) + m % (hidden_dim/block_parallel)][m / (hidden_dim/block_parallel)] = data[m];
        }
    }
}


int SpinQuant_Prefilling_Decoding_test(int argc, char* argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> distF(-1.0f/HIDDEN_DIM, 1.0f/HIDDEN_DIM);
    std::uniform_int_distribution<int>   dist4(-8, 7);
    std::uniform_int_distribution<int>   dist8(-128, 127);

    // Float initializer
    auto initFloatVec = [&](auto &cont, int width) {
        for (auto &vec : cont)
            for (int i = 0; i < width; ++i)
                vec[i] = distF(rng);
    };
    // Int4 initializer
    auto initInt4Vec = [&](auto &cont, int width) {
        for (auto &vec : cont)
            for (int i = 0; i < width; ++i)
                vec[i] = ap_int<4>(dist4(rng));
    };
    // Int8 initializer
    auto initInt8Vec = [&](auto &cont, int width) {
        for (auto &vec : cont)
            for (int i = 0; i < width; ++i)
                vec[i] = ap_int<8>(dist8(rng));
    };

    // 1) prefilling 
    cout << "Prefilling parameters:\n";
    cout << "Token Parallel: " << TOKEN_PARALLEL << endl;
    cout << "Pre QKVO Weight Parallel: " << PRE_QKVO_W_PARALLEL << endl;
    cout << "Pre QKVO Weight Parallel Read: " << PRE_QKVO_W_PARALLEL_READ << endl;
    cout << "Pre K Parallel: " << PRE_K_PARALLEL << endl;
    cout << "Pre V Parallel: " << PRE_V_PARALLEL << endl;
    cout << "Pre FFN Weight Parallel: " << PRE_FFN_W_PARALLEL << endl;
    cout << "Pre FFN Weight Parallel Read: " << PRE_FFN_W_PARALLEL_READ << endl;

    // Input/Output mmap
    vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> pref_io_mmap(
        (DECODER_LAYER_NUM + 1) * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM
    );
    cout << "pref_io_mmap size: " << pref_io_mmap.size() << endl;

    // Linear Layer QKVO weight mmap
    vector<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>> wk_wq_mmap(
        DECODER_LAYER_NUM * ((KV_HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL + (HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL)*HIDDEN_DIM
    );
    cout << "wk_wq_mmap size: " << wk_wq_mmap.size() << endl;
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> wk_wq_s_sum_mmap(DECODER_LAYER_NUM *  (KV_HIDDEN_DIM + HIDDEN_DIM));
    cout << "wk_wq_s_sum_mmap size: " << wk_wq_s_sum_mmap.size() << endl;

    vector<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>> wv_wo_mmap(
        DECODER_LAYER_NUM * ((KV_HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL + (HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL)*HIDDEN_DIM
    );
    cout << "wv_wo_mmap size: " << wv_wo_mmap.size() << endl;
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> wv_wo_s_sum_mmap(DECODER_LAYER_NUM * (KV_HIDDEN_DIM + HIDDEN_DIM));
    cout << "wv_wo_s_sum_mmap size: " << wv_wo_s_sum_mmap.size() << endl;
    
    // MHA
    vector<hls::vector<ap_int<8>, PRE_K_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_K_PARALLEL>>> pref_k_cache(
        DECODER_LAYER_NUM * KV_HEAD_NUM * MAX_PRE_SEQ_LEN/PRE_K_PARALLEL * HEAD_DIM
    );
    cout << "k_cache size: " << pref_k_cache.size() << endl;

    vector<hls::vector<ap_int<8>, PRE_V_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_V_PARALLEL>>> pref_v_cache(
        DECODER_LAYER_NUM * KV_HIDDEN_DIM/PRE_V_PARALLEL * MAX_PRE_SEQ_LEN
    );
    cout << "v_cache size: " << pref_v_cache.size() << endl;

    // FFN
    vector<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM/2>>> w_ffn_gate_mmaps[PRE_FFN_W_BLOCK_NUM];
    for(int i = 0; i < PRE_FFN_W_BLOCK_NUM; ++i) {
        w_ffn_gate_mmaps[i].resize(
            DECODER_LAYER_NUM * ((INTER_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * HIDDEN_DIM
        );
    }
    cout << "w_ffn_gate_mmaps size: " << w_ffn_gate_mmaps[0].size() << endl;
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> w_ffn_gate_s_sum_mmap(DECODER_LAYER_NUM * INTER_DIM);
    cout << "w_ffn_gate_s_sum_mmap size: " << w_ffn_gate_s_sum_mmap.size() << endl;

    vector<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM/2>>> w_ffn_up_mmaps[PRE_FFN_W_BLOCK_NUM];
    for(int i = 0; i < PRE_FFN_W_BLOCK_NUM; ++i) {
        w_ffn_up_mmaps[i].resize(
            DECODER_LAYER_NUM * ((INTER_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * HIDDEN_DIM
        );
    }
    cout << "w_ffn_up_mmaps size: " << w_ffn_up_mmaps[0].size() << endl;
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> w_ffn_up_s_sum_mmap(DECODER_LAYER_NUM * INTER_DIM);
    cout << "w_ffn_up_s_sum_mmap size: " << w_ffn_up_s_sum_mmap.size() << endl;

    vector<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM/2>>> w_ffn_down_mmaps[PRE_FFN_W_BLOCK_NUM];
    for(int i = 0; i < PRE_FFN_W_BLOCK_NUM; ++i) {
        w_ffn_down_mmaps[i].resize(
            DECODER_LAYER_NUM * ((HIDDEN_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * INTER_DIM
        );
    }
    cout << "w_ffn_down_mmaps size: " << w_ffn_down_mmaps[0].size() << endl;
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> w_ffn_down_s_sum_mmap(DECODER_LAYER_NUM * HIDDEN_DIM);
    cout << "w_ffn_down_s_sum_mmap size: " << w_ffn_down_s_sum_mmap.size() << endl;

    // Layer Norm weight mmap
    vector<float, tapa::aligned_allocator<float>> gamma_beta_mmap_0(DECODER_LAYER_NUM * HIDDEN_DIM);
    cout << "gamma_beta_mmap_0 size: " << gamma_beta_mmap_0.size() << endl;
    vector<float, tapa::aligned_allocator<float>> gamma_beta_mmap_1(DECODER_LAYER_NUM * HIDDEN_DIM);
    cout << "gamma_beta_mmap_1 size: " << gamma_beta_mmap_1.size() << endl;

    // // Residual cache
    // vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> res0_cache_mmap(MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM);
    // cout << "res0_cache_mmap size: " << res0_cache_mmap.size() << endl;
    // vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> res1_cache_mmap(MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM);
    // cout << "res1_cache_mmap size: " << res1_cache_mmap.size() << endl;
    
    // Initialize buffers
    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        prefilling_read_int4_bin_as_int8_weight_mmap<PRE_QKVO_W_PARALLEL_READ, PRE_QKVO_W_PARALLEL, HIDDEN_DIM, KV_HIDDEN_DIM>(
            "k_proj", i, wk_wq_mmap, i * ((KV_HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL) * HIDDEN_DIM
        );
    }
    cout << "Prefilling: Finished reading k_proj weights." << endl;

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        prefilling_read_int4_bin_as_int8_weight_mmap<PRE_QKVO_W_PARALLEL_READ, PRE_QKVO_W_PARALLEL, HIDDEN_DIM, HIDDEN_DIM>(
            "q_proj", i, wk_wq_mmap, DECODER_LAYER_NUM * ((KV_HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL) * HIDDEN_DIM + i * ((HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL) * HIDDEN_DIM
        );
    }
    cout << "Prefilling: Finished reading q_proj weights." << endl;

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        prefilling_read_int4_bin_as_int8_weight_mmap<PRE_QKVO_W_PARALLEL_READ, PRE_QKVO_W_PARALLEL, HIDDEN_DIM, KV_HIDDEN_DIM>(
            "v_proj", i, wv_wo_mmap, i * ((KV_HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL) * HIDDEN_DIM
        );
    }
    cout << "Prefilling: Finished reading v_proj weights." << endl;

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        prefilling_read_int4_bin_as_int8_weight_mmap<PRE_QKVO_W_PARALLEL_READ, PRE_QKVO_W_PARALLEL, HIDDEN_DIM, HIDDEN_DIM>(
            "o_proj", i, wv_wo_mmap, DECODER_LAYER_NUM * ((KV_HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL) * HIDDEN_DIM + i * ((HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL) * HIDDEN_DIM
        );
    }
    cout << "Prefilling: Finished reading o_proj weights." << endl;

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        prefilling_read_int4_bin_as_blocked_int8_weight_mmaps<PRE_FFN_W_PARALLEL_READ, PRE_FFN_W_PARALLEL, HIDDEN_DIM, INTER_DIM, PRE_FFN_W_BLOCK_NUM>(
            "gate_proj", i, w_ffn_gate_mmaps, i * ((INTER_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * HIDDEN_DIM
        );
    }
    cout << "Prefilling: Finished reading gate_proj weights." << endl;

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        prefilling_read_int4_bin_as_blocked_int8_weight_mmaps<PRE_FFN_W_PARALLEL_READ, PRE_FFN_W_PARALLEL, HIDDEN_DIM, INTER_DIM, PRE_FFN_W_BLOCK_NUM>(
            "up_proj", i, w_ffn_up_mmaps, i * ((INTER_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * HIDDEN_DIM
        );
    }
    cout << "Prefilling: Finished reading up_proj weights." << endl;

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        prefilling_read_int4_bin_as_blocked_int8_weight_mmaps<PRE_FFN_W_PARALLEL_READ, PRE_FFN_W_PARALLEL, INTER_DIM, HIDDEN_DIM, PRE_FFN_W_BLOCK_NUM>(
            "down_proj", i, w_ffn_down_mmaps, i * ((HIDDEN_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * INTER_DIM
        );
    }
    cout << "Prefilling: Finished reading down_proj weights." << endl;



    #include "parameters/w_k_proj_s_sum.h"
    #include "parameters/w_v_proj_s_sum.h"
    #include "parameters/w_q_proj_s_sum.h"
    #include "parameters/w_o_proj_s_sum.h"
    #include "parameters/w_gate_proj_s_sum.h"
    #include "parameters/w_up_proj_s_sum.h"
    #include "parameters/w_down_proj_s_sum.h"
    #include "parameters/w_lm_head_lm_head.h"
    #include "parameters/w_rmsnorm.h"
    
    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        int bias = i * KV_HIDDEN_DIM;
        for(int j = 0; j < KV_HIDDEN_DIM; j++){
            wk_wq_s_sum_mmap[bias + j][0] = w_k_proj_s[i][j];
            wk_wq_s_sum_mmap[bias + j][1] = w_k_proj_sum[i][j];
            wv_wo_s_sum_mmap[bias + j][0] = w_v_proj_s[i][j];
            wv_wo_s_sum_mmap[bias + j][1] = w_v_proj_sum[i][j];
        }
    }

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        int bias = DECODER_LAYER_NUM * KV_HIDDEN_DIM + i * HIDDEN_DIM;
        for(int j = 0; j < HIDDEN_DIM; j++){
            wk_wq_s_sum_mmap[bias + j][0] = w_q_proj_s[i][j];
            wk_wq_s_sum_mmap[bias + j][1] = w_q_proj_sum[i][j];
            wv_wo_s_sum_mmap[bias + j][0] = w_o_proj_s[i][j];
            wv_wo_s_sum_mmap[bias + j][1] = w_o_proj_sum[i][j];
        }
    }

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        for(int j = 0; j < INTER_DIM; j++){
            w_ffn_gate_s_sum_mmap[i * INTER_DIM + j][0] = w_gate_proj_s[i][j];
            w_ffn_gate_s_sum_mmap[i * INTER_DIM + j][1] = w_gate_proj_sum[i][j];
            w_ffn_up_s_sum_mmap[i * INTER_DIM + j][0] = w_up_proj_s[i][j];
            w_ffn_up_s_sum_mmap[i * INTER_DIM + j][1] = w_up_proj_sum[i][j];
        }
    }

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++){
            w_ffn_down_s_sum_mmap[i * HIDDEN_DIM + j][0] = w_down_proj_s[i][j];
            w_ffn_down_s_sum_mmap[i * HIDDEN_DIM + j][1] = w_down_proj_sum[i][j];
        }
    }

    cout << "Prefilling: Finished reading all weights' scaling factors." << endl;

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++){
            gamma_beta_mmap_0[i * HIDDEN_DIM + j] = RMSNorm_weight[2 * i][j];
            gamma_beta_mmap_1[i * HIDDEN_DIM + j] = RMSNorm_weight[2 * i + 1][j];
        }
    }
    cout << "Prefilling: Finished reading layer norm weights." << endl;
       
    
    // 2) decoding
    cout << "Decoding parameters:\n";
    cout << "Token Block Parallel: " << T_BLOCK_PARALLEL << endl;
    cout << "Token QKVO FFN LM_HEAD Block Parallel: " << T_QKVO_FFN_BLOCK_PARALLEL << endl;
    cout << "Dec QKVO FFN LM_HEAD Weight Parallel: " << DEC_QKVO_FFN_W_PARALLEL << endl;
    cout << "Dec MHA Head Parallel: " << DEC_HEAD_PARALLEL << endl;
    cout << "Dec MHA K Weight Parallel: " << DEC_K_PARALLEL << endl;
    cout << "Dec MHA V Weight Parallel: " << DEC_V_PARALLEL << endl;
    
    // random seeds
    vector<float, tapa::aligned_allocator<float>> rand_seeds_mmap(MAX_DEC_SEQ_LEN);

    // decoder token idx
    vector<int, tapa::aligned_allocator<int>> sampled_token_idx_mmap(MAX_DEC_SEQ_LEN);

    // vocab library
    vector<hls::vector<float, T_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_BLOCK_PARALLEL>>> vocab_lib(
        VOCAB_SIZE * HIDDEN_DIM / T_BLOCK_PARALLEL
    );
    cout << "vocab_lib size: " << vocab_lib.size() << endl;

    // Input/Output mmap
    static vector<hls::vector<float, T_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_BLOCK_PARALLEL>>> dec_io_mmap(
        MAX_DEC_SEQ_LEN * (DECODER_LAYER_NUM + 1) * HIDDEN_DIM / T_BLOCK_PARALLEL
    );
    cout << "io_mmap size: " << dec_io_mmap.size() << endl;

    // Linear Layer QKVO weight mmap
    cout << "w_qkvo_FFN_size: " << w_qkvo_FFN_size << endl;
    // vector<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>>> w_qkvo_FFN_mmaps[T_QKVO_FFN_BLOCK_PARALLEL];
    // for (int i = 0; i < T_QKVO_FFN_BLOCK_PARALLEL; i++) {
    //     w_qkvo_FFN_mmaps[i].resize(w_qkvo_FFN_size);
    // }

    // vector<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>>> w_qkvo_FFN_mmaps_half_0[T_QKVO_FFN_BLOCK_PARALLEL/2];
    // for (int i = 0; i < T_QKVO_FFN_BLOCK_PARALLEL/2; i++) {
    //     w_qkvo_FFN_mmaps_half_0[i].resize(w_qkvo_FFN_size);
    // }
    // vector<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>>> w_qkvo_FFN_mmaps_half_1[T_QKVO_FFN_BLOCK_PARALLEL/2];
    // for (int i = 0; i < T_QKVO_FFN_BLOCK_PARALLEL/2; i++) {
    //     w_qkvo_FFN_mmaps_half_1[i].resize(w_qkvo_FFN_size);
    // }


    // Linear Layer QKVO weight_s_sum mmap
    cout << "w_qkvo_FFN_s_size: " << w_s_qkvo_FFN_size << endl;
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> w_s_sum_qkvo_FFN_mmap(
        w_s_qkvo_FFN_size
    );

    // KV caches
    const int k_caches_size = DECODER_LAYER_NUM * KV_HEAD_NUM/DEC_HEAD_PARALLEL * MAX_SUM_SEQ_LEN/DEC_K_PARALLEL * HEAD_DIM;
    cout << "k_caches size: " << k_caches_size << endl;
    const int v_caches_size = DECODER_LAYER_NUM * (KV_HEAD_NUM/DEC_HEAD_PARALLEL * HEAD_DIM/DEC_V_PARALLEL) * MAX_SUM_SEQ_LEN;
    cout << "v_caches size: " << v_caches_size << endl;
    // vector<hls::vector<ap_int<8>, DEC_K_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_K_PARALLEL>>> k_caches[DEC_HEAD_PARALLEL];
    // for(int i = 0; i < DEC_HEAD_PARALLEL; i++){
    //     k_caches[i].resize(DECODER_LAYER_NUM * KV_HEAD_NUM/DEC_HEAD_PARALLEL * MAX_SUM_SEQ_LEN/DEC_K_PARALLEL * HEAD_DIM);
    // }
    // cout << "k_caches size: " << k_caches[0].size() << endl;

    // vector<hls::vector<ap_int<8>, DEC_V_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_V_PARALLEL>>> v_caches[DEC_HEAD_PARALLEL];
    // for(int i = 0; i < DEC_HEAD_PARALLEL; i++){
    //     v_caches[i].resize(DECODER_LAYER_NUM * (KV_HEAD_NUM/DEC_HEAD_PARALLEL * HEAD_DIM/DEC_V_PARALLEL) * MAX_SUM_SEQ_LEN);
    // }
    // cout << "v_caches size: " << v_caches[0].size() << endl;

    vector<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>>> w_qkvo_FFN_mmaps_half_0_k_caches[T_QKVO_FFN_BLOCK_PARALLEL/2];
    for (int i = 0; i < T_QKVO_FFN_BLOCK_PARALLEL/2; i++) {
        w_qkvo_FFN_mmaps_half_0_k_caches[i].resize(w_qkvo_FFN_size + k_caches_size);
    }
    vector<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>>> w_qkvo_FFN_mmaps_half_1_v_caches[T_QKVO_FFN_BLOCK_PARALLEL/2];
    for (int i = 0; i < T_QKVO_FFN_BLOCK_PARALLEL/2; i++) {
        w_qkvo_FFN_mmaps_half_1_v_caches[i].resize(w_qkvo_FFN_size + v_caches_size);
    }

    // Layer Norm weight mmap
    vector<hls::vector<float, T_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_BLOCK_PARALLEL>>> gamma_beta_mmap(
        (2 * DECODER_LAYER_NUM + 1) * HIDDEN_DIM / T_BLOCK_PARALLEL
    );
    cout << "gamma_beta_mmap size: " << gamma_beta_mmap.size() << endl;

    // Initialize buffers with random data

    // initialize all the vocab_lib vectors and weights
    // initFloatVec(vocab_lib, T_BLOCK_PARALLEL);
    decoding_read_embedding_lib_from_bin<T_BLOCK_PARALLEL, HIDDEN_DIM, VOCAB_SIZE>(
        "model_embed_tokens_fp32", vocab_lib
    );
    cout << "Decoding: Finished reading vocab lib" << endl;


    for (int i = 0; i < DECODER_LAYER_NUM; i++) {
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, KV_HIDDEN_DIM, 2>(
            "k_proj", i, w_qkvo_FFN_mmaps_half_0_k_caches, w_kv_addr_bias + i * KV_HIDDEN_DIM_PAD / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 0
        );
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, KV_HIDDEN_DIM, 2>(
            "k_proj", i, w_qkvo_FFN_mmaps_half_1_v_caches, w_kv_addr_bias + i * KV_HIDDEN_DIM_PAD / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 1
        );
    }
    cout << "Decoding: Finished reading k_proj weights." << endl;

    for (int i = 0; i < DECODER_LAYER_NUM; i++) {
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, KV_HIDDEN_DIM, 2>(
            "v_proj", i, w_qkvo_FFN_mmaps_half_0_k_caches, w_kv_addr_bias + i * KV_HIDDEN_DIM_PAD / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 0, KV_HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL
        );
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, KV_HIDDEN_DIM, 2>(
            "v_proj", i, w_qkvo_FFN_mmaps_half_1_v_caches, w_kv_addr_bias + i * KV_HIDDEN_DIM_PAD / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 1, KV_HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL
        );
    }
    cout << "Decoding: Finished reading v_proj weights." << endl;

    for (int i = 0; i < DECODER_LAYER_NUM; i++) {
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, HIDDEN_DIM, 2>(
            "q_proj", i, w_qkvo_FFN_mmaps_half_0_k_caches, w_q_addr_bias + i * HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 0
        );
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, HIDDEN_DIM, 2>(
            "q_proj", i, w_qkvo_FFN_mmaps_half_1_v_caches, w_q_addr_bias + i * HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 1
        );
    }
    cout << "Decoding: Finished reading q_proj weights." << endl;

    for (int i = 0; i < DECODER_LAYER_NUM; i++) {
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, HIDDEN_DIM, 2>(
            "o_proj", i, w_qkvo_FFN_mmaps_half_0_k_caches, w_o_addr_bias + i * HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 0
        );
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, HIDDEN_DIM, 2>(
            "o_proj", i, w_qkvo_FFN_mmaps_half_1_v_caches, w_o_addr_bias + i * HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 1
        );
    }
    cout << "Decoding: Finished reading o_proj weights." << endl;

    for (int i = 0; i < DECODER_LAYER_NUM; i++) {
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, INTER_DIM, 2>(
            "up_proj", i, w_qkvo_FFN_mmaps_half_0_k_caches, w_ffn_up_addr_bias + i * INTER_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 0
        );
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, INTER_DIM, 2>(
            "up_proj", i, w_qkvo_FFN_mmaps_half_1_v_caches, w_ffn_up_addr_bias + i * INTER_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 1
        );
    }
    cout << "Decoding: Finished reading up_proj weights." << endl;

    for (int i = 0; i < DECODER_LAYER_NUM; i++) {
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, INTER_DIM, 2>(
            "gate_proj", i, w_qkvo_FFN_mmaps_half_0_k_caches, w_ffn_gate_addr_bias + i * INTER_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 0
        );
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, INTER_DIM, 2>(
            "gate_proj", i, w_qkvo_FFN_mmaps_half_1_v_caches, w_ffn_gate_addr_bias + i * INTER_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 1
        );
    }
    cout << "Decoding: Finished reading gate_proj weights." << endl;

    for (int i = 0; i < DECODER_LAYER_NUM; i++) {
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, HIDDEN_DIM, 2>(
            "down_proj", i, w_qkvo_FFN_mmaps_half_0_k_caches, w_ffn_down_addr_bias + i * HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * INTER_DIM, 0
        );
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, HIDDEN_DIM, 2>(
            "down_proj", i, w_qkvo_FFN_mmaps_half_1_v_caches, w_ffn_down_addr_bias + i * HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * INTER_DIM, 1
        );
    }
    cout << "Decoding: Finished reading down_proj weights." << endl;

    decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, VOCAB_SIZE_PAD, 2>(
        "lm_head", 0, w_qkvo_FFN_mmaps_half_0_k_caches, w_vocab_addr_bias, 0
    );
    decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, VOCAB_SIZE_PAD, 2>(
        "lm_head", 0, w_qkvo_FFN_mmaps_half_1_v_caches, w_vocab_addr_bias, 1
    );
    cout << "Decoding: Finished reading lm_head weights." << endl;

    // initFloatVec(w_s_sum_qkvo_FFN_mmap, 2);
    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        for(int t = 0; t < T_QKVO_FFN_BLOCK_PARALLEL; t++){
            int bias_k = w_s_kv_addr_bias + i * KV_HIDDEN_DIM_PAD + t * KV_HIDDEN_DIM_PAD/T_QKVO_FFN_BLOCK_PARALLEL;
            int bias_v = bias_k + KV_HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL;
            for(int j = 0; j < KV_HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL; j++){
                w_s_sum_qkvo_FFN_mmap[bias_k + j][0] = w_k_proj_s[i][t * KV_HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL + j];
                w_s_sum_qkvo_FFN_mmap[bias_k + j][1] = w_k_proj_sum[i][t * KV_HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL + j];
                w_s_sum_qkvo_FFN_mmap[bias_v + j][0] = w_v_proj_s[i][t * KV_HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL + j];
                w_s_sum_qkvo_FFN_mmap[bias_v + j][1] = w_v_proj_sum[i][t * KV_HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL + j];
            }
        }
    }

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        int bias = w_s_q_addr_bias + i * HIDDEN_DIM;
        for(int j = 0; j < HIDDEN_DIM; j++){
            w_s_sum_qkvo_FFN_mmap[bias + j][0] = w_q_proj_s[i][j];
            w_s_sum_qkvo_FFN_mmap[bias + j][1] = w_q_proj_sum[i][j];
        }
    }

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        int bias = w_s_o_addr_bias + i * HIDDEN_DIM;
        for(int j = 0; j < HIDDEN_DIM; j++){
            w_s_sum_qkvo_FFN_mmap[bias + j][0] = w_o_proj_s[i][j];
            w_s_sum_qkvo_FFN_mmap[bias + j][1] = w_o_proj_sum[i][j];
        }
    }

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        int bias = w_s_ffn_up_addr_bias + i * INTER_DIM;
        for(int j = 0; j < INTER_DIM; j++){
            w_s_sum_qkvo_FFN_mmap[bias + j][0] = w_up_proj_s[i][j];
            w_s_sum_qkvo_FFN_mmap[bias + j][1] = w_up_proj_sum[i][j];
        }
    }

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        int bias = w_s_ffn_gate_addr_bias + i * INTER_DIM;
        for(int j = 0; j < INTER_DIM; j++){
            w_s_sum_qkvo_FFN_mmap[bias + j][0] = w_gate_proj_s[i][j];
            w_s_sum_qkvo_FFN_mmap[bias + j][1] = w_gate_proj_sum[i][j];
        }
    }

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        int bias = w_s_ffn_down_addr_bias + i * HIDDEN_DIM;
        for(int j = 0; j < HIDDEN_DIM; j++){
            w_s_sum_qkvo_FFN_mmap[bias + j][0] = w_down_proj_s[i][j];
            w_s_sum_qkvo_FFN_mmap[bias + j][1] = w_down_proj_sum[i][j];
        }
    }

    for(int j = 0; j < VOCAB_SIZE; j++){
        w_s_sum_qkvo_FFN_mmap[w_s_vocab_addr_bias + j][0] = w_lm_head_lm_head_s[j];
        w_s_sum_qkvo_FFN_mmap[w_s_vocab_addr_bias + j][1] = w_lm_head_lm_head_sum[j];
    }
    cout << "Decoding: Finished reading weight_s_sum." << endl;


    // initFloatVec(gamma_beta_mmap, T_BLOCK_PARALLEL);
    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        for(int k = 0; k < T_BLOCK_PARALLEL; k++){
            for(int j = 0; j < HIDDEN_DIM/T_BLOCK_PARALLEL; j++){
                gamma_beta_mmap[i * HIDDEN_DIM/T_BLOCK_PARALLEL + j][k] = RMSNorm_weight[2 * i][k * HIDDEN_DIM/T_BLOCK_PARALLEL + j];
                gamma_beta_mmap[(DECODER_LAYER_NUM + i) * HIDDEN_DIM/T_BLOCK_PARALLEL + j][k] = RMSNorm_weight[2 * i + 1][k * HIDDEN_DIM/T_BLOCK_PARALLEL + j];
            }
        }
    }
    for(int k = 0; k < T_BLOCK_PARALLEL; k++){
        for(int j = 0; j < HIDDEN_DIM/T_BLOCK_PARALLEL; j++){
            gamma_beta_mmap[(2 * DECODER_LAYER_NUM) * HIDDEN_DIM/T_BLOCK_PARALLEL + j][k] = RMSNorm_weight[(2 * DECODER_LAYER_NUM)][k * HIDDEN_DIM/T_BLOCK_PARALLEL + j];
        }
    }
    cout << "Decoding: Finished reading RMSNorm weights." << endl;


    // 3) set the model tokenizer
    // install logger before loading model
    llama_log_set(llama_silent_log, nullptr);
    
    // defaults
    static const char *MODEL_PATH = "llama-3.2-1b-f16.gguf";
    std::string in_path = "my_prompt.txt";
    std::string out_path = "my_answer.txt";

    // override from CLI
    // usage: ./tokenizer_test [input.txt] [output.txt]
    if (argc >= 2) MODEL_PATH = argv[1];
    if (argc >= 3) in_path = argv[2];
    if (argc >= 4) out_path = argv[3];

    // load model (vocab only)
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model *model = llama_model_load_from_file(MODEL_PATH, mparams);
    if (!model) {
        std::cerr << "Failed to load model: " << MODEL_PATH << "\n";
        return 1;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    if (!vocab) {
        std::cerr << "Failed to get vocab\n";
        llama_model_free(model);
        return 1;
    }

    // 3) Run the prefilling and decoding kernels
    
    const int num_runs = 100;
    int64_t pref_total_time_ns = 0;
    int64_t dec_total_time_ns = 0;
    std::cout << "kernel begins running " << num_runs << " times …\n";
    
    for (int run = 0; run < num_runs; ++run) {
    // Call Linear_Layer_tb
    // Apply initialization
        cout << "\n --- Ready for next request (q to quit) ---\n";
        char run_char;
        cin >> run_char;
        if (run_char == 'q') {
            break;
        }

        // Zero out the io vectors
        for (int idx = 0; idx < pref_io_mmap.size(); idx++) {
            pref_io_mmap[idx] = 0.0f;
        }

        // // encode tokens to ids
        std::string raw_prompt = read_file(in_path);
        if (raw_prompt.empty()) {
            std::cerr << "Failed to read prompt file: " << in_path << "\n";
            return 1;
        }

        // std::cout << "Prompt string (" << in_path << "):\n";
        // std::cout << "----------------------------------------\n";
        // std::cout << prompt << "\n";
        // std::cout << "----------------------------------------\n\n";

        // 3) build messages (system + user), like Python
        std::string system_prompt;
        std::string user_prompt;

        std::size_t pos = raw_prompt.find('#');
        if (pos == std::string::npos) {
            // no '#': fallback
            system_prompt = "You are a helpful assistant.";
            user_prompt = raw_prompt;
        } else {
            system_prompt = raw_prompt.substr(0, pos);
            user_prompt   = raw_prompt.substr(pos + 1);
        }
        // trim both ends a little (very simple trim)
        auto trim = [](std::string &s) {
            while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r' || s.back() == '\t'))
                s.pop_back();
            while (!s.empty() && (s.front() == ' ' || s.front() == '\n' || s.front() == '\r' || s.front() == '\t'))
                s.erase(s.begin());
        };
        trim(system_prompt);
        trim(user_prompt);
        std::vector<Message> messages = {
            {"system", system_prompt},
            {"user",   user_prompt}
        };

        // 4) apply chat template
        std::string chat_text = apply_chat_template(messages, /*add_generation_prompt=*/true);

        std::cout << "Chat text (templated):\n";
        std::cout << "----------------------------------------\n";
        std::cout << chat_text << "\n";
        std::cout << "----------------------------------------\n\n";

        auto token_idx = encode_text(vocab, chat_text,
                           /*add_special=*/false,   // we already added <|begin_of_text|>
                           /*parse_special=*/true); // we used special token strings
        if (token_idx.empty()) {
            std::cerr << "Tokenization failed\n";
            llama_model_free(model);
            return 1;
        }
        // std::cout << "Encoded token IDs (" << token_idx.size() << "):\n";
        // for (auto id : token_idx) std::cout << id << " ";
        // std::cout << "\n\n";

        prefilling_read_embedding_from_bin<TOKEN_PARALLEL, HIDDEN_DIM>("model_embed_tokens_fp32", token_idx, pref_io_mmap, 0, false);
        cout << "Prefilling: Finished reading input embedding." << endl;

        int test_pre_seq_len = token_idx.size() - 1;

        int pad_factor = TOKEN_PARALLEL > PRE_K_PARALLEL ? TOKEN_PARALLEL : PRE_K_PARALLEL;
        int test_pre_seq_len_pad = (token_idx.size() - 1 + pad_factor - 1) / pad_factor * pad_factor;

        // zero out kv caches and residuals
        for (auto &vec : pref_k_cache)      for (int i = 0; i < PRE_K_PARALLEL; ++i) vec[i] = 0;
        for (auto &vec : pref_v_cache)      for (int i = 0; i < PRE_V_PARALLEL; ++i) vec[i] = 0;

        // for (auto &vec : res0_cache_mmap) for (int i = 0; i < TOKEN_PARALLEL; ++i) vec[i] = 0.0f;
        // for (auto &vec : res1_cache_mmap) for (int i = 0; i < TOKEN_PARALLEL; ++i) vec[i] = 0.0f;

        cout << "Prefill kernel begins running!\n";
        int64_t pref_kernel_time_ns = tapa::invoke(
            SpinQuant_Prefilling, 
            FLAGS_bitstream_pref,
            tapa::read_write_mmap<hls::vector<float, TOKEN_PARALLEL>>(pref_io_mmap),
            tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>(wk_wq_mmap),
            tapa::read_only_mmap<hls::vector<float, 2>>(wk_wq_s_sum_mmap),
            tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>(wv_wo_mmap),
            tapa::read_only_mmap<hls::vector<float, 2>>(wv_wo_s_sum_mmap),
            tapa::read_write_mmap<hls::vector<ap_int<8>, PRE_K_PARALLEL>>(pref_k_cache),
            tapa::read_write_mmap<hls::vector<ap_int<8>, PRE_V_PARALLEL>>(pref_v_cache),
            tapa::read_only_mmaps<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM/2>, PRE_FFN_W_BLOCK_NUM>(w_ffn_gate_mmaps),
            tapa::read_only_mmap<hls::vector<float, 2>>(w_ffn_gate_s_sum_mmap),
            tapa::read_only_mmaps<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM/2>, PRE_FFN_W_BLOCK_NUM>(w_ffn_up_mmaps),
            tapa::read_only_mmap<hls::vector<float, 2>>(w_ffn_up_s_sum_mmap),
            tapa::read_only_mmaps<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM/2>, PRE_FFN_W_BLOCK_NUM>(w_ffn_down_mmaps),
            tapa::read_only_mmap<hls::vector<float, 2>>(w_ffn_down_s_sum_mmap),
            tapa::read_only_mmap<float>(gamma_beta_mmap_0),
            tapa::read_only_mmap<float>(gamma_beta_mmap_1),
            // tapa::read_write_mmap<hls::vector<float, TOKEN_PARALLEL>>(res0_cache_mmap),
            // tapa::read_write_mmap<hls::vector<float, TOKEN_PARALLEL>>(res1_cache_mmap),
            // MAX_PRE_SEQ_LEN
            test_pre_seq_len_pad
        );
        
        // double pref_t_s = pref_kernel_time_ns * 1e-9;
        // std::cout << "  Run " << run << " — kernel time: " << pref_t_s << " s\n";
        // pref_total_time_ns += pref_kernel_time_ns;

        // for(int i = 0; i < 32; i++) std::cout << pref_io_mmap[i][0] << " ";
        // std::cout << std::endl;
        // for(int i = 0; i < 32; i++) std::cout << pref_io_mmap[(MAX_PRE_SEQ_LEN / TOKEN_PARALLEL) * HIDDEN_DIM + i][0] << " ";
        // std::cout << std::endl;


        // launch decoding kernel

        // init decoding random seeds fro sampling
        for (int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
            // rand_seeds_mmap[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.8;
            // rand_seeds_mmap[i] = 0.01f;
            float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
            float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
            float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2); // N(0,1)
            float x = std::fabs(z0);
            x = x / 3.0f;
            if (x > 1.0f) x = 0.0f;
            rand_seeds_mmap[i] = x;
        }
        
        // Zero out the io vectors
        for (int idx = 0; idx < dec_io_mmap.size(); idx++) {
            dec_io_mmap[idx] = 0.0f;
        }

        // load the last token_idx[-1] to decoding's input/output buffer
        for (size_t idx = 0; idx < HIDDEN_DIM/T_BLOCK_PARALLEL; ++idx) {
            dec_io_mmap[idx] = vocab_lib[token_idx[token_idx.size() - 1] * (HIDDEN_DIM/T_BLOCK_PARALLEL) + idx];
        }

        // zero out kv caches
        for (int i = 0; i < DEC_HEAD_PARALLEL; i++){
            for(int j = w_qkvo_FFN_size; j < w_qkvo_FFN_mmaps_half_0_k_caches[i].size(); j++){
                for(int k = 0; k < DEC_K_PARALLEL; k++){
                    w_qkvo_FFN_mmaps_half_0_k_caches[i][j][k] = 0;
                }
            }
        }
        for (int i = 0; i < DEC_HEAD_PARALLEL; i++){
            for(int j = w_qkvo_FFN_size; j < w_qkvo_FFN_mmaps_half_1_v_caches[i].size(); j++){
                for(int k = 0; k < DEC_V_PARALLEL; k++){
                    w_qkvo_FFN_mmaps_half_1_v_caches[i][j][k] = 0;
                }
            }
        }


        // init KV caches
        for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
            for (int i = 0; i < test_pre_seq_len; i++) {
                for (int h = 0; h < KV_HEAD_NUM; h++) {
                    for (int j = 0; j < HEAD_DIM; j++) {
                        int write_idx = ((layer * KV_HEAD_NUM/DEC_HEAD_PARALLEL + h % (KV_HEAD_NUM/DEC_HEAD_PARALLEL)) * MAX_SUM_SEQ_LEN + i)/DEC_K_PARALLEL *
                                    HEAD_DIM + j;
                        int write_sub_idx = i % DEC_K_PARALLEL;

                        int read_idx = ((layer * KV_HEAD_NUM + h) * MAX_PRE_SEQ_LEN + i)/PRE_K_PARALLEL * HEAD_DIM + j;
                        int read_sub_idx = i % PRE_K_PARALLEL;

                        w_qkvo_FFN_mmaps_half_0_k_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][w_qkvo_FFN_size + write_idx][write_sub_idx] = pref_k_cache[read_idx][read_sub_idx];
                    
                        write_idx = ((layer * (KV_HEAD_NUM/DEC_HEAD_PARALLEL) + h % (KV_HEAD_NUM/DEC_HEAD_PARALLEL)) * HEAD_DIM + j)/DEC_V_PARALLEL * MAX_SUM_SEQ_LEN + i;
                        write_sub_idx = j % DEC_V_PARALLEL;

                        read_idx = ((layer * KV_HIDDEN_DIM+ h * HEAD_DIM + j)/PRE_V_PARALLEL  * MAX_PRE_SEQ_LEN) + i;
                        read_sub_idx = (h * HEAD_DIM + j) % PRE_V_PARALLEL;

                        w_qkvo_FFN_mmaps_half_1_v_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][w_qkvo_FFN_size + write_idx][write_sub_idx] = pref_v_cache[read_idx][read_sub_idx];
                    }
                }
            }
        }
        cout << "Decoding: Finished loading decoding's input and KV caches." << endl;

        // run the kernel
        cout << "Decode kernel begins running!\n";

        
        int64_t dec_kernel_time_ns = tapa::invoke(
            SpinQuant_Decoding, 
            FLAGS_bitstream_dec,
            tapa::read_only_mmap<hls::vector<float, T_BLOCK_PARALLEL>>(vocab_lib),
            tapa::read_write_mmap<hls::vector<float, T_BLOCK_PARALLEL>>(dec_io_mmap),
            // tapa::read_only_mmaps<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL>(w_qkvo_FFN_mmaps),
            // tapa::read_only_mmaps<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL/2>(w_qkvo_FFN_mmaps_half_0),
            // tapa::read_only_mmaps<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL/2>(w_qkvo_FFN_mmaps_half_1),
            tapa::read_write_mmaps<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, T_QKVO_FFN_BLOCK_PARALLEL/2>(w_qkvo_FFN_mmaps_half_0_k_caches),
            tapa::read_write_mmaps<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, T_QKVO_FFN_BLOCK_PARALLEL/2>(w_qkvo_FFN_mmaps_half_1_v_caches),
            tapa::read_only_mmap<hls::vector<float, 2>>(w_s_sum_qkvo_FFN_mmap),
            // tapa::read_write_mmaps<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL>(k_caches),
            // tapa::read_write_mmaps<hls::vector<ap_int<8>, DEC_V_PARALLEL>, DEC_HEAD_PARALLEL>(v_caches),
            tapa::read_only_mmap<hls::vector<float, T_BLOCK_PARALLEL>>(gamma_beta_mmap),
            tapa::read_only_mmap<float>(rand_seeds_mmap),
            tapa::write_only_mmap<int>(sampled_token_idx_mmap),
            // MAX_PRE_SEQ_LEN,
            test_pre_seq_len,
            MAX_DEC_SEQ_LEN
            // 256
        );
        
        
        // double dec_t_s = dec_kernel_time_ns * 1e-9;
        // std::cout << "  Run " << run << " — kernel time: " << dec_t_s << " s\n";
        // dec_total_time_ns += dec_kernel_time_ns;

        // for(int i = 0; i < 32; i++) std::cout << dec_io_mmap[i][0] << " ";
        // std::cout << std::endl;
        // for(int i = 0; i < 32; i++) std::cout << dec_io_mmap[MAX_DEC_SEQ_LEN * (HIDDEN_DIM/T_BLOCK_PARALLEL) + i][0] << " ";
        // std::cout << std::endl;

        // char fname[256];
        // std::snprintf(fname, sizeof(fname), "my_sampled_token_idx_%d.txt", run);
        // std::ofstream fout(fname);
        // if (!fout) {
        //     std::cerr << "Failed to open " << fname << " for writing\n";
        // } else {
        //     int n = sampled_token_idx_mmap.size();  // or any actual length you want
        //     for (int i = 0; i < n; ++i) {
        //         fout << sampled_token_idx_mmap[i];
        //         if (i + 1 < n) fout << " ";
        //     }
        //     fout << "\n";
        // }

        std::vector<llama_token> ids_for_decode;
        for(int i = 0; i < MAX_DEC_SEQ_LEN - 1; i++) {
            int token_id = sampled_token_idx_mmap[i];
            ids_for_decode.push_back(static_cast<llama_token>(token_id));
            if (token_id == 128009) break;
        }

        // std::cout << "IDs used for decode (" << ids_for_decode.size() << "):\n";
        // for (auto id : ids_for_decode) std::cout << id << " ";
        // std::cout << "\n\n";

        // 5. decode
        std::string decoded = decode_tokens_concat(
            vocab,
            ids_for_decode,
            /*remove_special=*/false,
            /*parse_special=*/true
        );

        std::cout << "Decoded string:\n";
        std::cout << "----------------------------------------\n";
        std::cout << decoded << "\n";
        std::cout << "(generate 1024 tokens in total and truncate by <|eot_id|>)\n";
        std::cout << "----------------------------------------\n";

        // 6. write to file
        if (!write_file(out_path, decoded)) {
            std::cerr << "Failed to write decoded text to: " << out_path << "\n";
            llama_model_free(model);
            return 1;
        }

        std::cout << "Decoded string written to: " << out_path << "\n";

        double pref_t_s = pref_kernel_time_ns * 1e-9;
        std::cout << "U280 FPGA Run " << run << " — prefill kernel time: " << pref_t_s << " s\n";
        std::cout << "U280 FPGA Run " << run << " — prefill sequence length: " << test_pre_seq_len << " \n";
        std::cout << "U280 FPGA Run " << run << " — prefill kernel throughput: " << test_pre_seq_len/pref_t_s << " tokens/s\n";
        // pref_total_time_ns += pref_kernel_time_ns;

        double dec_t_s = dec_kernel_time_ns * 1e-9;
        std::cout << "U280 FPGA Run " << run << " — decode kernel time: " << ids_for_decode.size()/float(MAX_DEC_SEQ_LEN) * dec_t_s <<" / " << dec_t_s << " s\n";
        std::cout << "U280 FPGA Run " << run << " — decode sequence length: " << ids_for_decode.size() << " / " << MAX_DEC_SEQ_LEN << " \n";
        std::cout << "U280 FPGA Run " << run << " — decode kernel throughput: " << MAX_DEC_SEQ_LEN/dec_t_s << " tokens/s\n";
        dec_total_time_ns += dec_kernel_time_ns;
    }

    // double pref_avg_s = (pref_total_time_ns / double(num_runs)) * 1e-9;
    // std::cout << "Average prefilling kernel time over " << num_runs << " runs: " << pref_avg_s << " s\n";

    // double dec_avg_s = (dec_total_time_ns / double(num_runs)) * 1e-9;
    // std::cout << "Average decoding kernel time over " << num_runs << " runs: " << dec_avg_s << " s\n";

    llama_model_free(model);
    return 0;
}

int main(int argc, char* argv[]) {
    return SpinQuant_Prefilling_Decoding_test(argc, argv);
}