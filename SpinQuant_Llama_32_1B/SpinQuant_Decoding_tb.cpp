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


DEFINE_string(bitstream, "", ""/*path to bitstream file, run csim if empty*/);

#include "SpinQuant_Decoding.h"

#include <algorithm>
#include <numeric>
#include <cmath>

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


void SpinQuant_Decoding_test(int argc, char* argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

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
    static vector<hls::vector<float, T_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_BLOCK_PARALLEL>>> io_mmap(
        (MAX_DEC_SEQ_LEN * (DECODER_LAYER_NUM + 1) + 1) * HIDDEN_DIM / T_BLOCK_PARALLEL
    );
    cout << "io_mmap size: " << io_mmap.size() << endl;

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

    vector<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>>> w_qkvo_FFN_mmaps_half_0[T_QKVO_FFN_BLOCK_PARALLEL/2];
    for (int i = 0; i < T_QKVO_FFN_BLOCK_PARALLEL/2; i++) {
        w_qkvo_FFN_mmaps_half_0[i].resize(w_qkvo_FFN_size);
    }
    vector<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>>> w_qkvo_FFN_mmaps_half_1[T_QKVO_FFN_BLOCK_PARALLEL/2];
    for (int i = 0; i < T_QKVO_FFN_BLOCK_PARALLEL/2; i++) {
        w_qkvo_FFN_mmaps_half_1[i].resize(w_qkvo_FFN_size);
    }

    // Linear Layer QKVO weight_s_sum mmap
    cout << "w_s_qkvo_FFN_size: " << w_s_qkvo_FFN_size << endl;
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> w_s_sum_qkvo_FFN_mmap(
        w_s_qkvo_FFN_size
    );

    // KV caches
    vector<hls::vector<ap_int<8>, DEC_K_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_K_PARALLEL>>> k_caches[DEC_HEAD_PARALLEL];
    for(int i = 0; i < DEC_HEAD_PARALLEL; i++){
        k_caches[i].resize(DECODER_LAYER_NUM * KV_HEAD_NUM/DEC_HEAD_PARALLEL * MAX_SUM_SEQ_LEN/DEC_K_PARALLEL * HEAD_DIM);
    }
    cout << "k_caches size: " << k_caches[0].size() << endl;

    vector<hls::vector<ap_int<8>, DEC_V_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_V_PARALLEL>>> v_caches[DEC_HEAD_PARALLEL];
    for(int i = 0; i < DEC_HEAD_PARALLEL; i++){
        v_caches[i].resize(DECODER_LAYER_NUM * (KV_HEAD_NUM/DEC_HEAD_PARALLEL * HEAD_DIM/DEC_V_PARALLEL) * MAX_SUM_SEQ_LEN);
    }
    cout << "v_caches size: " << v_caches[0].size() << endl;

    // Layer Norm weight mmap
    vector<hls::vector<float, T_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_BLOCK_PARALLEL>>> gamma_beta_mmap(
        (2 * DECODER_LAYER_NUM + 1) * HIDDEN_DIM / T_BLOCK_PARALLEL
    );
    cout << "gamma_beta_mmap size: " << gamma_beta_mmap.size() << endl;

    // 2) Initialize buffers with random data
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> distF(-1.0f/sqrt(HIDDEN_DIM), 1.0f/sqrt(HIDDEN_DIM));
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

    // initialize all the vocab_lib vectors and weights
    
    // initFloatVec(vocab_lib, T_BLOCK_PARALLEL);
    decoding_read_embedding_lib_from_bin<T_BLOCK_PARALLEL, HIDDEN_DIM, VOCAB_SIZE>(
        "model_embed_tokens_fp32", vocab_lib
    );
    cout << "Decoding: Finished reading vocab lib" << endl;

    for (int i = 0; i < T_QKVO_FFN_BLOCK_PARALLEL/2; i++) {
        initInt8Vec(w_qkvo_FFN_mmaps_half_0[i], DEC_QKVO_FFN_W_PARALLEL/2);
    }
    for (int i = 0; i < T_QKVO_FFN_BLOCK_PARALLEL/2; i++) {
        initInt8Vec(w_qkvo_FFN_mmaps_half_1[i], DEC_QKVO_FFN_W_PARALLEL/2);
    }


    for (int i = 0; i < DECODER_LAYER_NUM; i++) {
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, KV_HIDDEN_DIM, 2>(
            "k_proj", i, w_qkvo_FFN_mmaps_half_0, w_kv_addr_bias + i * KV_HIDDEN_DIM_PAD / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 0
        );
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, KV_HIDDEN_DIM, 2>(
            "k_proj", i, w_qkvo_FFN_mmaps_half_1, w_kv_addr_bias + i * KV_HIDDEN_DIM_PAD / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 1
        );
    }
    cout << "Decoding: Finished reading k_proj weights." << endl;

    for (int i = 0; i < DECODER_LAYER_NUM; i++) {
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, KV_HIDDEN_DIM, 2>(
            "v_proj", i, w_qkvo_FFN_mmaps_half_0, w_kv_addr_bias + i * KV_HIDDEN_DIM_PAD / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 0, KV_HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL
        );
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, KV_HIDDEN_DIM, 2>(
            "v_proj", i, w_qkvo_FFN_mmaps_half_1, w_kv_addr_bias + i * KV_HIDDEN_DIM_PAD / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 1, KV_HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL
        );
    }
    cout << "Decoding: Finished reading v_proj weights." << endl;

    for (int i = 0; i < DECODER_LAYER_NUM; i++) {
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, HIDDEN_DIM, 2>(
            "q_proj", i, w_qkvo_FFN_mmaps_half_0, w_q_addr_bias + i * HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 0
        );
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, HIDDEN_DIM, 2>(
            "q_proj", i, w_qkvo_FFN_mmaps_half_1, w_q_addr_bias + i * HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 1
        );
    }
    cout << "Decoding: Finished reading q_proj weights." << endl;

    for (int i = 0; i < DECODER_LAYER_NUM; i++) {
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, HIDDEN_DIM, 2>(
            "o_proj", i, w_qkvo_FFN_mmaps_half_0, w_o_addr_bias + i * HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 0
        );
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, HIDDEN_DIM, 2>(
            "o_proj", i, w_qkvo_FFN_mmaps_half_1, w_o_addr_bias + i * HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 1
        );
    }
    cout << "Decoding: Finished reading o_proj weights." << endl;

    for (int i = 0; i < DECODER_LAYER_NUM; i++) {
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, INTER_DIM, 2>(
            "up_proj", i, w_qkvo_FFN_mmaps_half_0, w_ffn_up_addr_bias + i * INTER_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 0
        );
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, INTER_DIM, 2>(
            "up_proj", i, w_qkvo_FFN_mmaps_half_1, w_ffn_up_addr_bias + i * INTER_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 1
        );
    }
    cout << "Decoding: Finished reading up_proj weights." << endl;

    for (int i = 0; i < DECODER_LAYER_NUM; i++) {
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, INTER_DIM, 2>(
            "gate_proj", i, w_qkvo_FFN_mmaps_half_0, w_ffn_gate_addr_bias + i * INTER_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 0
        );
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, INTER_DIM, 2>(
            "gate_proj", i, w_qkvo_FFN_mmaps_half_1, w_ffn_gate_addr_bias + i * INTER_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM, 1
        );
    }
    cout << "Decoding: Finished reading gate_proj weights." << endl;

    for (int i = 0; i < DECODER_LAYER_NUM; i++) {
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, HIDDEN_DIM, 2>(
            "down_proj", i, w_qkvo_FFN_mmaps_half_0, w_ffn_down_addr_bias + i * HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * INTER_DIM, 0
        );
        decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, HIDDEN_DIM, 2>(
            "down_proj", i, w_qkvo_FFN_mmaps_half_1, w_ffn_down_addr_bias + i * HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * INTER_DIM, 1
        );
    }
    cout << "Decoding: Finished reading down_proj weights." << endl;

    decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, VOCAB_SIZE_PAD, 2>(
        "lm_head", 0, w_qkvo_FFN_mmaps_half_0, w_vocab_addr_bias, 0
    );
    decoding_read_int4_bin_as_int8_weight_mmap<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, VOCAB_SIZE_PAD, 2>(
        "lm_head", 0, w_qkvo_FFN_mmaps_half_1, w_vocab_addr_bias, 1
    );
    cout << "Decoding: Finished reading lm_head weights." << endl;


    #include "parameters/w_k_proj_s_sum.h"
    #include "parameters/w_v_proj_s_sum.h"
    #include "parameters/w_q_proj_s_sum.h"
    #include "parameters/w_o_proj_s_sum.h"
    #include "parameters/w_gate_proj_s_sum.h"
    #include "parameters/w_up_proj_s_sum.h"
    #include "parameters/w_down_proj_s_sum.h"
    #include "parameters/w_lm_head_lm_head.h"
    #include "parameters/w_rmsnorm.h"   

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
    

    // 3) run kernel
    size_t io_init_vecs = HIDDEN_DIM/T_BLOCK_PARALLEL;

    const int num_runs = 1;
    int64_t total_time_ns = 0;
    std::cout << "kernel begins running " << num_runs << " times …\n";
    
    for (int run = 0; run < num_runs; ++run) {
    // Call Linear_Layer_tb
    // Apply initialization
        // init rand_seeds
        for (int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
            rand_seeds_mmap[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
        
        // Zero out the io vectors
        for (int idx = 0; idx < io_mmap.size(); idx++) {
            io_mmap[idx] = 0.0f;
        }

        // // Random init first token's embedding
        // int first_token_idx = rand() % VOCAB_SIZE;
        // for (size_t idx = 0; idx < io_init_vecs; ++idx) {
        //     io_mmap[idx] = vocab_lib[first_token_idx * io_init_vecs + idx];
        // }

        std::vector<int> token_idx;
        std::ifstream fin("my_prompt_token_idx.txt");
        if (!fin) {
            std::cerr << "Failed to open token_idx.txt\n";
            return;
        }
        int id;
        while (fin >> id && token_idx.size() < MAX_PRE_SEQ_LEN)  {
            token_idx.push_back(id);
        }
        std::cout << "Loaded " << token_idx.size() << " tokens:\n";

        // for (size_t idx = 0; idx < io_init_vecs; ++idx) {
        //     io_mmap[idx] = vocab_lib[token_idx[token_idx.size() - 1] * io_init_vecs + idx];
        // }

        // for(int n = 0; n < 16; n ++){
        //     for (size_t idx = 0; idx < 8; ++idx) {
        //         cout << vocab_lib[token_idx[token_idx.size() - 1 - n] * io_init_vecs + idx][0] << " ";
        //     }
        //     cout << endl;
        // }

        for (size_t idx = 0; idx < io_init_vecs; ++idx) {
            io_mmap[idx] = vocab_lib[128000 * io_init_vecs + idx];
        }

        // for (size_t idx = 0; idx < io_init_vecs; ++idx) {
        //     for(int k = 0; k < T_BLOCK_PARALLEL; k++){
        //         io_mmap[idx][k] = (k * io_init_vecs + idx + 1) * 0.001 - 1.5;
        //     }
        // }

        // Random init KV caches
        for (int i = 0; i < MAX_SUM_SEQ_LEN; i++) {
            for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
                for (int h = 0; h < KV_HEAD_NUM; h++) {
                    float k_scale = K_s[layer][h];
                    float v_scale = V_s[layer][h];
                    for (int j = 0; j < HEAD_DIM; j++) {
                        int qval_k = std::round(128*distF(rng) / k_scale);
                        qval_k = std::max(-128, std::min(127, qval_k));
                        int idx = ((layer * KV_HEAD_NUM/DEC_HEAD_PARALLEL + h % (KV_HEAD_NUM/DEC_HEAD_PARALLEL)) * MAX_SUM_SEQ_LEN + i)/DEC_K_PARALLEL *
                                    HEAD_DIM + j;
                        int sub_idx = i % DEC_K_PARALLEL;
                        if(i < MAX_PRE_SEQ_LEN){
                            k_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = ap_int<8>(qval_k);
                            // k_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = ap_int<8>(0);
                        }
                        else {
                            k_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = ap_int<8>(0);
                        }

                        int qval_v = std::round(128*distF(rng) / v_scale);
                        qval_v = std::max(-128, std::min(127, qval_v));
                        idx = ((layer * (KV_HEAD_NUM/DEC_HEAD_PARALLEL) + h % (KV_HEAD_NUM/DEC_HEAD_PARALLEL)) * HEAD_DIM + j)/DEC_V_PARALLEL * MAX_SUM_SEQ_LEN + i;
                        sub_idx = j % DEC_V_PARALLEL;
                        if(i < MAX_PRE_SEQ_LEN){
                            v_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = ap_int<8>(qval_v);
                            // v_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = ap_int<8>(0);
                        }
                        else {
                            v_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = ap_int<8>(0);
                        }
                    }
                }
            }
        }

        // run the kernel
        cout << "kernel begins running!\n";

        
        int64_t kernel_time_ns = tapa::invoke(
            SpinQuant_Decoding, 
            FLAGS_bitstream,
            tapa::read_only_mmap<hls::vector<float, T_BLOCK_PARALLEL>>(vocab_lib),
            tapa::read_write_mmap<hls::vector<float, T_BLOCK_PARALLEL>>(io_mmap),
            // tapa::read_only_mmaps<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL>(w_qkvo_FFN_mmaps),
            // tapa::read_only_mmaps<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL/2>(w_qkvo_FFN_mmaps_half_0),
            // tapa::read_only_mmaps<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL/2>(w_qkvo_FFN_mmaps_half_1),
            tapa::read_only_mmaps<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, T_QKVO_FFN_BLOCK_PARALLEL/2>(w_qkvo_FFN_mmaps_half_0),
            tapa::read_only_mmaps<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, T_QKVO_FFN_BLOCK_PARALLEL/2>(w_qkvo_FFN_mmaps_half_1),
            tapa::read_only_mmap<hls::vector<float, 2>>(w_s_sum_qkvo_FFN_mmap),
            tapa::read_write_mmaps<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL>(k_caches),
            tapa::read_write_mmaps<hls::vector<ap_int<8>, DEC_V_PARALLEL>, DEC_HEAD_PARALLEL>(v_caches),
            tapa::read_only_mmap<hls::vector<float, T_BLOCK_PARALLEL>>(gamma_beta_mmap),
            tapa::read_only_mmap<float>(rand_seeds_mmap),
            tapa::write_only_mmap<int>(sampled_token_idx_mmap),
            // MAX_PRE_SEQ_LEN,
            0,
            MAX_DEC_SEQ_LEN
        );
            
        double t_s = kernel_time_ns * 1e-9;
        std::cout << "  Run " << run << " — kernel time: " << t_s << " s\n";
        total_time_ns += kernel_time_ns;

        for(int i = 0; i < 32; i++) std::cout << io_mmap[i][0] << " ";
        std::cout << std::endl;
        for(int i = 0; i < 32; i++) std::cout << io_mmap[MAX_DEC_SEQ_LEN * (HIDDEN_DIM/T_BLOCK_PARALLEL) + i][0] << " ";
        std::cout << std::endl;

        char fname[256];
        std::snprintf(fname, sizeof(fname), "my_sampled_token_idx_%d.txt", run);
        std::ofstream fout(fname);
        if (!fout) {
            std::cerr << "Failed to open " << fname << " for writing\n";
        } else {
            int n = sampled_token_idx_mmap.size();  // or any actual length you want
            for (int i = 0; i < n; ++i) {
                fout << sampled_token_idx_mmap[i];
                if (i + 1 < n) fout << " ";
            }
            fout << "\n";
        }
    }

    double avg_s = (total_time_ns / double(num_runs)) * 1e-9;
    std::cout << "Average kernel time over " << num_runs << " runs: " << avg_s << " s\n";



}

int main(int argc, char* argv[]) {
    SpinQuant_Decoding_test(argc, argv);
    return 0;
}