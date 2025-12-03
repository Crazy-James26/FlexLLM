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

#include "SpinQuant_Prefilling.h"


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

void SpinQuant_Prefilling_test(int argc, char* argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

    cout << "Token Parallel: " << TOKEN_PARALLEL << endl;
    cout << "Pre QKVO Weight Parallel: " << PRE_QKVO_W_PARALLEL << endl;
    cout << "Pre QKVO Weight Parallel Read: " << PRE_QKVO_W_PARALLEL_READ << endl;
    cout << "Pre K Parallel: " << PRE_K_PARALLEL << endl;
    cout << "Pre V Parallel: " << PRE_V_PARALLEL << endl;
    cout << "Pre FFN Weight Parallel: " << PRE_FFN_W_PARALLEL << endl;
    cout << "Pre FFN Weight Parallel Read: " << PRE_FFN_W_PARALLEL_READ << endl;

    // Input/Output mmap
    vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> io_mmap((DECODER_LAYER_NUM + 1) * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM);

    // Linear Layer QKVO weight mmap
    vector<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>> wk_wq_mmap(
        DECODER_LAYER_NUM * ((KV_HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL + (HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL)*HIDDEN_DIM
    );
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> wk_wq_s_sum_mmap(DECODER_LAYER_NUM *  (KV_HIDDEN_DIM + HIDDEN_DIM));

    vector<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>> wv_wo_mmap(
        DECODER_LAYER_NUM * ((KV_HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL + (HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL)*HIDDEN_DIM
    );
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> wv_wo_s_sum_mmap(DECODER_LAYER_NUM * (KV_HIDDEN_DIM + HIDDEN_DIM));
    
    // MHA
    vector<hls::vector<ap_int<8>, PRE_K_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_K_PARALLEL>>> k_cache(DECODER_LAYER_NUM * KV_HEAD_NUM * MAX_PRE_SEQ_LEN/PRE_K_PARALLEL * HEAD_DIM);
    vector<hls::vector<ap_int<8>, PRE_V_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_V_PARALLEL>>> v_cache(DECODER_LAYER_NUM * KV_HIDDEN_DIM/PRE_V_PARALLEL * MAX_PRE_SEQ_LEN);

    // FFN
    vector<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>>> w_ffn_gate_mmap(
        DECODER_LAYER_NUM * ((INTER_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * HIDDEN_DIM
    );
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> w_ffn_gate_s_sum_mmap(DECODER_LAYER_NUM * INTER_DIM);
    vector<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>>> w_ffn_up_mmap(
        DECODER_LAYER_NUM * ((INTER_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * HIDDEN_DIM
    );
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> w_ffn_up_s_sum_mmap(DECODER_LAYER_NUM * INTER_DIM);
    vector<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>>> w_ffn_down_mmap(
        DECODER_LAYER_NUM * ((HIDDEN_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * INTER_DIM
    );
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> w_ffn_down_s_sum_mmap(DECODER_LAYER_NUM * HIDDEN_DIM);

    // Layer Norm weight mmap
    vector<float, tapa::aligned_allocator<float>> gamma_beta_mmap_0(DECODER_LAYER_NUM * HIDDEN_DIM);
    vector<float, tapa::aligned_allocator<float>> gamma_beta_mmap_1(DECODER_LAYER_NUM * HIDDEN_DIM);

    // // Residual cache
    // vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> res0_cache_mmap(MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM);
    // vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> res1_cache_mmap(MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM);
    
    // 2) Initialize buffers with random data
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

    // initInt8Vec(wk_wq_mmap, PRE_QKVO_W_PARALLEL_READ/2);
    // initFloatVec(wk_wq_s_sum_mmap, 2);
    // initInt8Vec(wv_wo_mmap, PRE_QKVO_W_PARALLEL_READ/2);
    // initFloatVec(wv_wo_s_sum_mmap, 2);

    // initInt8Vec(w_ffn_gate_mmap, PRE_FFN_W_PARALLEL_READ/2);
    // initFloatVec(w_ffn_gate_s_sum_mmap, 2);
    // initInt8Vec(w_ffn_up_mmap, PRE_FFN_W_PARALLEL_READ/2);
    // initFloatVec(w_ffn_up_s_sum_mmap, 2);
    // initInt8Vec(w_ffn_down_mmap, PRE_FFN_W_PARALLEL_READ/2);
    // initFloatVec(w_ffn_down_s_sum_mmap, 2);

    // for (auto &f : gamma_beta_mmap_0) f = distF(rng);
    // for (auto &f : gamma_beta_mmap_1) f = distF(rng);

    
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
        prefilling_read_int4_bin_as_int8_weight_mmap<PRE_FFN_W_PARALLEL_READ, PRE_FFN_W_PARALLEL, HIDDEN_DIM, INTER_DIM>(
            "gate_proj", i, w_ffn_gate_mmap, i * ((INTER_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * HIDDEN_DIM
        );
    }
    cout << "Prefilling: Finished reading gate_proj weights." << endl;

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        prefilling_read_int4_bin_as_int8_weight_mmap<PRE_FFN_W_PARALLEL_READ, PRE_FFN_W_PARALLEL, HIDDEN_DIM, INTER_DIM>(
            "up_proj", i, w_ffn_up_mmap, i * ((INTER_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * HIDDEN_DIM
        );
    }
    cout << "Prefilling: Finished reading up_proj weights." << endl;

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        prefilling_read_int4_bin_as_int8_weight_mmap<PRE_FFN_W_PARALLEL_READ, PRE_FFN_W_PARALLEL, INTER_DIM, HIDDEN_DIM>(
            "down_proj", i, w_ffn_down_mmap, i * ((HIDDEN_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * INTER_DIM
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
       
    

    const int num_runs = 1;
    int64_t total_time_ns = 0;
    std::cout << "kernel begins running " << num_runs << " times …\n";
    
    for (int run = 0; run < num_runs; ++run) {
    // Call Linear_Layer_tb
    // Apply initialization
        
        size_t io_init_vecs = (MAX_PRE_SEQ_LEN / TOKEN_PARALLEL) * HIDDEN_DIM;

        // // Random init first chunk
        // for (size_t idx = 0; idx < io_init_vecs; ++idx) {
        //     for (int j = 0; j < TOKEN_PARALLEL; ++j) {
        //         io_mmap[idx][j] = distF(rng);
        //     }
        // }
        
        // Zero out the remaining vectors
        for (size_t idx = io_init_vecs; idx < io_mmap.size(); ++idx) {
            io_mmap[idx] = 0.0f;
        }
        
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
        int test_seq_len = token_idx.size();

        prefilling_read_embedding_from_bin<TOKEN_PARALLEL, HIDDEN_DIM>("model_embed_tokens_fp32", token_idx, io_mmap, 0);
        cout << "Prefilling: Finished reading input embedding." << endl;

        // for(int idx = 0; idx < MAX_PRE_SEQ_LEN/TOKEN_PARALLEL; idx++) {
        //     for(int k = 0; k < HIDDEN_DIM; k++){
        //         io_mmap[idx * HIDDEN_DIM + k] = (k + 1) * 0.001f - 1.5;
        //     }
        // }

        for (auto &vec : k_cache)      for (int i = 0; i < PRE_K_PARALLEL; ++i) vec[i] = 0;
        for (auto &vec : v_cache)      for (int i = 0; i < PRE_V_PARALLEL; ++i) vec[i] = 0;

        // for (auto &vec : res0_cache_mmap) for (int i = 0; i < TOKEN_PARALLEL; ++i) vec[i] = 0.0f;
        // for (auto &vec : res1_cache_mmap) for (int i = 0; i < TOKEN_PARALLEL; ++i) vec[i] = 0.0f;

        cout << "kernel begins running!\n";
        int64_t kernel_time_ns = tapa::invoke(
            SpinQuant_Prefilling, 
            FLAGS_bitstream,
            tapa::read_write_mmap<hls::vector<float, TOKEN_PARALLEL>>(io_mmap),
            tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>(wk_wq_mmap),
            tapa::read_only_mmap<hls::vector<float, 2>>(wk_wq_s_sum_mmap),
            tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>(wv_wo_mmap),
            tapa::read_only_mmap<hls::vector<float, 2>>(wv_wo_s_sum_mmap),
            tapa::read_write_mmap<hls::vector<ap_int<8>, PRE_K_PARALLEL>>(k_cache),
            tapa::read_write_mmap<hls::vector<ap_int<8>, PRE_V_PARALLEL>>(v_cache),
            tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>>(w_ffn_gate_mmap),
            tapa::read_only_mmap<hls::vector<float, 2>>(w_ffn_gate_s_sum_mmap),
            tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>>(w_ffn_up_mmap),
            tapa::read_only_mmap<hls::vector<float, 2>>(w_ffn_up_s_sum_mmap),
            tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>>(w_ffn_down_mmap),
            tapa::read_only_mmap<hls::vector<float, 2>>(w_ffn_down_s_sum_mmap),
            tapa::read_only_mmap<float>(gamma_beta_mmap_0),
            tapa::read_only_mmap<float>(gamma_beta_mmap_1),
            // tapa::read_write_mmap<hls::vector<float, TOKEN_PARALLEL>>(res0_cache_mmap),
            // tapa::read_write_mmap<hls::vector<float, TOKEN_PARALLEL>>(res1_cache_mmap),
            MAX_PRE_SEQ_LEN
        );

        double t_s = kernel_time_ns * 1e-9;
        std::cout << "  Run " << run << " — kernel time: " << t_s << " s\n";
        total_time_ns += kernel_time_ns;

        for(int i = 0; i < 32; i++) std::cout << io_mmap[i][0] << " ";
        std::cout << std::endl;
        for(int i = 0; i < 32; i++) std::cout << io_mmap[io_init_vecs + i][0] << " ";
        std::cout << std::endl;
    }

    double avg_s = (total_time_ns / double(num_runs)) * 1e-9;
    std::cout << "Average kernel time over " << num_runs << " runs: " << avg_s << " s\n";
}

int main(int argc, char* argv[]) {
    SpinQuant_Prefilling_test(argc, argv);
    return 0;
}