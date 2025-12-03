#include <iostream>
#include <gflags/gflags.h>

DEFINE_string(bitstream, "", ""/*path to bitstream file, run csim if empty*/);

#include "config.h"
#include "MHA_i8xi8.h"

#include <random>
#include <limits>

void pref_MHA_test(int argc, char* argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

    vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> q_mmap(DECODER_LAYER_NUM * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM);
    vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> k_mmap(DECODER_LAYER_NUM * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * KV_HIDDEN_DIM);
    vector<hls::vector<ap_int<8>, PRE_K_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_K_PARALLEL>>> k_cache(DECODER_LAYER_NUM * KV_HEAD_NUM * MAX_PRE_SEQ_LEN/PRE_K_PARALLEL * HEAD_DIM);
    vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> a_mmap(DECODER_LAYER_NUM * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * Q_HEAD_NUM * MAX_PRE_SEQ_LEN);
    
    float q_gold[DECODER_LAYER_NUM][MAX_PRE_SEQ_LEN][HIDDEN_DIM];
    float k_gold[DECODER_LAYER_NUM][MAX_PRE_SEQ_LEN][KV_HIDDEN_DIM];
    float a_gold[DECODER_LAYER_NUM][Q_HEAD_NUM][MAX_PRE_SEQ_LEN][MAX_PRE_SEQ_LEN];

    //todo: initialize input and weight
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dist(-128.0/sqrt(HIDDEN_DIM), 128.0/sqrt(HIDDEN_DIM));
    
    // Input: generate and store FP32 input into q_mmap, quantize to fake-quant input_gold
    for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
        for (int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
            // Generate FP32 and determine min/max for quantization
            for (int h = 0; h < Q_HEAD_NUM; h++) {
                for (int j = 0; j < HEAD_DIM; j++) {
                    float val = dist(gen);
                    q_gold[layer][i][h * HEAD_DIM + j] = val;  // Will be overwritten with quantized value

                    int idx = layer * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM + (i / TOKEN_PARALLEL) * HIDDEN_DIM + h * HEAD_DIM + j;
                    int sub_idx = i % TOKEN_PARALLEL;
                    q_mmap[idx][sub_idx] = val;  // Store original fp32
                }

                float scale = Q_s[layer][h];

                for (int j = 0; j < HEAD_DIM; j++) {
                    int qval = std::round(q_gold[layer][i][h * HEAD_DIM + j]/sqrt_HEAD_DIM / scale);
                    qval = std::max(-128, std::min(127, qval));
                    q_gold[layer][i][h * HEAD_DIM + j] = qval * scale;
                }
            }
        }

        // Weight: generate, quantize symmetrically per output channel (row)
        for (int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
            for (int h = 0; h < KV_HEAD_NUM; h++) {
                for (int j = 0; j < HEAD_DIM; j++) {
                    float val = dist(gen);
                    k_gold[layer][i][h * HEAD_DIM + j] = val;

                    int idx = layer * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * KV_HIDDEN_DIM + (i / TOKEN_PARALLEL) * KV_HIDDEN_DIM + h * HEAD_DIM + j;
                    int sub_idx = i % TOKEN_PARALLEL;
                    k_mmap[idx][sub_idx] = val;  // Store original fp32
                }
                
                float scale = K_s[layer][h];

                for (int j = 0; j < HEAD_DIM; j++) {
                    int qval = std::round(k_gold[layer][i][h * HEAD_DIM + j] / scale);
                    qval = std::max(-128, std::min(127, qval));
                    k_gold[layer][i][h * HEAD_DIM + j] = qval * scale;

                    // int idx = layer * KV_HEAD_NUM * MAX_PRE_SEQ_LEN/PRE_K_PARALLEL * HEAD_DIM + (h * MAX_PRE_SEQ_LEN + i) / PRE_K_PARALLEL * HEAD_DIM + j;
                    // int sub_idx = i % PRE_K_PARALLEL;
                    // k_cache[idx][sub_idx] = ap_int<8>(qval);
                }
            }
        }

        for(int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
            for(int h = 0; h < KV_HEAD_NUM; h++) {
                for(int g = 0; g < ATTN_GROUP_NUM; g++){
                    int q_h = h * ATTN_GROUP_NUM + g;
                    for(int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                        a_gold[layer][q_h][i][j] = 0;
                        for(int k = 0; k < HEAD_DIM; k++){
                            a_gold[layer][q_h][i][j] += q_gold[layer][i][q_h * HEAD_DIM + k] * k_gold[layer][j][h * HEAD_DIM + k];
                        }
                    }
                }
            }   
        }
    }

    cout << "kernel begins running!\n";

    int64_t kernel_time_ns = tapa::invoke(
        MHA_i8xi8_qxk_Prefilling_tb, 
        FLAGS_bitstream,
        tapa::read_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(q_mmap),
        tapa::read_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(k_mmap),
        tapa::read_write_mmap<hls::vector<ap_int<8>, PRE_K_PARALLEL>>(k_cache),
        tapa::write_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(a_mmap),
        MAX_PRE_SEQ_LEN
    );
    cout << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;

    bool correct = true;
    for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
        for(int h = 0; h < Q_HEAD_NUM; h++) {
            for(int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
                for(int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                    float actual = a_mmap[layer * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * Q_HEAD_NUM * MAX_PRE_SEQ_LEN + (i/TOKEN_PARALLEL * Q_HEAD_NUM + h) * MAX_PRE_SEQ_LEN + j][i % TOKEN_PARALLEL];
                    float expect = a_gold[layer][h][i][j];
                    float error = fabs(expect - actual);
                    if(error > 1e-3 * abs(expect)){
                        correct = false;
                        std::cout << "Mismatch at (" << i << ", " << j << "): "
                                << "My: " << actual
                                << ", Ref: " << expect
                                << ", Diff: " << error
                                << std::endl;
                    }
                }
            }
        }
    }

    if (correct) {
        std::cout << "✅ MHA_qxv passed correctness check!" << std::endl;
    } else {
        std::cout << "❌ MHA_qxv Layer failed!" << std::endl;
    }
    

    // Input: scale and softmax a_gold, and then quantize it with fake-quant
    for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
        for (int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
            for (int h = 0; h < Q_HEAD_NUM; h++) {
                //scale
                for (int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                    if(j > i) 
                        a_gold[layer][h][i][j] = -1e38;
                }

                //softmax
                float attn_exp[MAX_PRE_SEQ_LEN];
                float attn_exp_sum = 0;
                for (int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                    attn_exp[j] = exp(a_gold[layer][h][i][j]);
                    attn_exp_sum += attn_exp[j];
                }

                for (int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                    a_gold[layer][h][i][j] = attn_exp[j] / attn_exp_sum;
                }

                float scale = A_s[layer][h];

                for (int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                    int qval = std::round(a_gold[layer][h][i][j] / scale);
                    qval = std::min(255, qval);
                    a_gold[layer][h][i][j] = qval * scale;
                }
            }
        }
    }


    // MHA_test
    vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> v_mmap(DECODER_LAYER_NUM * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * KV_HIDDEN_DIM);
    vector<hls::vector<ap_int<8>, PRE_V_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_V_PARALLEL>>> v_cache(DECODER_LAYER_NUM * KV_HIDDEN_DIM/PRE_V_PARALLEL * MAX_PRE_SEQ_LEN);

    vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> o_mmap(DECODER_LAYER_NUM * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM);

    float v_gold[DECODER_LAYER_NUM][KV_HIDDEN_DIM][MAX_PRE_SEQ_LEN];
    float o_gold[DECODER_LAYER_NUM][MAX_PRE_SEQ_LEN][HIDDEN_DIM];

    
    // Weight: generate, quantize symmetrically per output channel (row)
    for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
        for (int h = 0; h < KV_HEAD_NUM; h++) {
            for (int i = 0; i < HEAD_DIM; i++) {
                for (int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                    float val = dist(gen);
                    v_gold[layer][h * HEAD_DIM + i][j] = val;

                    int idx = layer * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * KV_HIDDEN_DIM + (j / TOKEN_PARALLEL) * KV_HIDDEN_DIM + h * HEAD_DIM + i;
                    int sub_idx = j % TOKEN_PARALLEL;
                    v_mmap[idx][sub_idx] = val;  // Store original fp32
                }
                
                float scale = V_s[layer][h];

                for (int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                    int qval = std::round(v_gold[layer][h * HEAD_DIM + i][j] / scale);
                    qval = std::max(-128, std::min(127, qval));
                    v_gold[layer][h * HEAD_DIM + i][j] = qval * scale;

                    // int idx = layer * KV_HIDDEN_DIM/PRE_V_PARALLEL * MAX_PRE_SEQ_LEN + ((h * HEAD_DIM + i) / PRE_V_PARALLEL) * MAX_PRE_SEQ_LEN +  j;
                    // int sub_idx = i % PRE_V_PARALLEL;
                    // v_cache[idx][sub_idx] = ap_int<8>(qval);
                }
            }
        }

        for(int i = 0; i < MAX_PRE_SEQ_LEN; i++){
            for(int h = 0; h < KV_HEAD_NUM; h++){
                for(int g = 0; g < ATTN_GROUP_NUM; g++){
                    int q_h = h * ATTN_GROUP_NUM + g;
                    for(int j = 0; j < HEAD_DIM; j++) {
                        o_gold[layer][i][q_h * HEAD_DIM + j] = 0;
                        for(int k = 0; k < MAX_PRE_SEQ_LEN; k++){
                            o_gold[layer][i][q_h * HEAD_DIM + j] += a_gold[layer][q_h][i][k] * v_gold[layer][h * HEAD_DIM + j][k];
                        }
                    }
                }
            }
        }
    }
    
    cout << "kernel begins running!\n";
    int64_t kernel_time_ns_1 = tapa::invoke(
        MHA_i8xi8_Prefilling_tb, 
        FLAGS_bitstream,
        tapa::read_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(q_mmap),
        tapa::read_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(k_mmap),
        tapa::read_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(v_mmap),
        tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_K_PARALLEL>>(k_cache),
        tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_V_PARALLEL>>(v_cache),
        tapa::write_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(o_mmap),
        MAX_PRE_SEQ_LEN
    );
    cout << "kernel time: " << kernel_time_ns_1 * 1e-9 << " s" << endl;

    for(int i = 0; i < 32; i++) std::cout << q_mmap[i][0] << " ";
    std::cout << std::endl;
    for(int i = 0; i < 32; i++) std::cout << k_cache[i][0] << " ";
    std::cout << std::endl;
    for(int i = 0; i < 32; i++) std::cout << v_cache[i][0] << " ";
    std::cout << std::endl;
    for(int i = 0; i < 32; i++) std::cout << o_mmap[i][0] << " ";
    std::cout << std::endl;

    bool correct_1 = true;
    for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
        for(int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
            for(int j = 0; j < HIDDEN_DIM; j++) {
                float actual = o_mmap[layer * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM + (i/TOKEN_PARALLEL) * HIDDEN_DIM + j][i % TOKEN_PARALLEL];
                float expect = o_gold[layer][i][j];
                float error = fabs(expect - actual);
                if(error > 5e-3 * abs(expect)){
                    correct_1 = false;
                    std::cout << "Mismatch at (" << i << ", " << j << "): "
                            << "My: " << actual
                            << ", Ref: " << expect
                            << ", Diff: " << error
                            << std::endl;
                }
            }
        }
    }

    if (correct_1) {
        std::cout << "✅ MHA passed correctness check!" << std::endl;
    } else {
        std::cout << "❌ MHA Layer failed!" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    pref_MHA_test(argc, argv);
}