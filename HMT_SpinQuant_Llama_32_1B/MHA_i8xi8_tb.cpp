#include <iostream>
#include <gflags/gflags.h>

DEFINE_string(bitstream, "", ""/*path to bitstream file, run csim if empty*/);

#include "config.h"
#include "MHA_i8xi8.h"

#include <random>
#include <limits>

void MHA_test(int argc, char* argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

    vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> pref_q_mmap(DECODER_LAYER_NUM * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM);
    vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> pref_k_mmap(DECODER_LAYER_NUM * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * KV_HIDDEN_DIM);
    vector<hls::vector<ap_int<8>, PRE_K_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_K_PARALLEL>>> pref_k_cache(DECODER_LAYER_NUM * KV_HEAD_NUM * MAX_PRE_SEQ_LEN/PRE_K_PARALLEL * HEAD_DIM);
    vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> pref_a_mmap(DECODER_LAYER_NUM * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * Q_HEAD_NUM * MAX_PRE_SEQ_LEN);
    
    float pref_q_gold[DECODER_LAYER_NUM][MAX_PRE_SEQ_LEN][HIDDEN_DIM];
    float pref_k_gold[DECODER_LAYER_NUM][MAX_PRE_SEQ_LEN][KV_HIDDEN_DIM];
    float pref_a_gold[DECODER_LAYER_NUM][Q_HEAD_NUM][MAX_PRE_SEQ_LEN][MAX_PRE_SEQ_LEN];

    //todo: initialize input and weight
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dist(-128.0/sqrt(HIDDEN_DIM), 128.0/sqrt(HIDDEN_DIM));
    
    // Input: generate and store FP32 input into pref_q_mmap, quantize to fake-quant input_gold
    for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
        for (int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
            // Generate FP32 and determine min/max for quantization
            for (int h = 0; h < Q_HEAD_NUM; h++) {
                for (int j = 0; j < HEAD_DIM; j++) {
                    float val = dist(gen);
                    pref_q_gold[layer][i][h * HEAD_DIM + j] = val;  // Will be overwritten with quantized value

                    int idx = layer * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM + (i / TOKEN_PARALLEL) * HIDDEN_DIM + h * HEAD_DIM + j;
                    int sub_idx = i % TOKEN_PARALLEL;
                    pref_q_mmap[idx][sub_idx] = val;  // Store original fp32
                }

                float scale = Q_s[layer][h];

                for (int j = 0; j < HEAD_DIM; j++) {
                    int qval = std::round(pref_q_gold[layer][i][h * HEAD_DIM + j]/sqrt_HEAD_DIM / scale);
                    qval = std::max(-128, std::min(127, qval));
                    pref_q_gold[layer][i][h * HEAD_DIM + j] = qval * scale;
                }
            }
        }

        // Weight: generate, quantize symmetrically per output channel (row)
        for (int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
            for (int h = 0; h < KV_HEAD_NUM; h++) {
                for (int j = 0; j < HEAD_DIM; j++) {
                    float val = dist(gen);
                    pref_k_gold[layer][i][h * HEAD_DIM + j] = val;

                    int idx = layer * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * KV_HIDDEN_DIM + (i / TOKEN_PARALLEL) * KV_HIDDEN_DIM + h * HEAD_DIM + j;
                    int sub_idx = i % TOKEN_PARALLEL;
                    pref_k_mmap[idx][sub_idx] = val;  // Store original fp32
                }
                
                float scale = K_s[layer][h];

                for (int j = 0; j < HEAD_DIM; j++) {
                    int qval = std::round(pref_k_gold[layer][i][h * HEAD_DIM + j] / scale);
                    qval = std::max(-128, std::min(127, qval));
                    pref_k_gold[layer][i][h * HEAD_DIM + j] = qval * scale;

                    // int idx = layer * KV_HEAD_NUM * MAX_PRE_SEQ_LEN/PRE_K_PARALLEL * HEAD_DIM + (h * MAX_PRE_SEQ_LEN + i) / PRE_K_PARALLEL * HEAD_DIM + j;
                    // int sub_idx = i % PRE_K_PARALLEL;
                    // pref_k_cache[idx][sub_idx] = ap_int<8>(qval);
                }
            }
        }

        for(int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
            for(int h = 0; h < KV_HEAD_NUM; h++) {
                for(int g = 0; g < ATTN_GROUP_NUM; g++){
                    int q_h = h * ATTN_GROUP_NUM + g;
                    for(int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                        pref_a_gold[layer][q_h][i][j] = 0;
                        for(int k = 0; k < HEAD_DIM; k++){
                            pref_a_gold[layer][q_h][i][j] += pref_q_gold[layer][i][q_h * HEAD_DIM + k] * pref_k_gold[layer][j][h * HEAD_DIM + k];
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
        tapa::read_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(pref_q_mmap),
        tapa::read_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(pref_k_mmap),
        tapa::read_write_mmap<hls::vector<ap_int<8>, PRE_K_PARALLEL>>(pref_k_cache),
        tapa::write_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(pref_a_mmap),
        MAX_PRE_SEQ_LEN
    );
    cout << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;

    bool correct = true;
    for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
        for(int h = 0; h < Q_HEAD_NUM; h++) {
            for(int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
                for(int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                    float actual = pref_a_mmap[layer * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * Q_HEAD_NUM * MAX_PRE_SEQ_LEN + (i/TOKEN_PARALLEL * Q_HEAD_NUM + h) * MAX_PRE_SEQ_LEN + j][i % TOKEN_PARALLEL];
                    float expect = pref_a_gold[layer][h][i][j];
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
    

    // Input: scale and softmax pref_a_gold, and then quantize it with fake-quant
    for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
        for (int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
            for (int h = 0; h < Q_HEAD_NUM; h++) {
                //scale
                for (int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                    if(j > i) 
                        pref_a_gold[layer][h][i][j] = -1e9;
                }

                //softmax
                float attn_exp[MAX_PRE_SEQ_LEN];
                float attn_exp_sum = 0;
                for (int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                    attn_exp[j] = exp(pref_a_gold[layer][h][i][j]);
                    attn_exp_sum += attn_exp[j];
                }

                for (int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                    pref_a_gold[layer][h][i][j] = attn_exp[j] / attn_exp_sum;
                }

                // fake_quantize symmetrically per output channel (row)
                for (int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                    float val = pref_a_gold[layer][h][i][j];
                }

                float scale = A_s[layer][h];

                for (int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                    int qval = std::round(pref_a_gold[layer][h][i][j] / scale);
                    qval = std::min(255, qval);
                    pref_a_gold[layer][h][i][j] = qval * scale;
                }
            }
        }
    }


    // MHA_test
    vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> pref_v_mmap(DECODER_LAYER_NUM * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * KV_HIDDEN_DIM);
    vector<hls::vector<ap_int<8>, PRE_V_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_V_PARALLEL>>> pref_v_cache(DECODER_LAYER_NUM * KV_HIDDEN_DIM/PRE_V_PARALLEL * MAX_PRE_SEQ_LEN);

    vector<hls::vector<float, TOKEN_PARALLEL>, tapa::aligned_allocator<hls::vector<float, TOKEN_PARALLEL>>> pref_o_mmap(DECODER_LAYER_NUM * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM);

    float pref_v_gold[DECODER_LAYER_NUM][KV_HIDDEN_DIM][MAX_PRE_SEQ_LEN];
    float pref_o_gold[DECODER_LAYER_NUM][MAX_PRE_SEQ_LEN][HIDDEN_DIM];

    
    // Weight: generate, quantize symmetrically per output channel (row)
    for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
        for (int h = 0; h < KV_HEAD_NUM; h++) {
            for (int i = 0; i < HEAD_DIM; i++) {
                for (int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                    float val = dist(gen);
                    pref_v_gold[layer][h * HEAD_DIM + i][j] = val;

                    int idx = layer * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * KV_HIDDEN_DIM + (j / TOKEN_PARALLEL) * KV_HIDDEN_DIM + h * HEAD_DIM + i;
                    int sub_idx = j % TOKEN_PARALLEL;
                    pref_v_mmap[idx][sub_idx] = val;  // Store original fp32
                }
                
                float scale = V_s[layer][h];

                for (int j = 0; j < MAX_PRE_SEQ_LEN; j++) {
                    int qval = std::round(pref_v_gold[layer][h * HEAD_DIM + i][j] / scale);
                    qval = std::max(-128, std::min(127, qval));
                    pref_v_gold[layer][h * HEAD_DIM + i][j] = qval * scale;

                    // int idx = layer * KV_HIDDEN_DIM/PRE_V_PARALLEL * MAX_PRE_SEQ_LEN + ((h * HEAD_DIM + i) / PRE_V_PARALLEL) * MAX_PRE_SEQ_LEN +  j;
                    // int sub_idx = i % PRE_V_PARALLEL;
                    // pref_v_cache[idx][sub_idx] = ap_int<8>(qval);
                }
            }
        }

        for(int i = 0; i < MAX_PRE_SEQ_LEN; i++){
            for(int h = 0; h < KV_HEAD_NUM; h++){
                for(int g = 0; g < ATTN_GROUP_NUM; g++){
                    int q_h = h * ATTN_GROUP_NUM + g;
                    for(int j = 0; j < HEAD_DIM; j++) {
                        pref_o_gold[layer][i][q_h * HEAD_DIM + j] = 0;
                        for(int k = 0; k < MAX_PRE_SEQ_LEN; k++){
                            pref_o_gold[layer][i][q_h * HEAD_DIM + j] += pref_a_gold[layer][q_h][i][k] * pref_v_gold[layer][h * HEAD_DIM + j][k];
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
        tapa::read_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(pref_q_mmap),
        tapa::read_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(pref_k_mmap),
        tapa::read_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(pref_v_mmap),
        tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_K_PARALLEL>>(pref_k_cache),
        tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_V_PARALLEL>>(pref_v_cache),
        tapa::write_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(pref_o_mmap),
        MAX_PRE_SEQ_LEN
    );
    cout << "kernel time: " << kernel_time_ns_1 * 1e-9 << " s" << endl;

    for(int i = 0; i < 32; i++) std::cout << pref_q_mmap[i][0] << " ";
    std::cout << std::endl;
    for(int i = 0; i < 32; i++) std::cout << pref_k_cache[i][0] << " ";
    std::cout << std::endl;
    for(int i = 0; i < 32; i++) std::cout << pref_v_cache[i][0] << " ";
    std::cout << std::endl;
    for(int i = 0; i < 32; i++) std::cout << pref_o_mmap[i][0] << " ";
    std::cout << std::endl;

    bool correct_1 = true;
    for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
        for(int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
            for(int j = 0; j < HIDDEN_DIM; j++) {
                float actual = pref_o_mmap[layer * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL * HIDDEN_DIM + (i/TOKEN_PARALLEL) * HIDDEN_DIM + j][i % TOKEN_PARALLEL];
                float expect = pref_o_gold[layer][i][j];
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






    // decoding
    vector<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>> dec_q_mmap(MAX_DEC_SEQ_LEN * DECODER_LAYER_NUM * HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL);
    vector<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>> dec_k_mmap(MAX_SUM_SEQ_LEN * DECODER_LAYER_NUM * KV_HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL);
    vector<hls::vector<ap_int<8>, DEC_K_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_K_PARALLEL>>> dec_k_caches[DEC_HEAD_PARALLEL];
    for(int i = 0; i < DEC_HEAD_PARALLEL; i++){
        dec_k_caches[i].resize(DECODER_LAYER_NUM * KV_HEAD_NUM/DEC_HEAD_PARALLEL * MAX_SUM_SEQ_LEN/DEC_K_PARALLEL * HEAD_DIM);
    }
    vector<hls::vector<float, DEC_HEAD_PARALLEL>, tapa::aligned_allocator<hls::vector<float, DEC_HEAD_PARALLEL>>> dec_a_mmap(MAX_DEC_SEQ_LEN * DECODER_LAYER_NUM * Q_HEAD_NUM/DEC_HEAD_PARALLEL * MAX_SUM_SEQ_LEN);
    
    float dec_q_gold[MAX_DEC_SEQ_LEN][DECODER_LAYER_NUM][HIDDEN_DIM];
    float dec_k_gold[MAX_SUM_SEQ_LEN][DECODER_LAYER_NUM][KV_HIDDEN_DIM];
    float dec_a_gold[MAX_DEC_SEQ_LEN][DECODER_LAYER_NUM][Q_HEAD_NUM][MAX_SUM_SEQ_LEN];
    
    // Input: generate and store FP32 input into dec_q_mmap, quantize to fake-quant input_gold
    
    for (int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
            // Generate FP32 and determine min/max for quantization
            for (int j = 0; j < HIDDEN_DIM; j++) {
                float val = dist(gen);
                dec_q_gold[i][layer][j] = val;  // Will be overwritten with quantized value

                int idx = (layer * MAX_DEC_SEQ_LEN + i) * HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL + j % (HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                int sub_idx = j / (HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                dec_q_mmap[idx][sub_idx] = val;  // Store original fp32
            }

            for (int h = 0; h < Q_HEAD_NUM; h++) {
                float scale = Q_s[layer][h];

                for (int j = 0; j < HEAD_DIM; j++) {
                    int qval = std::round(dec_q_gold[i][layer][h * HEAD_DIM + j]/sqrt_HEAD_DIM / scale);
                    qval = std::max(-128, std::min(127, qval));
                    dec_q_gold[i][layer][h * HEAD_DIM + j] = qval * scale;
                }
            }
        }
    }

    // Weight: generate, quantize symmetrically per output channel (row)
    for (int i = 0; i < MAX_SUM_SEQ_LEN; i++) {
        for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
            for (int j = 0; j < KV_HIDDEN_DIM; j++) {
                float val = i < MAX_PRE_SEQ_LEN ? pref_k_gold[layer][i][j] : dist(gen);
                dec_k_gold[i][layer][j] = val;

                int idx = (layer * MAX_SUM_SEQ_LEN + i) * KV_HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL + j % (KV_HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                int sub_idx = j / (KV_HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                dec_k_mmap[idx][sub_idx] = val;  // Store original fp32
            }
            
            for (int h = 0; h < KV_HEAD_NUM; h++) {
                float scale = K_s[layer][h];

                for (int j = 0; j < HEAD_DIM; j++) {
                    int qval = std::round(dec_k_gold[i][layer][h * HEAD_DIM + j] / scale);
                    qval = std::max(-128, std::min(127, qval));
                    if(i < MAX_PRE_SEQ_LEN){
                        int idx = ((layer * KV_HEAD_NUM/DEC_HEAD_PARALLEL + h % (KV_HEAD_NUM/DEC_HEAD_PARALLEL)) * MAX_SUM_SEQ_LEN + i)/DEC_K_PARALLEL *
                                    HEAD_DIM + j;
                        int sub_idx = i % DEC_K_PARALLEL;
                        // dec_k_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = ap_int<8>(qval);

                        int read_idx = ((layer * KV_HEAD_NUM + h) * MAX_PRE_SEQ_LEN + i)/PRE_K_PARALLEL * HEAD_DIM + j;
                        int read_sub_idx = i % PRE_K_PARALLEL;
                        dec_k_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = pref_k_cache[read_idx][read_sub_idx];

                    }
                    dec_k_gold[i][layer][h * HEAD_DIM + j] = qval * scale;
                }
            }
        }
    }


    for(int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
            for(int h = 0; h < KV_HEAD_NUM; h++) {
                for(int g = 0; g < ATTN_GROUP_NUM; g++){
                    int q_h = h * ATTN_GROUP_NUM + g;
                    for(int j = 0; j < MAX_SUM_SEQ_LEN; j++) {
                        dec_a_gold[i][layer][q_h][j] = 0;
                        if(j <= MAX_PRE_SEQ_LEN + i){
                            for(int k = 0; k < HEAD_DIM; k++){
                                dec_a_gold[i][layer][q_h][j] += dec_q_gold[i][layer][q_h * HEAD_DIM + k] * dec_k_gold[j][layer][h * HEAD_DIM + k];
                            }
                        }
                    }
                }
            }
        }
    }

    cout << "kernel begins running!\n";

    int64_t kernel_time_ns_2 = tapa::invoke(
        MHA_i8xi8_qxk_decoding_tb, 
        FLAGS_bitstream,
        tapa::read_only_mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>(dec_q_mmap),
        tapa::read_only_mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>(dec_k_mmap),
        tapa::read_write_mmaps<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL>(dec_k_caches),
        tapa::write_only_mmap<hls::vector<float, DEC_HEAD_PARALLEL>>(dec_a_mmap),
        MAX_PRE_SEQ_LEN,
        MAX_DEC_SEQ_LEN
    );
    cout << "kernel time: " << kernel_time_ns_2 * 1e-9 << " s" << endl;

    bool correct_2 = true;
    for(int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
            for(int h = 0; h < Q_HEAD_NUM; h++) {
                for(int j = 0; j <= MAX_PRE_SEQ_LEN + i; j++) {
                    int idx = ((i * DECODER_LAYER_NUM + layer) * Q_HEAD_NUM/DEC_HEAD_PARALLEL + h % (Q_HEAD_NUM/DEC_HEAD_PARALLEL)) * MAX_SUM_SEQ_LEN + j;
                    int sub_idx = h / (Q_HEAD_NUM/DEC_HEAD_PARALLEL);
                    float actual = dec_a_mmap[idx][sub_idx];
                    float expect = dec_a_gold[i][layer][h][j];
                    float error = fabs(expect - actual);
                    if(error > 1e-3 * abs(expect)){
                        correct_2 = false;
                        std::cout << "Mismatch at (" << i << ", " << layer << ", " << ", " << h << j << "): "
                                << "My: " << actual
                                << ", Ref: " << expect
                                << ", Diff: " << error
                                << std::endl;
                    }
                }
            }
        }
    }

    if (correct_2) {
        std::cout << "✅ MHA_qxv passed correctness check!" << std::endl;
    } else {
        std::cout << "❌ MHA_qxv Layer failed!" << std::endl;
    }

    // Input: scale and softmax dec_a_gold, and then quantize it with fake-quant
    for (int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
            for (int h = 0; h < Q_HEAD_NUM; h++) {
                //softmax
                float attn_exp[MAX_SUM_SEQ_LEN];
                float attn_exp_sum = 0;
                for (int j = 0; j <= MAX_PRE_SEQ_LEN + i; j++) {
                    attn_exp[j] = exp(dec_a_gold[i][layer][h][j]);
                    attn_exp_sum += attn_exp[j];
                }

                for (int j = 0; j <= MAX_PRE_SEQ_LEN + i; j++) {
                    dec_a_gold[i][layer][h][j] = attn_exp[j] / attn_exp_sum;
                }

                // fake_quantize symmetrically per output channel (row)
                for (int j = 0; j <= MAX_PRE_SEQ_LEN + i; j++) {
                    float val = dec_a_gold[i][layer][h][j];
                }

                float scale = A_s[layer][h];

                for (int j = 0; j <= MAX_PRE_SEQ_LEN + i; j++) {
                    int qval = std::round(dec_a_gold[i][layer][h][j] / scale);
                    qval = std::min(255, qval);
                    dec_a_gold[i][layer][h][j] = qval * scale;
                }
            }
        }
    }


    // MHA_test
    vector<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>> dec_v_mmap(MAX_SUM_SEQ_LEN * DECODER_LAYER_NUM * KV_HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL);
    vector<hls::vector<ap_int<8>, DEC_V_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_V_PARALLEL>>> dec_v_caches[DEC_HEAD_PARALLEL];
    for(int i = 0; i < DEC_HEAD_PARALLEL; i++){
        dec_v_caches[i].resize(DECODER_LAYER_NUM * (KV_HEAD_NUM/DEC_HEAD_PARALLEL * HEAD_DIM/DEC_V_PARALLEL) * MAX_SUM_SEQ_LEN);
    }
    vector<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>> dec_o_mmap(MAX_DEC_SEQ_LEN * DECODER_LAYER_NUM * HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL);

    float dec_v_gold[MAX_SUM_SEQ_LEN][DECODER_LAYER_NUM][KV_HIDDEN_DIM];
    float dec_o_gold[MAX_DEC_SEQ_LEN][DECODER_LAYER_NUM][HIDDEN_DIM];

    
    // Weight: generate, quantize symmetrically per output channel (row)
    for (int i = 0; i < MAX_SUM_SEQ_LEN; i++) {
        for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
            for (int j = 0; j < KV_HIDDEN_DIM; j++) {
                float val = i < MAX_PRE_SEQ_LEN ? pref_v_gold[layer][j][i] : dist(gen);
                dec_v_gold[i][layer][j] = val;

                int idx = (layer * MAX_SUM_SEQ_LEN + i) * KV_HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL + j % (KV_HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                int sub_idx = j / (KV_HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                dec_v_mmap[idx][sub_idx] = val;  // Store original fp32
            }
            
            for (int h = 0; h < KV_HEAD_NUM; h++) {
                float scale = V_s[layer][h];

                for (int j = 0; j < HEAD_DIM; j++) {
                    int qval = std::round(dec_v_gold[i][layer][h * HEAD_DIM + j] / scale);
                    qval = std::max(-128, std::min(127, qval));
                    if(i < MAX_PRE_SEQ_LEN){
                        int idx = ((layer * (KV_HEAD_NUM/DEC_HEAD_PARALLEL) + h % (KV_HEAD_NUM/DEC_HEAD_PARALLEL)) * HEAD_DIM + j)/DEC_V_PARALLEL * MAX_SUM_SEQ_LEN + i;
                        int sub_idx = j % DEC_V_PARALLEL;
                        // dec_v_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = ap_int<8>(qval);

                        int read_idx = ((layer * KV_HIDDEN_DIM+ h * HEAD_DIM + j)/PRE_V_PARALLEL  * MAX_PRE_SEQ_LEN) + i;
                        int read_sub_idx = (h * HEAD_DIM + j) % PRE_V_PARALLEL;
                        dec_v_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = pref_v_cache[read_idx][read_sub_idx];
                    }
                    dec_v_gold[i][layer][h * HEAD_DIM + j] = qval * scale;
                }
            }
        }
    }

    for(int i = 0; i < MAX_DEC_SEQ_LEN; i++){
        for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
            for(int h = 0; h < KV_HEAD_NUM; h++){
                for(int g = 0; g < ATTN_GROUP_NUM; g++){
                    int q_h = h * ATTN_GROUP_NUM + g;
                    for(int j = 0; j < HEAD_DIM; j++) {
                        dec_o_gold[i][layer][q_h * HEAD_DIM + j] = 0;
                        for(int k = 0; k <= MAX_PRE_SEQ_LEN + i; k++){
                            dec_o_gold[i][layer][q_h * HEAD_DIM + j] += dec_a_gold[i][layer][q_h][k] * dec_v_gold[k][layer][h * HEAD_DIM + j];
                        }
                    }
                }
            }
        }
    }   
    
    cout << "kernel begins running!\n";
    int64_t kernel_time_ns_3 = tapa::invoke(
        MHA_i8xi8_decoding_tb, 
        FLAGS_bitstream,
        tapa::read_only_mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>(dec_q_mmap),
        tapa::read_only_mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>(dec_k_mmap),
        tapa::read_only_mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>(dec_v_mmap),
        tapa::read_write_mmaps<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL>(dec_k_caches),
        tapa::read_write_mmaps<hls::vector<ap_int<8>, DEC_V_PARALLEL>, DEC_HEAD_PARALLEL>(dec_v_caches),
        tapa::write_only_mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>(dec_o_mmap),
        MAX_PRE_SEQ_LEN,
        MAX_DEC_SEQ_LEN
    );
    cout << "kernel time: " << kernel_time_ns_3 * 1e-9 << " s" << endl;

    for(int i = 0; i < 32; i++) std::cout << dec_q_mmap[i][0] << " ";
    std::cout << std::endl;
    for(int i = 0; i < 32; i++) std::cout << dec_o_mmap[i][0] << " ";
    std::cout << std::endl;

    bool correct_3 = true;
    for(int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
            for(int j = 0; j < HIDDEN_DIM; j++) {
                int idx = (layer * MAX_DEC_SEQ_LEN + i) * HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL + j % (HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                int sub_idx = j / (HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                float actual = dec_o_mmap[idx][sub_idx];
                float expect = dec_o_gold[i][layer][j];
                float error = fabs(expect - actual);
                if(error > 5e-3 * abs(expect)){
                    correct_3 = false;
                    std::cout << "Mismatch at (" << i << ", " << layer << ", " << j << "): "
                            << "My: " << actual
                            << ", Ref: " << expect
                            << ", Diff: " << error
                            << std::endl;
                }
            }
        }
    }

    if (correct_3) {
        std::cout << "✅ MHA passed correctness check!" << std::endl;
    } else {
        std::cout << "❌ MHA Layer failed!" << std::endl;
    }


}

int main(int argc, char* argv[]) {
    MHA_test(argc, argv);
}