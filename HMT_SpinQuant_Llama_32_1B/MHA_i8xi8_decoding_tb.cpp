#include <iostream>
#include <gflags/gflags.h>

DEFINE_string(bitstream, "", ""/*path to bitstream file, run csim if empty*/);

#include "config.h"
#include "MHA_i8xi8.h"

#include <random>
#include <limits>

void MHA_test(int argc, char* argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

    vector<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>> q_mmap(MAX_DEC_SEQ_LEN * DECODER_LAYER_NUM * HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL);
    vector<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>> k_mmap(MAX_SUM_SEQ_LEN * DECODER_LAYER_NUM * KV_HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL);
    vector<hls::vector<ap_int<8>, DEC_K_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_K_PARALLEL>>> k_caches[DEC_HEAD_PARALLEL];
    for(int i = 0; i < DEC_HEAD_PARALLEL; i++){
        k_caches[i].resize(DECODER_LAYER_NUM * KV_HEAD_NUM/DEC_HEAD_PARALLEL * MAX_SUM_SEQ_LEN/DEC_K_PARALLEL * HEAD_DIM);
    }
    vector<hls::vector<float, DEC_HEAD_PARALLEL>, tapa::aligned_allocator<hls::vector<float, DEC_HEAD_PARALLEL>>> a_mmap(MAX_DEC_SEQ_LEN * DECODER_LAYER_NUM * Q_HEAD_NUM/DEC_HEAD_PARALLEL * MAX_SUM_SEQ_LEN);
    
    float q_gold[MAX_DEC_SEQ_LEN][DECODER_LAYER_NUM][HIDDEN_DIM];
    float k_gold[MAX_SUM_SEQ_LEN][DECODER_LAYER_NUM][KV_HIDDEN_DIM];
    float a_gold[MAX_DEC_SEQ_LEN][DECODER_LAYER_NUM][Q_HEAD_NUM][MAX_SUM_SEQ_LEN];

    //todo: initialize input and weight
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dist(-128.0/sqrt(HIDDEN_DIM), 128.0/sqrt(HIDDEN_DIM));
    
    // Input: generate and store FP32 input into q_mmap, quantize to fake-quant input_gold
    
    for (int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
            // Generate FP32 and determine min/max for quantization
            for (int j = 0; j < HIDDEN_DIM; j++) {
                float val = dist(gen);
                q_gold[i][layer][j] = val;  // Will be overwritten with quantized value

                int idx = (layer * MAX_DEC_SEQ_LEN + i) * HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL + j % (HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                int sub_idx = j / (HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                q_mmap[idx][sub_idx] = val;  // Store original fp32
            }

            for (int h = 0; h < Q_HEAD_NUM; h++) {
                float scale = Q_s[layer][h];

                for (int j = 0; j < HEAD_DIM; j++) {
                    int qval = std::round(q_gold[i][layer][h * HEAD_DIM + j]/sqrt_HEAD_DIM / scale);
                    qval = std::max(-128, std::min(127, qval));
                    q_gold[i][layer][h * HEAD_DIM + j] = qval * scale;
                }
            }
        }
    }

    // Weight: generate, quantize symmetrically per output channel (row)
    for (int i = 0; i < MAX_SUM_SEQ_LEN; i++) {
        for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
            for (int j = 0; j < KV_HIDDEN_DIM; j++) {
                float val = dist(gen);
                k_gold[i][layer][j] = val;

                int idx = (layer * MAX_SUM_SEQ_LEN + i) * KV_HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL + j % (KV_HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                int sub_idx = j / (KV_HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                k_mmap[idx][sub_idx] = val;  // Store original fp32
            }
            
            for (int h = 0; h < KV_HEAD_NUM; h++) {
                float scale = K_s[layer][h];

                for (int j = 0; j < HEAD_DIM; j++) {
                    int qval = std::round(k_gold[i][layer][h * HEAD_DIM + j] / scale);
                    qval = std::max(-128, std::min(127, qval));
                    if(i < MAX_PRE_SEQ_LEN){
                        int idx = ((layer * KV_HEAD_NUM/DEC_HEAD_PARALLEL + h % (KV_HEAD_NUM/DEC_HEAD_PARALLEL)) * MAX_SUM_SEQ_LEN + i)/DEC_K_PARALLEL *
                                    HEAD_DIM + j;
                        int sub_idx = i % DEC_K_PARALLEL;
                        k_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = ap_int<8>(qval);
                    }
                    k_gold[i][layer][h * HEAD_DIM + j] = qval * scale;
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
                        a_gold[i][layer][q_h][j] = 0;
                        if(j <= MAX_PRE_SEQ_LEN + i){
                            for(int k = 0; k < HEAD_DIM; k++){
                                a_gold[i][layer][q_h][j] += q_gold[i][layer][q_h * HEAD_DIM + k] * k_gold[j][layer][h * HEAD_DIM + k];
                            }
                        }
                    }
                }
            }
        }
    }

    cout << "kernel begins running!\n";

    int64_t kernel_time_ns = tapa::invoke(
        MHA_i8xi8_qxk_decoding_tb, 
        FLAGS_bitstream,
        tapa::read_only_mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>(q_mmap),
        tapa::read_only_mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>(k_mmap),
        tapa::read_write_mmaps<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL>(k_caches),
        tapa::write_only_mmap<hls::vector<float, DEC_HEAD_PARALLEL>>(a_mmap),
        MAX_PRE_SEQ_LEN,
        MAX_DEC_SEQ_LEN
    );
    cout << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;

    bool correct = true;
    for(int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
            for(int h = 0; h < Q_HEAD_NUM; h++) {
                for(int j = 0; j <= MAX_PRE_SEQ_LEN + i; j++) {
                    int idx = ((i * DECODER_LAYER_NUM + layer) * Q_HEAD_NUM/DEC_HEAD_PARALLEL + h % (Q_HEAD_NUM/DEC_HEAD_PARALLEL)) * MAX_SUM_SEQ_LEN + j;
                    int sub_idx = h / (Q_HEAD_NUM/DEC_HEAD_PARALLEL);
                    float actual = a_mmap[idx][sub_idx];
                    float expect = a_gold[i][layer][h][j];
                    float error = fabs(expect - actual);
                    if(error > 1e-3 * abs(expect)){
                        correct = false;
                        std::cout << "Mismatch at (" << i << ", " << layer << ", " << h << ", " << j << "): "
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
    for (int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
            for (int h = 0; h < Q_HEAD_NUM; h++) {
                //softmax
                float attn_exp[MAX_SUM_SEQ_LEN];
                float attn_exp_sum = 0;
                for (int j = 0; j <= MAX_PRE_SEQ_LEN + i; j++) {
                    attn_exp[j] = exp(a_gold[i][layer][h][j]);
                    attn_exp_sum += attn_exp[j];
                }

                for (int j = 0; j <= MAX_PRE_SEQ_LEN + i; j++) {
                    a_gold[i][layer][h][j] = attn_exp[j] / attn_exp_sum;
                }

                // fake_quantize symmetrically per output channel (row)
                for (int j = 0; j <= MAX_PRE_SEQ_LEN + i; j++) {
                    float val = a_gold[i][layer][h][j];
                }

                float scale = A_s[layer][h];

                for (int j = 0; j <= MAX_PRE_SEQ_LEN + i; j++) {
                    int qval = std::round(a_gold[i][layer][h][j] / scale);
                    qval = std::min(255, qval);
                    a_gold[i][layer][h][j] = qval * scale;
                }
            }
        }
    }


    // MHA_test
    vector<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>> v_mmap(MAX_SUM_SEQ_LEN * DECODER_LAYER_NUM * KV_HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL);
    vector<hls::vector<ap_int<8>, DEC_V_PARALLEL>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_V_PARALLEL>>> v_caches[DEC_HEAD_PARALLEL];
    for(int i = 0; i < DEC_HEAD_PARALLEL; i++){
        v_caches[i].resize(DECODER_LAYER_NUM * (KV_HEAD_NUM/DEC_HEAD_PARALLEL * HEAD_DIM/DEC_V_PARALLEL) * MAX_SUM_SEQ_LEN);
    }
    vector<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>> o_mmap(MAX_DEC_SEQ_LEN * DECODER_LAYER_NUM * HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL);

    float v_gold[MAX_SUM_SEQ_LEN][DECODER_LAYER_NUM][KV_HIDDEN_DIM];
    float o_gold[MAX_DEC_SEQ_LEN][DECODER_LAYER_NUM][HIDDEN_DIM];

    
    // Weight: generate, quantize symmetrically per output channel (row)
    for (int i = 0; i < MAX_SUM_SEQ_LEN; i++) {
        for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
            for (int j = 0; j < KV_HIDDEN_DIM; j++) {
                float val = dist(gen);
                v_gold[i][layer][j] = val;

                int idx = (layer * MAX_SUM_SEQ_LEN + i) * KV_HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL + j % (KV_HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                int sub_idx = j / (KV_HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                v_mmap[idx][sub_idx] = val;  // Store original fp32
            }
            
            for (int h = 0; h < KV_HEAD_NUM; h++) {
                float scale = V_s[layer][h];

                for (int j = 0; j < HEAD_DIM; j++) {
                    int qval = std::round(v_gold[i][layer][h * HEAD_DIM + j] / scale);
                    qval = std::max(-128, std::min(127, qval));
                    if(i < MAX_PRE_SEQ_LEN){
                        int idx = ((layer * (KV_HEAD_NUM/DEC_HEAD_PARALLEL) + h % (KV_HEAD_NUM/DEC_HEAD_PARALLEL)) * HEAD_DIM + j)/DEC_V_PARALLEL * MAX_SUM_SEQ_LEN + i;
                        int sub_idx = j % DEC_V_PARALLEL;
                        v_caches[h/(KV_HEAD_NUM/DEC_HEAD_PARALLEL)][idx][sub_idx] = ap_int<8>(qval);
                    }
                    v_gold[i][layer][h * HEAD_DIM + j] = qval * scale;
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
                        o_gold[i][layer][q_h * HEAD_DIM + j] = 0;
                        for(int k = 0; k <= MAX_PRE_SEQ_LEN + i; k++){
                            o_gold[i][layer][q_h * HEAD_DIM + j] += a_gold[i][layer][q_h][k] * v_gold[k][layer][h * HEAD_DIM + j];
                        }
                    }
                }
            }
        }
    }   
    
    cout << "kernel begins running!\n";
    int64_t kernel_time_ns_1 = tapa::invoke(
        MHA_i8xi8_decoding_tb, 
        FLAGS_bitstream,
        tapa::read_only_mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>(q_mmap),
        tapa::read_only_mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>(k_mmap),
        tapa::read_only_mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>(v_mmap),
        tapa::read_write_mmaps<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL>(k_caches),
        tapa::read_write_mmaps<hls::vector<ap_int<8>, DEC_V_PARALLEL>, DEC_HEAD_PARALLEL>(v_caches),
        tapa::write_only_mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>(o_mmap),
        MAX_PRE_SEQ_LEN,
        MAX_DEC_SEQ_LEN
    );
    cout << "kernel time: " << kernel_time_ns_1 * 1e-9 << " s" << endl;

    for(int i = 0; i < 32; i++) std::cout << q_mmap[i][0] << " ";
    std::cout << std::endl;
    for(int i = 0; i < 32; i++) std::cout << o_mmap[i][0] << " ";
    std::cout << std::endl;

    bool correct_1 = true;
    for(int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int layer = 0; layer < DECODER_LAYER_NUM; layer++){
            for(int j = 0; j < HIDDEN_DIM; j++) {
                int idx = (layer * MAX_DEC_SEQ_LEN + i) * HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL + j % (HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                int sub_idx = j / (HIDDEN_DIM / T_QKVO_FFN_BLOCK_PARALLEL);
                float actual = o_mmap[idx][sub_idx];
                float expect = o_gold[i][layer][j];
                float error = fabs(expect - actual);
                if(error > 5e-3 * abs(expect)){
                    correct_1 = false;
                    std::cout << "Mismatch at (" << i << ", " << layer << ", " << j << "): "
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
    MHA_test(argc, argv);
}