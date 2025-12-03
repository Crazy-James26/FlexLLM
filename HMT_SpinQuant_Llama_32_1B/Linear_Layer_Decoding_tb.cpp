#include <iostream>
#include <gflags/gflags.h>

DEFINE_string(bitstream, "", ""/*path to bitstream file, run csim if empty*/);

#include "config.h"
#include "Linear_Layer_test.h"


void Linear_Layer_q_test(int argc, char* argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);


    vector<hls::vector<ap_int<4>, T_BLOCK_PARALLEL>> input_mmap(MAX_DEC_SEQ_LEN*HIDDEN_DIM/T_BLOCK_PARALLEL);
    vector<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>>> weight_mmaps[T_BLOCK_PARALLEL];
    for(int i = 0; i < T_BLOCK_PARALLEL; i++){
        weight_mmaps[i].resize(HIDDEN_DIM/(T_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL)*HIDDEN_DIM);
    }
    vector<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, T_BLOCK_PARALLEL>> output_mmap(MAX_DEC_SEQ_LEN*HIDDEN_DIM/T_BLOCK_PARALLEL);

    ap_int<4> input_gold[MAX_DEC_SEQ_LEN][HIDDEN_DIM];
    ap_int<4> weight_gold[HIDDEN_DIM][HIDDEN_DIM];
    ap_int<log2_HIDDEN_DIM + 8> output_gold[MAX_DEC_SEQ_LEN][HIDDEN_DIM];

    for(int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            ap_int<4> data = static_cast<ap_int<4>>(rand() % 16);  // Random input
            input_mmap[i * HIDDEN_DIM / T_BLOCK_PARALLEL + j % (HIDDEN_DIM / T_BLOCK_PARALLEL)][j / (HIDDEN_DIM / T_BLOCK_PARALLEL)] = data;
            // input_mmap[(i * HIDDEN_DIM + j)/ T_BLOCK_PARALLEL][j % T_BLOCK_PARALLEL] = data;
            input_gold[i][j] = data;
        }
    }

    for(int I = 0; I < T_BLOCK_PARALLEL; I++) {
        for(int i = 0; i < HIDDEN_DIM/T_BLOCK_PARALLEL; i++) {
            for(int j = 0; j < HIDDEN_DIM; j++) {
                ap_int<4> data = static_cast<ap_int<4>>(rand() % 16 - 8);  // Random input
                weight_mmaps[I][i/DEC_QKVO_FFN_W_PARALLEL * HIDDEN_DIM + j][(i/2) % (DEC_QKVO_FFN_W_PARALLEL/2)].range((i%2) * 4 + 3, (i%2) * 4) = data;
                weight_gold[I * HIDDEN_DIM/T_BLOCK_PARALLEL + i][j] = data;
                // weight_mmaps[I][i/DEC_QKVO_FFN_W_PARALLEL * HIDDEN_DIM + j][i % DEC_QKVO_FFN_W_PARALLEL] = data;
                // weight_gold[i * T_BLOCK_PARALLEL + I][j] = data;
            }
        }
    }

    // Call Linear_Layer_tb
    cout << "kernel begins running!\n";
    int64_t kernel_time_ns = tapa::invoke(
        Linear_Layer_q_Decoding_tb, 
        FLAGS_bitstream,
        tapa::read_only_mmap<hls::vector<ap_int<4>, T_BLOCK_PARALLEL>>(input_mmap),
        tapa::read_only_mmaps<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, T_BLOCK_PARALLEL>(weight_mmaps),
        tapa::write_only_mmap<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, T_BLOCK_PARALLEL>>(output_mmap),
        MAX_DEC_SEQ_LEN
    );
    cout << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;


    for(int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            output_gold[i][j] = 0;
            for(int k = 0; k < HIDDEN_DIM; k++){
                output_gold[i][j] += ap_uint<4>(input_gold[i][k]) * weight_gold[j][k];
            }
        }
    }

    bool correct = true;
    for(int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            if(output_mmap[i * HIDDEN_DIM / T_BLOCK_PARALLEL + j % (HIDDEN_DIM / T_BLOCK_PARALLEL)][j / (HIDDEN_DIM / T_BLOCK_PARALLEL)] != output_gold[i][j]){
            // if(output_mmap[(i * HIDDEN_DIM + j)/ T_BLOCK_PARALLEL][j % T_BLOCK_PARALLEL] != output_gold[i][j]){
                correct = false;
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                        << "My: " << output_mmap[i * HIDDEN_DIM / T_BLOCK_PARALLEL + j % (HIDDEN_DIM / T_BLOCK_PARALLEL)][j / (HIDDEN_DIM / T_BLOCK_PARALLEL)]
                        << ", Ref: " << output_gold[i][j]
                        << std::endl;
            }
        }
    }

    if (correct) {
        std::cout << "✅ Linear Layer passed correctness check!" << std::endl;
    } else {
        std::cout << "❌ Linear Layer failed!" << std::endl;
    }
}

#include <random>
#include <limits>

void QuantWrapper_Linear_Layer_q_test(int argc, char* argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);


    vector<hls::vector<float, T_BLOCK_PARALLEL>> input_mmap(MAX_DEC_SEQ_LEN*HIDDEN_DIM/T_BLOCK_PARALLEL);
    vector<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>>> weight_mmaps[T_BLOCK_PARALLEL];
    for(int i = 0; i < T_BLOCK_PARALLEL; i++){
        weight_mmaps[i].resize(HIDDEN_DIM/(T_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL)*HIDDEN_DIM);
    }
    vector<hls::vector<float, 2>> weight_s_sum_mmap(HIDDEN_DIM);
    vector<hls::vector<float, T_BLOCK_PARALLEL>> output_mmap(MAX_DEC_SEQ_LEN*HIDDEN_DIM/T_BLOCK_PARALLEL);

    float input_gold[MAX_DEC_SEQ_LEN][HIDDEN_DIM];
    float weight_gold[HIDDEN_DIM][HIDDEN_DIM];
    float output_fq_gold[MAX_DEC_SEQ_LEN][HIDDEN_DIM];

    ap_uint<4> q_input_val[MAX_DEC_SEQ_LEN][HIDDEN_DIM];
    float input_s_val[MAX_DEC_SEQ_LEN];
    float input_b_val[MAX_DEC_SEQ_LEN];
    ap_int<4> q_weight_val[HIDDEN_DIM][HIDDEN_DIM];
    float weight_s_val[HIDDEN_DIM];
    float weight_sum_val[HIDDEN_DIM];
    ap_int<log2_HIDDEN_DIM + 8> q_output_val[MAX_DEC_SEQ_LEN][HIDDEN_DIM];
    float output_gold[MAX_DEC_SEQ_LEN][HIDDEN_DIM];


    //todo: initialize input and weight
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    // Input: generate and store FP32 input into input_mmap, quantize to fake-quant input_gold
    for (int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        float row_min = std::numeric_limits<float>::max();
        float row_max = std::numeric_limits<float>::lowest();

        // Generate FP32 and determine min/max for quantization
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float val = dist(gen);
            input_gold[i][j] = val;  // Will be overwritten with quantized value
            row_min = std::min(row_min, val);
            row_max = std::max(row_max, val);
        }

        float scale = (row_max - row_min) / 15.0f;  // 4-bit asymmetric
        if (scale == 0) scale = 1.0f;
        input_s_val[i] = scale;
        input_b_val[i] = row_min;

        for (int j = 0; j < HIDDEN_DIM; j++) {
            float val = input_gold[i][j];
            int qval = std::round((val - row_min) / scale);
            qval = std::max(0, std::min(15, qval));
            q_input_val[i][j] = qval;
            input_gold[i][j] = qval * scale + row_min;
            input_mmap[i * HIDDEN_DIM / T_BLOCK_PARALLEL + j % (HIDDEN_DIM / T_BLOCK_PARALLEL)][j / (HIDDEN_DIM / T_BLOCK_PARALLEL)] = val;
            // input_mmap[(i * HIDDEN_DIM + j)/ T_BLOCK_PARALLEL][j % T_BLOCK_PARALLEL] = val;  // Store original fp32
        }
    }

    // Weight: generate, quantize symmetrically per output channel (row)
    for(int I = 0; I < T_BLOCK_PARALLEL; I++) {
        for(int i = 0; i < HIDDEN_DIM/T_BLOCK_PARALLEL; i++) {
            float w_max = 0;
            
            for (int j = 0; j < HIDDEN_DIM; j++) {
                float val = dist(gen);
                weight_gold[I * HIDDEN_DIM/T_BLOCK_PARALLEL + i][j] = val;
                // weight_gold[i * T_BLOCK_PARALLEL + I][j] = val;
                w_max = std::max(w_max, std::abs(val));
            }
            
            float scale = w_max / 7.0f;  // 4-bit symmetric: [-8, 7]
            if (scale == 0) scale = 1.0f;
            weight_s_sum_mmap[I * HIDDEN_DIM/T_BLOCK_PARALLEL + i][0] = scale;
            weight_s_val[I * HIDDEN_DIM/T_BLOCK_PARALLEL + i] = scale;
            // weight_s_val[i * T_BLOCK_PARALLEL + I] = scale;

            weight_s_sum_mmap[I * HIDDEN_DIM/T_BLOCK_PARALLEL + i][1] = 0;
            for (int j = 0; j < HIDDEN_DIM; j++) {
                int qval = std::round(weight_gold[I * HIDDEN_DIM/T_BLOCK_PARALLEL + i][j] / scale);
                // int qval = std::round(weight_gold[i * T_BLOCK_PARALLEL + I][j] / scale);
                qval = std::max(-8, std::min(7, qval));
                weight_gold[I * HIDDEN_DIM/T_BLOCK_PARALLEL + i][j] = qval * scale;
                // weight_gold[i * T_BLOCK_PARALLEL + I][j] = qval * scale;

                int flat_idx = (i / DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM + j;
                int sub_idx = (i/2) % (DEC_QKVO_FFN_W_PARALLEL/2);
                int bit_idx = i % 2;
                weight_mmaps[I][flat_idx][sub_idx].range(bit_idx * 4 + 3, bit_idx * 4) = ap_int<4>(qval);

                q_weight_val[I * HIDDEN_DIM/T_BLOCK_PARALLEL + i][j] = qval;
                weight_s_sum_mmap[I * HIDDEN_DIM/T_BLOCK_PARALLEL + i][1] 
                    += weight_gold[I * HIDDEN_DIM/T_BLOCK_PARALLEL + i][j];
            }
            weight_sum_val[I * HIDDEN_DIM/T_BLOCK_PARALLEL + i] = weight_s_sum_mmap[I * HIDDEN_DIM/T_BLOCK_PARALLEL + i][1];
        }
    }
    


    // Call Linear_Layer_tb
    cout << "kernel begins running!\n";
    int64_t kernel_time_ns = tapa::invoke(
        QuantWrapper_Linear_Layer_q_Decoding_tb, 
        FLAGS_bitstream,
        tapa::read_only_mmap<hls::vector<float, T_BLOCK_PARALLEL>>(input_mmap),
        tapa::read_only_mmaps<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, T_BLOCK_PARALLEL>(weight_mmaps),
        tapa::read_only_mmap<hls::vector<float, 2>>(weight_s_sum_mmap),
        tapa::write_only_mmap<hls::vector<float, T_BLOCK_PARALLEL>>(output_mmap),
        MAX_DEC_SEQ_LEN
    );
    cout << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;

    // fake_quant gold output
    for(int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            output_fq_gold[i][j] = 0;
            for(int k = 0; k < HIDDEN_DIM; k++){
                output_fq_gold[i][j] += input_gold[i][k] * weight_gold[j][k];
            }
        }
    }

    // quant gold output
    for(int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            q_output_val[i][j] = 0;
            for(int k = 0; k < HIDDEN_DIM; k++){
                q_output_val[i][j] += q_input_val[i][k] * q_weight_val[j][k];
            }
            output_gold[i][j] = q_output_val[i][j] * input_s_val[i] * weight_s_val[j]
                                + input_b_val[i] * weight_sum_val[j];
        }
    }


    bool correct = true;
    for(int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            float diff = std::abs(output_mmap[i * HIDDEN_DIM / T_BLOCK_PARALLEL + j % (HIDDEN_DIM / T_BLOCK_PARALLEL)][j / (HIDDEN_DIM / T_BLOCK_PARALLEL)] - output_fq_gold[i][j]);
            // float diff = std::abs(output_mmap[(i * HIDDEN_DIM + j)/ T_BLOCK_PARALLEL][j % T_BLOCK_PARALLEL] - output_fq_gold[i][j]);
            if(diff > 1e-2 * std::abs(output_fq_gold[i][j])) {
                correct = false;
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                        << "My: " << output_mmap[i * HIDDEN_DIM / T_BLOCK_PARALLEL + j % (HIDDEN_DIM / T_BLOCK_PARALLEL)][j / (HIDDEN_DIM / T_BLOCK_PARALLEL)]
                        << ", Ref_fq: " << output_fq_gold[i][j]
                        << ", Ref: " << output_gold[i][j]
                        << std::endl;
            }
        }
    }

    if (correct) {
        std::cout << "✅ QuantWrapper Linear Layer passed correctness check!" << std::endl;
    } else {
        std::cout << "❌ QuantWrapper Linear Layer failed!" << std::endl;
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


void QuantWrapper_Linear_Layer_q_test_real(int argc, char* argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);


    vector<hls::vector<float, T_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_BLOCK_PARALLEL>>> input_mmap(MAX_DEC_SEQ_LEN*HIDDEN_DIM/T_BLOCK_PARALLEL);
    vector<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>>> weight_mmaps[T_BLOCK_PARALLEL];
    for(int i = 0; i < T_BLOCK_PARALLEL; i++){
        weight_mmaps[i].resize(HIDDEN_DIM/(T_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL)*HIDDEN_DIM);
    }
    vector<hls::vector<float, 2>, tapa::aligned_allocator<hls::vector<float, 2>>> weight_s_sum_mmap(HIDDEN_DIM);
    vector<hls::vector<float, T_BLOCK_PARALLEL>, tapa::aligned_allocator<hls::vector<float, T_BLOCK_PARALLEL>>> output_mmap(MAX_DEC_SEQ_LEN*HIDDEN_DIM/T_BLOCK_PARALLEL);


    //todo: initialize input and weight
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    // Input: generate and store FP32 input into input_mmap, quantize to fake-quant input_gold
    for (int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        // Generate FP32 and determine min/max for quantization
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float val = (j + 1) * 0.001 + 1.5;
            input_mmap[i * HIDDEN_DIM / T_BLOCK_PARALLEL + j % (HIDDEN_DIM / T_BLOCK_PARALLEL)][j / (HIDDEN_DIM / T_BLOCK_PARALLEL)] = val;
        }
    }

    for (int i = 0; i < DECODER_LAYER_NUM; i++) {
        decoding_read_int4_bin_as_int8_weight_mmap<T_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, HIDDEN_DIM>(
            "q_proj", i, weight_mmaps, i * HIDDEN_DIM / (T_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM
        );
    }
    cout << "Decoding: Finished reading k_proj weights." << endl;

    #include "parameters/w_q_proj_s_sum.h"
    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        int bias = i * HIDDEN_DIM;
        for(int j = 0; j < HIDDEN_DIM; j++){
            weight_s_sum_mmap[bias + j][0] = w_q_proj_s[i][j];
            weight_s_sum_mmap[bias + j][1] = w_q_proj_sum[i][j];
        }
    }

    // Call Linear_Layer_tb
    cout << "kernel begins running!\n";
    int64_t kernel_time_ns = tapa::invoke(
        QuantWrapper_Linear_Layer_q_Decoding_tb, 
        FLAGS_bitstream,
        tapa::read_only_mmap<hls::vector<float, T_BLOCK_PARALLEL>>(input_mmap),
        tapa::read_only_mmaps<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, T_BLOCK_PARALLEL>(weight_mmaps),
        tapa::read_only_mmap<hls::vector<float, 2>>(weight_s_sum_mmap),
        tapa::write_only_mmap<hls::vector<float, T_BLOCK_PARALLEL>>(output_mmap),
        MAX_DEC_SEQ_LEN
    );
    cout << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;

    

    bool correct = true;
    for(int i = 0; i < MAX_DEC_SEQ_LEN; i++) {
        for(int j = 0; j < 8; j++) {
            std::cout << output_mmap[i * HIDDEN_DIM / T_BLOCK_PARALLEL + j % (HIDDEN_DIM / T_BLOCK_PARALLEL)][j / (HIDDEN_DIM / T_BLOCK_PARALLEL)] << " ";
        }
        cout << " ... ";
        for(int j = HIDDEN_DIM-8; j < HIDDEN_DIM; j++) {
            std::cout << output_mmap[i * HIDDEN_DIM / T_BLOCK_PARALLEL + j % (HIDDEN_DIM / T_BLOCK_PARALLEL)][j / (HIDDEN_DIM / T_BLOCK_PARALLEL)] << " ";
        }
        cout << endl;
    }

}



int main(int argc, char* argv[]) {
    // Linear_Layer_q_test(argc, argv);
    // QuantWrapper_Linear_Layer_q_test(argc, argv);
    QuantWrapper_Linear_Layer_q_test_real(argc, argv);
    return 0;
}