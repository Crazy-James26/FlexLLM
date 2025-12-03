#include <iostream>
#include <random>
#include <limits>
#include <gflags/gflags.h>

DEFINE_string(bitstream, "", ""/*path to bitstream file, run csim if empty*/);

#include "config.h"
#include "Linear_Layer_test.h"


void Linear_Layer_q_test(int argc, char* argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);


    vector<hls::vector<ap_int<4>, TOKEN_PARALLEL>> input_mmap(MAX_PRE_SEQ_LEN/TOKEN_PARALLEL*HIDDEN_DIM);
    vector<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>> weight_mmap(HIDDEN_DIM/PRE_QKVO_W_PARALLEL*HIDDEN_DIM);
    vector<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> output_mmap(MAX_PRE_SEQ_LEN/TOKEN_PARALLEL*HIDDEN_DIM);

    float input_fp32[MAX_PRE_SEQ_LEN][HIDDEN_DIM];
    ap_int<4> input_gold[MAX_PRE_SEQ_LEN][HIDDEN_DIM];
    float weight_fp32[HIDDEN_DIM][HIDDEN_DIM];
    ap_int<4> weight_gold[HIDDEN_DIM][HIDDEN_DIM];
    ap_int<log2_HIDDEN_DIM + 8> output_gold[MAX_PRE_SEQ_LEN][HIDDEN_DIM];

    // for(int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
    //     for(int j = 0; j < HIDDEN_DIM; j++) {
    //         ap_int<4> data = static_cast<ap_int<4>>(rand() % 16);  // Random input
    //         input_mmap[(i/TOKEN_PARALLEL) * HIDDEN_DIM + j][i % TOKEN_PARALLEL] = data;
    //         input_gold[i][j] = data;
    //     }
    // }

    // for(int i = 0; i < HIDDEN_DIM/2; i++) {
    //     for(int j = 0; j < HIDDEN_DIM; j++) {
    //         ap_int<4> data_0 = static_cast<ap_int<4>>(rand() % 16 - 8);  // Random input
    //         ap_int<4> data_1 = static_cast<ap_int<4>>(rand() % 16 - 8);  // Random input
    //         ap_int<8> data = (data_1, data_0);
    //         weight_mmap[(i / (PRE_QKVO_W_PARALLEL/2)) * HIDDEN_DIM + j][i % (PRE_QKVO_W_PARALLEL/2)] = data;
    //         weight_gold[2 * i][j] = data_0;
    //         weight_gold[2 * i + 1][j] = data_1;
    //     }
    // }

    //todo: initialize input and weight
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    // Input: generate and store FP32 input into input_mmap, quantize to fake-quant input_gold
    for (int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
        float row_min = std::numeric_limits<float>::max();
        float row_max = std::numeric_limits<float>::lowest();

        // Generate FP32 and determine min/max for quantization
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float val = dist(gen);
            input_fp32[i][j] = val;  // Will be overwritten with quantized value
            row_min = std::min(row_min, val);
            row_max = std::max(row_max, val);
        }

        float scale = (row_max - row_min) / 15.0f;  // 4-bit asymmetric
        if (scale == 0) scale = 1.0f;

        for (int j = 0; j < HIDDEN_DIM; j++) {
            float val = input_fp32[i][j];
            int qval = std::round((val - row_min) / scale);
            qval = std::max(0, std::min(15, qval));
            input_gold[i][j] = ap_uint<4>(qval);
            int idx = (i / TOKEN_PARALLEL) * HIDDEN_DIM + j;
            int sub_idx = i % TOKEN_PARALLEL;
            input_mmap[idx][sub_idx] = ap_uint<4>(qval);  // Store original fp32
        }
    }

    // Weight: generate, quantize symmetrically per output channel (row)
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float w_max = 0;
        
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float val = dist(gen);
            weight_fp32[i][j] = val;
            w_max = std::max(w_max, std::abs(val));
        }
        
        float scale = w_max / 7.0f;  // 4-bit symmetric: [-8, 7]
        if (scale == 0) scale = 1.0f;

        for (int j = 0; j < HIDDEN_DIM; j++) {
            int qval = std::round(weight_fp32[i][j] / scale);
            qval = std::max(-8, std::min(7, qval));
            weight_gold[i][j] = qval;

            int flat_idx = ((i/2) / (PRE_QKVO_W_PARALLEL/2)) * HIDDEN_DIM + j;
            int sub_idx = (i/2) % (PRE_QKVO_W_PARALLEL/2);
            int bit_idx = i % 2;
            weight_mmap[flat_idx][sub_idx].range(bit_idx * 4 + 3, bit_idx * 4) = ap_int<4>(qval);
        }
    }



    // Call Linear_Layer_tb
    cout << "kernel begins running!\n";
    int64_t kernel_time_ns = tapa::invoke(
        Linear_Layer_q_Prefilling_tb, 
        FLAGS_bitstream,
        tapa::read_only_mmap<hls::vector<ap_int<4>, TOKEN_PARALLEL>>(input_mmap),
        tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>(weight_mmap),
        tapa::write_only_mmap<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>(output_mmap),
        MAX_PRE_SEQ_LEN
    );
    cout << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;


    for(int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            output_gold[i][j] = 0;
            for(int k = 0; k < HIDDEN_DIM; k++){
                output_gold[i][j] += ap_uint<4>(input_gold[i][k]) * weight_gold[j][k];
            }
        }
    }

    bool correct = true;
    for(int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            if(output_mmap[(i/TOKEN_PARALLEL) * HIDDEN_DIM + j][i % TOKEN_PARALLEL] != output_gold[i][j]){
                correct = false;
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                        << "My: " << output_mmap[(i/TOKEN_PARALLEL) * HIDDEN_DIM + j][i % TOKEN_PARALLEL]
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


void QuantWrapper_Linear_Layer_q_test(int argc, char* argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);


    vector<hls::vector<float, TOKEN_PARALLEL>> input_mmap(MAX_PRE_SEQ_LEN/TOKEN_PARALLEL*HIDDEN_DIM);
    vector<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>> weight_mmap(HIDDEN_DIM/PRE_QKVO_W_PARALLEL*HIDDEN_DIM);
    vector<hls::vector<float, 2>> weight_s_sum_mmap(HIDDEN_DIM);
    vector<hls::vector<float, TOKEN_PARALLEL>> output_mmap(MAX_PRE_SEQ_LEN/TOKEN_PARALLEL*HIDDEN_DIM);

    float input_gold[MAX_PRE_SEQ_LEN][HIDDEN_DIM];
    float weight_gold[HIDDEN_DIM][HIDDEN_DIM];
    float output_fq_gold[MAX_PRE_SEQ_LEN][HIDDEN_DIM];

    ap_uint<4> q_input_val[MAX_PRE_SEQ_LEN][HIDDEN_DIM];
    float input_s_val[MAX_PRE_SEQ_LEN];
    float input_b_val[MAX_PRE_SEQ_LEN];
    ap_int<4> q_weight_val[HIDDEN_DIM][HIDDEN_DIM];
    float weight_s_val[HIDDEN_DIM];
    float weight_sum_val[HIDDEN_DIM];
    ap_int<log2_HIDDEN_DIM + 8> q_output_val[MAX_PRE_SEQ_LEN][HIDDEN_DIM];
    float output_gold[MAX_PRE_SEQ_LEN][HIDDEN_DIM];


    //todo: initialize input and weight
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    // Input: generate and store FP32 input into input_mmap, quantize to fake-quant input_gold
    for (int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
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
        // printf("golden %d: Input scale: %f, zero point: %f\n", i, scale, row_min);

        for (int j = 0; j < HIDDEN_DIM; j++) {
            float val = input_gold[i][j];
            int qval = std::round((val - row_min) / scale);
            qval = std::max(0, std::min(15, qval));
            q_input_val[i][j] = qval;
            input_gold[i][j] = qval * scale + row_min;

            int idx = (i / TOKEN_PARALLEL) * HIDDEN_DIM + j;
            int sub_idx = i % TOKEN_PARALLEL;
            input_mmap[idx][sub_idx] = val;  // Store original fp32
        }
    }

    // Weight: generate, quantize symmetrically per output channel (row)
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float w_max = 0;
        
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float val = dist(gen);
            weight_gold[i][j] = val;
            w_max = std::max(w_max, std::abs(val));
        }
        
        float scale = w_max / 7.0f;  // 4-bit symmetric: [-8, 7]
        if (scale == 0) scale = 1.0f;
        weight_s_sum_mmap[i][0] = scale;
        weight_s_val[i] = scale;

        weight_s_sum_mmap[i][1] = 0;
        for (int j = 0; j < HIDDEN_DIM; j++) {
            int qval = std::round(weight_gold[i][j] / scale);
            qval = std::max(-8, std::min(7, qval));
            weight_gold[i][j] = qval * scale;

            int flat_idx = ((i/2) / (PRE_QKVO_W_PARALLEL/2)) * HIDDEN_DIM + j;
            int sub_idx = (i/2) % (PRE_QKVO_W_PARALLEL/2);
            int bit_idx = i % 2;
            weight_mmap[flat_idx][sub_idx].range(bit_idx * 4 + 3, bit_idx * 4) = ap_int<4>(qval);
            q_weight_val[i][j] = qval;

            weight_s_sum_mmap[i][1] += weight_gold[i][j];
        }
        weight_sum_val[i] = weight_s_sum_mmap[i][1];
    }
    
    // fake_quant gold output
    for(int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            output_fq_gold[i][j] = 0;
            for(int k = 0; k < HIDDEN_DIM; k++){
                output_fq_gold[i][j] += input_gold[i][k] * weight_gold[j][k];
            }
        }
    }

    // quant gold output
    for(int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            q_output_val[i][j] = 0;
            for(int k = 0; k < HIDDEN_DIM; k++){
                q_output_val[i][j] += q_input_val[i][k] * q_weight_val[j][k];
            }
            output_gold[i][j] = q_output_val[i][j] * input_s_val[i] * weight_s_val[j]
                                + input_b_val[i] * weight_sum_val[j];
        }
    }



    // Call Linear_Layer_tb
    cout << "kernel begins running!\n";
    int64_t kernel_time_ns = tapa::invoke(
        QuantWrapper_Linear_Layer_q_Prefilling_tb, 
        FLAGS_bitstream,
        tapa::read_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(input_mmap),
        tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>(weight_mmap),
        tapa::read_only_mmap<hls::vector<float, 2>>(weight_s_sum_mmap),
        tapa::write_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(output_mmap),
        MAX_PRE_SEQ_LEN
    );
    cout << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;


    bool correct = true;
    for(int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            float diff = std::abs(output_mmap[(i/TOKEN_PARALLEL) * HIDDEN_DIM + j][i % TOKEN_PARALLEL] - output_fq_gold[i][j]);
            if(diff > 1e-2 * std::abs(output_fq_gold[i][j])) {
                correct = false;
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                        << "My: " << output_mmap[(i/TOKEN_PARALLEL) * HIDDEN_DIM + j][i % TOKEN_PARALLEL]
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


void QuantWrapper_Linear_Layer_q_test_real(int argc, char* argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);


    vector<hls::vector<float, TOKEN_PARALLEL>> input_mmap(MAX_PRE_SEQ_LEN/TOKEN_PARALLEL*HIDDEN_DIM);
    vector<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>, tapa::aligned_allocator<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>> weight_mmap(HIDDEN_DIM/PRE_QKVO_W_PARALLEL*HIDDEN_DIM);
    vector<hls::vector<float, 2>> weight_s_sum_mmap(HIDDEN_DIM);
    vector<hls::vector<float, TOKEN_PARALLEL>> output_mmap(MAX_PRE_SEQ_LEN/TOKEN_PARALLEL*HIDDEN_DIM);


    //todo: initialize input and weight
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    // Input: generate and store FP32 input into input_mmap, quantize to fake-quant input_gold
    for (int i = 0; i < MAX_PRE_SEQ_LEN; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float val = (j + 1) * 0.001 - 1.5;
            int idx = (i / TOKEN_PARALLEL) * HIDDEN_DIM + j;
            int sub_idx = i % TOKEN_PARALLEL;
            input_mmap[idx][sub_idx] = val;  // Store original fp32
        }
    }

    for(int i = 0; i < DECODER_LAYER_NUM; i++) {
        prefilling_read_int4_bin_as_int8_weight_mmap<PRE_QKVO_W_PARALLEL_READ, PRE_QKVO_W_PARALLEL, HIDDEN_DIM, HIDDEN_DIM>(
            "q_proj", i, weight_mmap, i * ((HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL) * HIDDEN_DIM
        );
    }
    cout << "Prefilling: Finished reading q_proj weights." << endl;

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
        QuantWrapper_Linear_Layer_q_Prefilling_tb, 
        FLAGS_bitstream,
        tapa::read_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(input_mmap),
        tapa::read_only_mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>>(weight_mmap),
        tapa::read_only_mmap<hls::vector<float, 2>>(weight_s_sum_mmap),
        tapa::write_only_mmap<hls::vector<float, TOKEN_PARALLEL>>(output_mmap),
        MAX_PRE_SEQ_LEN
    );
    cout << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;

    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 8; j++) {
            cout << output_mmap[(i/TOKEN_PARALLEL) * HIDDEN_DIM + j][i % TOKEN_PARALLEL] << " ";
        }
        cout << " ... ";
        for(int j = 0; j < 8; j++) {
            cout << output_mmap[(i/TOKEN_PARALLEL) * HIDDEN_DIM + HIDDEN_DIM - 8 + j][i % TOKEN_PARALLEL] << " ";
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


