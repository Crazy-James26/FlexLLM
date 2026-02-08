#ifndef __QUANT_H__
#define __QUANT_H__
#include "config.h"
#include "data_io.h"

template <int qint_bit, bool is_act_asym, int io_parallel, int max_input_dim, int max_seq_len=MAX_PRE_SEQ_LEN>
void pref_quant_layer_fp32_qint(
    tapa::istream<hls::vector<float, io_parallel>>& input_stream,
    tapa::ostream<hls::vector<float, 1+is_act_asym>>& input_s_b_stream, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<qint_bit>, io_parallel>>& output_stream, //asym_quant: ap_uint<qint_bit>, sym_quant: ap_int<qint_bit>
    int seq_len = max_seq_len,
    int input_dim = max_input_dim
){
    float input_buffer[io_parallel][max_input_dim];
    #pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=1
    #pragma HLS bind_storage variable=input_buffer type=ram_2p impl=uram
    float input_max[io_parallel];
    #pragma HLS ARRAY_PARTITION variable=input_max complete dim=1
    float input_min[io_parallel];
    #pragma HLS ARRAY_PARTITION variable=input_min complete dim=1

    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        init_max_min: for(int m = 0; m < io_parallel; m++){
        #pragma HLS unroll
            input_max[m] = -1e32;
            input_min[m] = 1e32;
        }

        load_input_loop: for(int k = 0; k < input_dim; k++){
        #pragma HLS PIPELINE II=1
            hls::vector<float, io_parallel> temp_pack = input_stream.read();
            for(int m = 0; m < io_parallel; m++){
                // temp_pack[m] = (k + 1) * 0.001 + 1.5; // for debug

                input_buffer[m][k] = temp_pack[m];
                if(temp_pack[m] > input_max[m]){
                    input_max[m] = temp_pack[m];
                }
                if(temp_pack[m] < input_min[m]){
                    input_min[m] = temp_pack[m];
                }
            }
            
            if(M==0 && k == 0) cout << "input data: ";
            if(M==0 && k < 8) cout << temp_pack[0] << " ";
            // if(M==0 && k == 7) cout << " ... ";
            // if(M==0 && k > input_dim - 8) cout << temp_pack[1] << " ";   
            if(M==0 && k == 7) cout << endl;  
                    
        }

        if(is_act_asym){
            float scale_base = (1 << qint_bit) - 1;
            float s[io_parallel];
            float b[io_parallel];
            
            asym_cal_s_b_loop: for(int m = 0; m < io_parallel; m++){
            #pragma HLS PIPELINE II=1
                // input_min[m] = input_min[m] > 0 ? 0 : input_min[m]; // when input_min > 0, set it to 0
                s[m] = (input_max[m] - input_min[m]) / scale_base;
                if(s[m] == 0) s[m] = 1;
                b[m] = input_min[m];
                hls::vector<float, 1+is_act_asym> s_b_pack;
                s_b_pack[0] = s[m];
                s_b_pack[1] = b[m];
                input_s_b_stream.write(s_b_pack);
            }
            
            
            hls::vector<ap_int<qint_bit>, io_parallel> out_pack;
            asym_quant_loop: for(int k = 0; k < input_dim; k++){
            #pragma HLS PIPELINE II=1
                for(int m = 0; m < io_parallel; m++){
                    float out_temp = (input_buffer[m][k] - b[m]) / s[m];
                    out_pack[m] = (ap_uint<qint_bit>) round(out_temp);
                    // ap_fixed<qint_bit, qint_bit, AP_RND, AP_SAT> out_temp_fixed = out_temp;
                    // out_pack[m] = out_temp_fixed;
                }
                output_stream.write(out_pack);
            }
        }

        else{
            float scale_base = (1 << (qint_bit - 1)) - 1;
            float s[io_parallel];
            sym_cal_s_loop: for(int m = 0; m < io_parallel; m++){
            #pragma HLS PIPELINE II=1
                float abs_max = abs(input_max[m]);
                float abs_min = abs(input_min[m]);
                s[m] = (abs_max > abs_min ? abs_max : abs_min) / scale_base;
                if(s[m] == 0) s[m] = 1;
                hls::vector<float, 1+is_act_asym> s_pack;
                s_pack[0] = s[m];
                input_s_b_stream.write(s_pack);
            }
            
            hls::vector<ap_int<qint_bit>, io_parallel> out_pack;
            sym_quant_loop: for(int k = 0; k < input_dim; k++){
            #pragma HLS PIPELINE II=1
                for(int m = 0; m < io_parallel; m++){
                    float out_temp = input_buffer[m][k] / s[m];
                    out_pack[m] = (ap_int<qint_bit>) round(out_temp);
                    // ap_fixed<qint_bit, qint_bit, AP_RND, AP_SAT> out_temp_fixed = out_temp;
                    // out_pack[m] = out_temp_fixed;
                }
                output_stream.write(out_pack);
            }
        }
        
    }
}


template <int qint_bit, bool is_act_asym, int io_parallel, int max_output_dim, int max_seq_len=MAX_PRE_SEQ_LEN>
void pref_dequant_layer_qint_fp32(
    tapa::istream<hls::vector<ap_int<qint_bit>, io_parallel>>& input_stream,
    tapa::istream<hls::vector<float, 1+is_act_asym>>& input_s_b_stream, //input's scale factor and zero point
    tapa::istream<hls::vector<float, 1+is_act_asym>>& weight_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, io_parallel>>& output_stream,
    int seq_len = max_seq_len,
    int output_dim = max_output_dim
){
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        float s[io_parallel];
        float b[io_parallel];

        for(int m = 0; m < io_parallel; m++){
        #pragma HLS PIPELINE II=1
            hls::vector<float, 1+is_act_asym> s_b_pack = input_s_b_stream.read();
            s[m] = s_b_pack[0];
            if(is_act_asym){
                b[m] = s_b_pack[1];
            }
            // printf("dequant %d: Input scale: %f, zero point: %f\n", m, s[m], b[m]);
        }
        
        output_dim_loop: for(int k = 0; k < output_dim; k++){
        #pragma HLS PIPELINE II=1
            hls::vector<ap_int<qint_bit>, io_parallel> temp_pack = input_stream.read();
            hls::vector<float, 1+is_act_asym> weight_s_sum_pack = weight_s_sum_stream.read();
            // printf("dequant %d: Weight scale: %f, row sum: %f\n", k, weight_s_sum_pack[0], weight_s_sum_pack[1]);
            hls::vector<float, io_parallel> output_pack;
            dequant_loop: for(int m = 0; m < io_parallel; m++){
                if(is_act_asym){
                    float dequant_temp = temp_pack[m] * s[m] * weight_s_sum_pack[0];
                    output_pack[m] = dequant_temp + b[m] * weight_s_sum_pack[1];
                }
                else{
                    float dequant_temp = temp_pack[m] * s[m] * weight_s_sum_pack[0];
                    output_pack[m] = dequant_temp;
                }
            }
            output_stream.write(output_pack);

            if(M==0 && k == 0) cout << "output data: ";
            if(M==0 && k < 8) cout << output_pack[0] << " ";
            // if(M==0 && k == 7) cout << " ... ";
            // if(M==0 && k > output_dim - 8) cout << output_pack[1] << " ";   
            if(M==0 && k == 7) cout << endl;          
        }
    }
}

template <int qint_bit, int io_parallel, int io_s_parallel, int max_input_dim, int head_num=1, int is_signed=true, int max_seq_len=MAX_PRE_SEQ_LEN>
void pref_static_sym_quant_layer_fp32_qint(
    tapa::istream<hls::vector<float, io_parallel>>& input_stream,
    tapa::istream<hls::vector<float, io_s_parallel>>& input_s_stream, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<qint_bit>, io_parallel>>& output_stream, //asym_quant: ap_uint<qint_bit>, sym_quant: ap_int<qint_bit>
    int seq_len = max_seq_len,
    int input_dim = max_input_dim
){
    const int scale_base = is_signed ? (1 << (qint_bit - 1)) : (1 << qint_bit);

    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        head_num_loop: for (int H = 0; H < head_num; H++){
            float s[io_parallel];
            for(int M = 0; M < io_parallel/io_s_parallel; M++){
            #pragma HLS PIPELINE II=1
                hls::vector<float, io_s_parallel> s_pack = input_s_stream.read();
                for(int m = 0; m < io_s_parallel; m++){
                #pragma HLS unroll
                    s[M * io_s_parallel + m] = s_pack[m];
                }
            }

            
            sym_quant_loop: for(int k = 0; k < input_dim; k++){
            #pragma HLS PIPELINE II=1
                hls::vector<float, io_parallel> in_pack = input_stream.read();
                hls::vector<ap_int<qint_bit>, io_parallel> out_pack;
                for(int m = 0; m < io_parallel; m++){
                    float scale_temp = in_pack[m] / s[m];
                    int round_temp = round(scale_temp);
                    int out_temp;
                    if(is_signed)
                        out_temp =  round_temp > scale_base - 1 ? scale_base - 1: 
                                    round_temp < -scale_base ? -scale_base:
                                    round_temp;
                    else
                        out_temp =  round_temp > scale_base - 1 ? scale_base - 1: round_temp;
                    out_pack[m] = (ap_int<qint_bit>) out_temp;
                }
                output_stream.write(out_pack);
            }
        }
    }
}


template <int qint_bit, int io_parallel, int io_s_parallel, int max_output_dim, int act_head_num=1, int weight_head_num=1, int max_seq_len=MAX_PRE_SEQ_LEN>
void pref_static_sym_dequant_layer_qint_fp32(
    tapa::istream<hls::vector<ap_int<qint_bit>, io_parallel>>& input_stream,
    tapa::istream<hls::vector<float, io_s_parallel>>& input_s_stream, //input's scale factor and zero point
    tapa::istream<float>& weight_s_stream, //weight's scale factor
    tapa::ostream<hls::vector<float, io_parallel>>& output_stream,
    int seq_len = max_seq_len,
    int output_dim = max_output_dim
){
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        head_loop: for(int h = 0; h < weight_head_num; h++){
            group_loop: for(int g = 0; g < act_head_num/weight_head_num; g++){
                float s[io_parallel];
                #pragma HLS ARRAY_PARTITION variable=s type=cyclic factor=io_s_parallel dim=1
                for(int M = 0; M < io_parallel/io_s_parallel; M++){
                #pragma HLS PIPELINE II=1
                    hls::vector<float, io_s_parallel>  s_pack = input_s_stream.read();
                    for(int m = 0; m < io_s_parallel; m++){
                        s[M * io_s_parallel + m] = s_pack[m];
                    }
                }
                
                output_dim_loop: for(int k = 0; k < output_dim; k++){
                #pragma HLS PIPELINE II=1
                    hls::vector<ap_int<qint_bit>, io_parallel> temp_pack = input_stream.read();
                    float weight_s = weight_s_stream.read();
                    hls::vector<float, io_parallel> output_pack;
                    dequant_loop: for(int m = 0; m < io_parallel; m++){
                        output_pack[m] = temp_pack[m] * s[m] * weight_s;
                    }
                    output_stream.write(output_pack);                
                }
            }
        }
    }
}


template <int qint_bit, int io_parallel, int max_input_dim, int head_num=1, int is_signed=true, int max_seq_len=MAX_PRE_SEQ_LEN, int decoder_layer_num=DECODER_LAYER_NUM>
void pref_static_sym_per_tensor_quant_layer_fp32_qint(
    tapa::istream<hls::vector<float, io_parallel>>& input_stream,
    tapa::ostream<hls::vector<ap_int<qint_bit>, io_parallel>>& output_stream, //asym_quant: ap_uint<qint_bit>, sym_quant: ap_int<qint_bit>
    const float input_s[decoder_layer_num][head_num],
    int block_id,
    int seq_len = max_seq_len,
    int input_dim = max_input_dim,
    float mha_scale_factor = 1.0
){
    const int scale_base = is_signed ? (1 << (qint_bit - 1)) : (1 << qint_bit);

    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        head_num_loop: for (int H = 0; H < head_num; H++){
            sym_quant_loop: for(int k = 0; k < input_dim; k++){
            #pragma HLS PIPELINE II=1
                hls::vector<float, io_parallel> in_pack = input_stream.read();
                hls::vector<ap_int<qint_bit>, io_parallel> out_pack;
                for(int m = 0; m < io_parallel; m++){
                    float scale_temp = in_pack[m] / (mha_scale_factor * input_s[block_id][H]);
                    int round_temp = round(scale_temp);
                    int out_temp;
                    if(is_signed)
                        out_temp =  round_temp > scale_base - 1 ? scale_base - 1: 
                                    round_temp < -scale_base ? -scale_base:
                                    round_temp;
                    else
                        out_temp =  round_temp > scale_base - 1 ? scale_base - 1: round_temp;
                    out_pack[m] = (ap_int<qint_bit>) out_temp;
                }
                output_stream.write(out_pack);

                if(M==0 && H == 0 && k == 0) cout << "mha input/output data: ";
                if(M==0 && H == 0 && k < 8) cout << "(" << in_pack[0] << ", " << out_pack[0] << ") ";
                if(M==0 && H == 0 && k == 7) cout << endl;


            }
        }
    }
}


template <int qint_bit, int io_parallel, int max_output_dim, int act_head_num=1, int weight_head_num=1, int max_seq_len=MAX_PRE_SEQ_LEN, int decoder_layer_num=DECODER_LAYER_NUM>
void pref_static_sym_per_tensor_dequant_layer_qint_fp32(
    tapa::istream<hls::vector<ap_int<qint_bit>, io_parallel>>& input_stream,
    tapa::ostream<hls::vector<float, io_parallel>>& output_stream,
    const float input_s[decoder_layer_num][act_head_num],
    const float weight_s[decoder_layer_num][weight_head_num],
    int block_id,
    int seq_len = max_seq_len,
    int output_dim = max_output_dim
){
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        head_loop: for(int h = 0; h < weight_head_num; h++){
            group_loop: for(int g = 0; g < act_head_num/weight_head_num; g++){
                output_dim_loop: for(int k = 0; k < output_dim; k++){
                #pragma HLS PIPELINE II=1
                    hls::vector<ap_int<qint_bit>, io_parallel> temp_pack = input_stream.read();
                    hls::vector<float, io_parallel> output_pack;
                    dequant_loop: for(int m = 0; m < io_parallel; m++){
                        output_pack[m] = temp_pack[m] * input_s[block_id][h*act_head_num/weight_head_num+g] * weight_s[block_id][h];
                    }
                    output_stream.write(output_pack);
                    
                    if(M==0 && h == 0 && g==0 && k == 0) cout << "mha input/output data: ";
                    if(M==0 && h == 0 && g==0 && k < 8) cout << "(" << temp_pack[0] << ", " << output_pack[0] << ") ";
                    if(M==0 && h == 0 && g==0 && k == 7) cout << endl;
                }
            }
        }
    }
}





template <int qint_bit, bool is_act_asym, int block_parallel, int max_input_dim>
void dec_quant_layer_fp32_qint(
    tapa::istream<hls::vector<float, block_parallel>>& input_stream,
    tapa::ostream<hls::vector<float, 1+is_act_asym>>& input_s_b_stream, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<qint_bit>, block_parallel>>& output_stream, //asym_quant: ap_uint<qint_bit>, sym_quant: ap_int<qint_bit>
    int input_dim = max_input_dim
){
    float input_buffer[block_parallel][max_input_dim/block_parallel];
    #pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=1
    #pragma HLS bind_storage variable=input_buffer type=ram_2p impl=uram
    float input_max[block_parallel];
    #pragma HLS ARRAY_PARTITION variable=input_max complete dim=1
    float input_min[block_parallel];
    #pragma HLS ARRAY_PARTITION variable=input_min complete dim=1
    float final_max, final_min;

    
    init_max_min: for(int m = 0; m < block_parallel; m++){
    #pragma HLS unroll
        input_max[m] = -1e32;
        input_min[m] = 1e32;
    }

    load_input_loop: for(int k = 0; k < input_dim/block_parallel; k++){
    #pragma HLS PIPELINE II=1
        hls::vector<float, block_parallel> temp_pack = input_stream.read();
        for(int m = 0; m < block_parallel; m++){
            // temp_pack[m] = ((m * input_dim/block_parallel + k) + 1) * 0.001 + 1.5; // for debug

            input_buffer[m][k] = temp_pack[m];
            if(temp_pack[m] > input_max[m]){
                input_max[m] = temp_pack[m];
            }
            if(temp_pack[m] < input_min[m]){
                input_min[m] = temp_pack[m];
            }
        } 
        if(k == 0) cout << "input data: ";
        if(k < 8) cout << temp_pack[0] << " ";
        if(k == 7) cout << endl;
    }

    // Calculate final max and min
    final_max = input_max[0];
    final_min = input_min[0];
    final_max_min_loop: for(int m = 1; m < block_parallel; m++){
        if(input_max[m] > final_max) final_max = input_max[m];
        if(input_min[m] < final_min) final_min = input_min[m];
    }

    if(is_act_asym){
        float scale_base = (1 << qint_bit) - 1;
        float s;
        float b;

        // final_min = final_min > 0 ? 0: final_min; // when input_min > 0, set it to 0
        s = (final_max - final_min) / scale_base;
        if(s == 0) s = 1;
        b = final_min;
        hls::vector<float, 1+is_act_asym> s_b_pack;
        s_b_pack[0] = s;
        s_b_pack[1] = b;
        input_s_b_stream.write(s_b_pack);
        
        hls::vector<ap_int<qint_bit>, block_parallel> out_pack;
        asym_quant_loop: for(int k = 0; k < input_dim/block_parallel; k++){
        #pragma HLS PIPELINE II=1
            for(int m = 0; m < block_parallel; m++){
                float out_temp = (input_buffer[m][k] - b) / s;
                out_pack[m] = (ap_uint<qint_bit>) round(out_temp);
            }
            output_stream.write(out_pack);
        }
    }

    else{
        float scale_base = (1 << (qint_bit - 1)) - 1;
        float s;
        
        float abs_max = abs(final_max);
        float abs_min = abs(final_min);
        s = (abs_max > abs_min ? abs_max : abs_min) / scale_base;
        if(s == 0) s = 1;
        hls::vector<float, 1+is_act_asym> s_pack;
        s_pack[0] = s;
        input_s_b_stream.write(s_pack);
        
        hls::vector<ap_int<qint_bit>, block_parallel> out_pack;
        sym_quant_loop: for(int k = 0; k < input_dim/block_parallel; k++){
        #pragma HLS PIPELINE II=1
            for(int m = 0; m < block_parallel; m++){
                float out_temp = input_buffer[m][k] / s;
                out_pack[m] = (ap_int<qint_bit>) round(out_temp);
                // ap_fixed<qint_bit, qint_bit, AP_RND, AP_SAT> out_temp_fixed = out_temp;
                // out_pack[m] = out_temp_fixed;
            }
            output_stream.write(out_pack);
        }
    }
}


template <int qint_bit, bool is_act_asym, int block_parallel, int max_output_dim>
void dec_dequant_layer_qint_fp32(
    tapa::istream<hls::vector<ap_int<qint_bit>, block_parallel>>& input_stream,
    tapa::istream<hls::vector<float, 1+is_act_asym>>& input_s_b_stream, //input's scale factor and zero point
    tapa::istream<hls::vector<float, block_parallel * (1+is_act_asym)>>& weight_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, block_parallel>>& output_stream,
    int output_dim = max_output_dim
){
    float s;
    float b;
    hls::vector<float, 1+is_act_asym> s_b_pack = input_s_b_stream.read();
    s = s_b_pack[0];
    if(is_act_asym){
        b = s_b_pack[1];
    }

    output_dim_loop: for(int k = 0; k < output_dim/block_parallel; k++){
    #pragma HLS PIPELINE II=1
        hls::vector<ap_int<qint_bit>, block_parallel> temp_pack = input_stream.read();
        hls::vector<float, block_parallel * (1+is_act_asym)> weight_s_sum_pack = weight_s_sum_stream.read();
        hls::vector<float, block_parallel> output_pack;
        dequant_loop: for(int m = 0; m < block_parallel; m++){
            if(is_act_asym){
                float dequant_temp = temp_pack[m] * s * weight_s_sum_pack[2 * m];
                output_pack[m] = dequant_temp + b * weight_s_sum_pack[2 * m + 1];
            }
            else{
                float dequant_temp = temp_pack[m] * s * weight_s_sum_pack[m];
                output_pack[m] = dequant_temp;
            }
        }
        output_stream.write(output_pack);                
    }
}


template <int qint_bit, bool is_act_asym, int block_parallel, int weight_parallel, int max_output_dim>
void dec_dequant_layer_qint_fp32_bandwidth(
    tapa::istream<hls::vector<ap_int<qint_bit>, block_parallel>>& input_stream,
    tapa::istream<hls::vector<float, 1+is_act_asym>>& input_s_b_stream, //input's scale factor and zero point
    tapa::istream<hls::vector<float, 1+is_act_asym>>& weight_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, block_parallel>>& output_stream,
    int output_dim = max_output_dim
){
    float s;
    float b;
    hls::vector<float, 1+is_act_asym> s_b_pack = input_s_b_stream.read();
    s = s_b_pack[0];
    if(is_act_asym){
        b = s_b_pack[1];
    }

    float weight_s[block_parallel * weight_parallel];
    #pragma HLS ARRAY_PARTITION variable=weight_s type=block factor=block_parallel dim=1
    float weight_sum[block_parallel * weight_parallel];
    #pragma HLS ARRAY_PARTITION variable=weight_sum type=block factor=block_parallel dim=1
    output_dim_loop: for(int k = 0; k < output_dim/(block_parallel * weight_parallel); k++){
        weight_s_sum_load_loop: for(int m = 0; m < block_parallel * weight_parallel; m++){
        #pragma HLS PIPELINE II=1
            hls::vector<float, 1+is_act_asym> weight_s_sum_pack = weight_s_sum_stream.read();
            if(is_act_asym){
                weight_s[m] = weight_s_sum_pack[0];
                weight_sum[m] = weight_s_sum_pack[1];
            }
            else{
                weight_s[m] = weight_s_sum_pack[0];
            }
        }

        dequant_weight_loop: for(int n = 0; n < weight_parallel; n++){
            hls::vector<ap_int<qint_bit>, block_parallel> temp_pack = input_stream.read();
            hls::vector<float, block_parallel> output_pack;
            dequant_block_loop: for(int m = 0; m < block_parallel; m++){
            #pragma HLS unroll
                if(is_act_asym){
                    float dequant_temp = temp_pack[m] * s * weight_s[m * weight_parallel + n];
                    output_pack[m] = dequant_temp + b * weight_sum[m * weight_parallel + n];
                }
                else{
                    float dequant_temp = temp_pack[m] * s * weight_s[m * weight_parallel + n];
                    output_pack[m] = dequant_temp;
                }
            }
            output_stream.write(output_pack);

            if(k==0 && n == 0) cout << "output data: ";
            if(k==0 && n < 8) cout << output_pack[0] << " ";
            if(k==0 && n == 7) cout << endl;
        }                
    }
}


template <int qint_bit, bool is_act_asym, int head_parallel, int max_input_dim, int head_num=1>
void dec_MHA_quant_layer_fp32_qint(
    tapa::istream<hls::vector<float, head_parallel>>& input_stream,
    tapa::ostream<hls::vector<float, head_parallel * (1+is_act_asym)>>& input_s_b_stream, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<qint_bit>, head_parallel>>& output_stream, //asym_quant: ap_uint<qint_bit>, sym_quant: ap_int<qint_bit>
    int input_dim = max_input_dim
){
    float input_buffer[head_parallel][max_input_dim];
    #pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=1
    #pragma HLS bind_storage variable=input_buffer type=ram_2p impl=uram
    float input_max[head_parallel];
    #pragma HLS ARRAY_PARTITION variable=input_max complete dim=1
    float input_min[head_parallel];
    #pragma HLS ARRAY_PARTITION variable=input_min complete dim=1
    float final_max, final_min;

    head_loop: for(int H = 0; H < head_num/head_parallel; H++){
        init_max_min: for(int m = 0; m < head_parallel; m++){
        #pragma HLS unroll
            input_max[m] = -1e32;
            input_min[m] = 1e32;
        }

        load_input_loop: for(int k = 0; k < input_dim; k++){
        #pragma HLS PIPELINE II=1
            hls::vector<float, head_parallel> temp_pack = input_stream.read();
            for(int m = 0; m < head_parallel; m++){
                input_buffer[m][k] = temp_pack[m];
                if(temp_pack[m] > input_max[m]){
                    input_max[m] = temp_pack[m];
                }
                if(temp_pack[m] < input_min[m]){
                    input_min[m] = temp_pack[m];
                }
            }                
        }

        if(is_act_asym){
            float scale_base = (1 << qint_bit) - 1;
            float s[head_parallel];
            float b[head_parallel];
            hls::vector<float, head_parallel * (1+is_act_asym)> s_b_pack;
            
            for(int m = 0; m < head_parallel; m++){
            #pragma HLS unroll
                s[m] = (input_max[m] - input_min[m]) / scale_base;
                if(s[m] == 0) s[m] = 1;
                b[m] = input_min[m];
                s_b_pack[2 * m] = s[m];
                s_b_pack[2 * m + 1] = b[m];
            }
            input_s_b_stream.write(s_b_pack);
            
            hls::vector<ap_int<qint_bit>, head_parallel> out_pack;
            asym_quant_loop: for(int k = 0; k < input_dim/head_parallel; k++){
            #pragma HLS PIPELINE II=1
                for(int m = 0; m < head_parallel; m++){
                    float out_temp = (input_buffer[m][k] - b[m]) / s[m];
                    out_pack[m] = (ap_uint<qint_bit>) round(out_temp);
                }
                output_stream.write(out_pack);
            }
        }

        else{
            float scale_base = (1 << (qint_bit - 1)) - 1;
            float s[head_parallel];
            hls::vector<float, head_parallel * (1+is_act_asym)> s_pack;
            
            for(int m = 0; m < head_parallel; m++){
            #pragma HLS PIPELINE II=1
                float abs_max = abs(input_max[m]);
                float abs_min = abs(input_min[m]);
                s[m] = (abs_max > abs_min ? abs_max : abs_min) / scale_base;
                if(s[m] == 0) s[m] = 1;
                s_pack[m] = s[m];
                input_s_b_stream.write(s_pack);
            }
            
            hls::vector<ap_int<qint_bit>, head_parallel> out_pack;
            sym_quant_loop: for(int k = 0; k < input_dim/head_parallel; k++){
            #pragma HLS PIPELINE II=1
                for(int m = 0; m < head_parallel; m++){
                    float out_temp = input_buffer[m][k] / s[m];
                    out_pack[m] = (ap_int<qint_bit>) round(out_temp);
                }
                output_stream.write(out_pack);
            }
        }
    }
}



template <int qint_bit, bool is_act_asym, int head_parallel, int max_output_dim, int head_num=1>
void dec_MHA_dequant_layer_qint_fp32(
    tapa::istream<hls::vector<ap_int<qint_bit>, head_parallel>>& input_stream,
    tapa::istream<hls::vector<float, head_parallel * (1+is_act_asym)>>& input_s_b_stream, //input's scale factor and zero point
    tapa::istream<hls::vector<float, head_parallel>>& weight_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, head_parallel>>& output_stream,
    int output_dim = max_output_dim
){
    head_loop: for(int H = 0; H < head_num/head_parallel; H++){
        auto s_b_pack = input_s_b_stream.read();

        output_dim_loop: for(int k = 0; k < output_dim; k++){
        #pragma HLS PIPELINE II=1
            hls::vector<ap_int<qint_bit>, head_parallel> temp_pack = input_stream.read();
            hls::vector<float, head_parallel * (1+is_act_asym)> weight_s_sum_pack = weight_s_sum_stream.read();
            hls::vector<float, head_parallel> output_pack;
            dequant_loop: for(int m = 0; m < head_parallel; m++){
                if(is_act_asym){
                    float dequant_temp = temp_pack[m] * s_b_pack[2 * m] * weight_s_sum_pack[2 * m];
                    output_pack[m] = dequant_temp + s_b_pack[2 * m + 1] * weight_s_sum_pack[2 * m + 1];
                }
                else{
                    float dequant_temp = temp_pack[m] * s_b_pack[m] * weight_s_sum_pack[m];
                    output_pack[m] = dequant_temp;
                }
            }
            output_stream.write(output_pack);                
        }
    }
}


template <int qint_bit, int head_parallel, int max_input_dim, int head_num=1, bool is_signed=true>
void dec_MHA_static_sym_quant_layer_fp32_qint(
    tapa::istream<hls::vector<float, head_parallel>>& input_stream,
    tapa::istream<hls::vector<float, head_parallel>>& input_s_stream, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<qint_bit>, head_parallel>>& output_stream, //asym_quant: ap_uint<qint_bit>, sym_quant: ap_int<qint_bit>
    int input_dim = max_input_dim
){
    const int scale_base = is_signed ? (1 << (qint_bit - 1)) : (1 << qint_bit);

    head_num_loop: for (int H = 0; H < head_num/head_parallel; H++){
        auto s_pack = input_s_stream.read();
        
        sym_quant_loop: for(int k = 0; k < input_dim; k++){
        #pragma HLS loop_tripcount min=1 max=max_input_dim
        #pragma HLS PIPELINE II=1
            hls::vector<float, head_parallel> in_pack = input_stream.read();
            hls::vector<ap_int<qint_bit>, head_parallel> out_pack;
            for(int m = 0; m < head_parallel; m++){
                float scale_temp = in_pack[m] / s_pack[m];
                int round_temp = round(scale_temp);
                int out_temp;
                if(is_signed)
                    out_temp =  round_temp > scale_base - 1 ? scale_base - 1: 
                                round_temp < -scale_base ? -scale_base:
                                round_temp;
                else
                    out_temp =  round_temp > scale_base - 1 ? scale_base - 1: round_temp;
                out_pack[m] = (ap_int<qint_bit>) out_temp;
            }
            output_stream.write(out_pack);
        }
    }
}


template <int qint_bit, int head_parallel, int max_output_dim, int act_head_num=1, int weight_head_num=1>
void dec_MHA_static_sym_dequant_layer_qint_fp32(
    tapa::istream<hls::vector<ap_int<qint_bit>, head_parallel>>& input_stream,
    tapa::istream<hls::vector<float, head_parallel>>& input_s_stream, //input's scale factor and zero point
    tapa::istream<hls::vector<float, head_parallel>>& weight_s_stream, //weight's scale factor
    tapa::ostream<hls::vector<float, head_parallel>>& output_stream,
    int output_dim = max_output_dim
){
    head_loop: for(int h = 0; h < weight_head_num; h++){
        group_loop: for(int g = 0; g < act_head_num/weight_head_num/head_parallel; g++){
            auto s_pack = input_s_stream.read();
        
            output_dim_loop: for(int k = 0; k < output_dim; k++){
            #pragma HLS loop_tripcount min=1 max=max_output_dim
            #pragma HLS PIPELINE II=1
                hls::vector<ap_int<qint_bit>, head_parallel> temp_pack = input_stream.read();
                auto weight_s_pack = weight_s_stream.read();
                hls::vector<float, head_parallel> output_pack;
                dequant_loop: for(int m = 0; m < head_parallel; m++){
                    output_pack[m] = temp_pack[m] * s_pack[m] * weight_s_pack[k];
                }
                output_stream.write(output_pack);                
            }
        }
    }
}


template <int qint_bit, int head_parallel, int max_input_dim, int head_num=1, bool is_signed=true, int decoder_layer_num=DECODER_LAYER_NUM>
void dec_MHA_static_sym_per_tensor_quant_layer_fp32_qint(
    tapa::istream<hls::vector<float, head_parallel>>& input_stream,
    tapa::ostream<hls::vector<ap_int<qint_bit>, head_parallel>>& output_stream, //asym_quant: ap_uint<qint_bit>, sym_quant: ap_int<qint_bit>
    const float input_s[decoder_layer_num][head_num],
    int block_id,
    int input_dim = max_input_dim,
    float mha_scale_factor = 1.0
){
    #pragma HLS ARRAY_PARTITION variable=input_s type=cyclic factor=head_parallel dim=2
    const int scale_base = is_signed ? (1 << (qint_bit - 1)) : (1 << qint_bit);
    
    head_num_loop: for (int H = 0; H < head_num/head_parallel; H++){
        sym_quant_loop: for(int k = 0; k < input_dim; k++){
        #pragma HLS loop_tripcount min=1 max=max_input_dim
        #pragma HLS PIPELINE II=1
            hls::vector<float, head_parallel> in_pack = input_stream.read();

            hls::vector<ap_int<qint_bit>, head_parallel> out_pack;
            for(int m = 0; m < head_parallel; m++){
                float scale_temp = in_pack[m] / (mha_scale_factor * input_s[block_id][m * (head_num/head_parallel) + H]);
                int round_temp = round(scale_temp);
                int out_temp;
                if(is_signed)
                    out_temp =  round_temp > scale_base - 1 ? scale_base - 1: 
                                round_temp < -scale_base ? -scale_base:
                                round_temp;
                else
                    out_temp =  round_temp > scale_base - 1 ? scale_base - 1: round_temp;
                out_pack[m] = (ap_int<qint_bit>) out_temp;
            }
            output_stream.write(out_pack);

            // if(H == 0 && k == 0) cout << "head " << H << " scale: " << input_s[block_id][H] << endl;
            if(H == 0 && k == 0) cout << "input/output data: ";
            if(H == 0 && k < 8) cout << "(" << in_pack[0] << ", " << out_pack[0] << ") ";;
            if(H == 0 && k == 7) cout << endl;
        }
    }
}


template <int qint_bit, int head_parallel, int max_output_dim, int act_head_num=1, int weight_head_num=1, int decoder_layer_num=DECODER_LAYER_NUM>
void dec_MHA_static_sym_per_tensor_dequant_layer_qint_fp32(
    tapa::istream<hls::vector<ap_int<qint_bit>, head_parallel>>& input_stream,
    tapa::ostream<hls::vector<float, head_parallel>>& output_stream,
    const float input_s[decoder_layer_num][act_head_num],
    const float weight_s[decoder_layer_num][weight_head_num],
    int block_id,
    int output_dim = max_output_dim
){
    #pragma HLS ARRAY_PARTITION variable=input_s type=cyclic factor=head_parallel dim=2
     #pragma HLS ARRAY_PARTITION variable=weight_s type=cyclic factor=head_parallel dim=2
    
    // head_loop: for(int h = 0; h < weight_head_num; h++){
    //     group_loop: for(int g = 0; g < act_head_num/weight_head_num/head_parallel; g++){
    //         output_dim_loop: for(int k = 0; k < output_dim; k++){
    //         #pragma HLS PIPELINE II=1
    //             hls::vector<ap_int<qint_bit>, head_parallel> temp_pack = input_stream.read();
    //             hls::vector<float, head_parallel> output_pack;
    //             dequant_loop: for(int m = 0; m < head_parallel; m++){
    //                 output_pack[m] = temp_pack[m] * 
    //                     input_s[block_id][h*(act_head_num/weight_head_num) + g*head_parallel + m] * 
    //                     weight_s[block_id][h]; //suppose weight_head_num > head_parallel
    //             }
    //             output_stream.write(output_pack);                
    //         }
    //     }
    // }

    head_loop: for(int h = 0; h < act_head_num/head_parallel; h++){
        output_dim_loop: for(int k = 0; k < output_dim; k++){
        #pragma HLS loop_tripcount min=1 max=max_output_dim
        #pragma HLS PIPELINE II=1
            hls::vector<ap_int<qint_bit>, head_parallel> temp_pack = input_stream.read();
            hls::vector<float, head_parallel> output_pack;
            dequant_loop: for(int m = 0; m < head_parallel; m++){
                int a_H = m * (act_head_num/head_parallel) + h;
                int w_H = a_H / (act_head_num/weight_head_num);
                output_pack[m] = temp_pack[m] * input_s[block_id][a_H] * weight_s[block_id][w_H];
            }  
            output_stream.write(output_pack); 
            
            // if(h == 0 && k == 0) cout << "head " << 0 << " act_scale: " << input_s[block_id][0] << " weight_scale: " << weight_s[block_id][0] << endl;
            if(h == 0 && k == 0) cout << "input/output data: ";
            if(h == 0 && k < 8) cout << "(" << temp_pack[0] << ", " << output_pack[0] << ") ";;
            if(h == 0 && k == 7) cout << endl;
        }
    }
}





#endif