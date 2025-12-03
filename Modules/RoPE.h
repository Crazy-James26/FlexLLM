#ifndef _RoPE_H_
#define _RoPE_H_
#include "config.h"




template <typename T, int io_parallel, int io_hidden_dim = HEAD_DIM, int max_head_num = Q_HEAD_NUM, int max_seq_Len = MAX_PRE_SEQ_LEN>
void pref_RoPE_layer(
    tapa::istream<hls::vector<T, io_parallel>>& input_stream,
    tapa::ostream<hls::vector<T, io_parallel>>& output_stream,
    const T PE_sin[max_seq_Len][io_hidden_dim/2],
    const T PE_cos[max_seq_Len][io_hidden_dim/2],
    int head_num = max_head_num,
    int seq_len = max_seq_Len
    // const T rope_theta = 500000,
    // const int factor = 32,
    // const int low_freq_factor = 1,
    // const int high_freq_factor = 4,
    // const int original_max_pe = 8192
){
    // static T PE_sin[max_seq_Len][io_hidden_dim/2];
    // static T PE_cos[max_seq_Len][io_hidden_dim/2];
    // #pragma HLS bind_storage variable=PE_sin type=ram_2p impl=uram
    // #pragma HLS bind_storage variable=PE_cos type=ram_2p impl=uram


    // const int high_wave_length = original_max_pe / high_freq_factor;
    // const int low_wave_length = original_max_pe / low_freq_factor;
    // init_cos_sin: for(int i = 0; i < io_hidden_dim/2; i++){
    //     T exponent = (2.0f * i) / T(io_hidden_dim);
    //     T inv_freq = std::pow(rope_theta, exponent);
    //     T freq = 1.0f / inv_freq;
    //     T wavelength = 2 * M_PI * inv_freq;

    //     T scale_freq;
    //     if(wavelength < high_wave_length) {
    //         scale_freq = freq;
    //     } else if(wavelength > low_wave_length){
    //         scale_freq = freq / factor;
    //     } else{
    //         T a =  (original_max_pe / wavelength - low_freq_factor) / (high_freq_factor - low_freq_factor);
    //         scale_freq = (1 - a) * (freq / factor) + a * freq;
    //     }
        
    //     for(int s = 0; s < max_seq_Len; s++){
    //         T theta = s * scale_freq;
    //         PE_sin[s][i] = std::sin(theta);
    //         PE_cos[s][i] = std::cos(theta);
    //     }
    // }

    
    hls::vector<T, io_parallel> reg_0[io_hidden_dim/2];
    hls::vector<T, io_parallel> reg_1[io_hidden_dim/2];
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_Len/io_parallel
        attn_head_loop: for (int H = 0; H < head_num; H++){
        #pragma HLS loop_tripcount min=1 max=max_head_num

            load_first_half_input_loop: for(int k = 0; k < io_hidden_dim/2; k++){
            #pragma HLS PIPELINE II=1
                reg_0[k] = input_stream.read();
            }

            load_second_half_loop: for(int k = 0; k < io_hidden_dim/2; k++){
            #pragma HLS PIPELINE II=1
                hls::vector<T, io_parallel> temp_pack_0 = reg_0[k];
                hls::vector<T, io_parallel> temp_pack_1 = input_stream.read();
                hls::vector<T, io_parallel> out_pack_0;
                hls::vector<T, io_parallel> out_pack_1;
                for(int m = 0; m < io_parallel; m++){
                    T temp_0 = temp_pack_0[m];
                    T temp_1 = temp_pack_1[m];
                    int s = M * io_parallel + m;
                    T sin_val = PE_sin[s][k];
                    T cos_val = PE_cos[s][k];
                    T out_0 = temp_0 * cos_val - temp_1 * sin_val;
                    T out_1 = temp_0 * sin_val + temp_1 * cos_val;
                    out_pack_0[m] = out_0;
                    out_pack_1[m] = out_1;
                }
                output_stream.write(out_pack_0);
                reg_1[k] = out_pack_1;
            }

            output_second_half_loop: for(int k = 0; k < io_hidden_dim/2; k++){
            #pragma HLS PIPELINE II=1
                output_stream.write(reg_1[k]);
            }
        }
    }

}

template <typename T, int head_parallel, int io_hidden_dim = HEAD_DIM, int max_head_num = Q_HEAD_NUM, int max_sum_seq_len=MAX_SUM_SEQ_LEN>
void dec_RoPE_layer(
    tapa::istream<hls::vector<T, head_parallel>>& input_stream,
    tapa::ostream<hls::vector<T, head_parallel>>& output_stream,
    const T PE_sin[max_sum_seq_len][io_hidden_dim/2],
    const T PE_cos[max_sum_seq_len][io_hidden_dim/2],
    int head_num = max_head_num,
    int seq_id = 0
    // const T rope_theta = 500000,
    // const int factor = 32,
    // const int low_freq_factor = 1,
    // const int high_freq_factor = 4,
    // const int original_max_pe = 8192
){
    // static T PE_sin[max_sum_seq_len][io_hidden_dim/2];
    // static T PE_cos[max_sum_seq_len][io_hidden_dim/2];

    // const int high_wave_length = original_max_pe / high_freq_factor;
    // const int low_wave_length = original_max_pe / low_freq_factor;
    // init_cos_sin: for(int i = 0; i < io_hidden_dim/2; i++){
    //     T exponent = (2.0f * i) / T(io_hidden_dim);
    //     T inv_freq = std::pow(rope_theta, exponent);
    //     T freq = 1.0f / inv_freq;
    //     T wavelength = 2 * M_PI * inv_freq;

    //     T scale_freq;
    //     if(wavelength < high_wave_length) {
    //         scale_freq = freq;
    //     } else if(wavelength > low_wave_length){
    //         scale_freq = freq / factor;
    //     } else{
    //         T a =  (original_max_pe / wavelength - low_freq_factor) / (high_freq_factor - low_freq_factor);
    //         scale_freq = (1 - a) * (freq / factor) + a * freq;
    //     }
        
    //     for(int s = 0; s < max_sum_seq_len; s++){
    //         T theta = s * scale_freq;
    //         PE_sin[s][i] = std::sin(theta);
    //         PE_cos[s][i] = std::cos(theta);
    //     }
    // }


    hls::vector<T, head_parallel> reg_0[io_hidden_dim/2];
    hls::vector<T, head_parallel> reg_1[io_hidden_dim/2];
    attn_head_loop: for (int H = 0; H < head_num/head_parallel; H++){
    #pragma HLS loop_tripcount min=1 max=max_head_num/head_parallel
        load_first_half_input_loop: for(int k = 0; k < io_hidden_dim/2; k++){
        #pragma HLS PIPELINE II=1
            reg_0[k] = input_stream.read();
        }

        load_second_half_loop: for(int k = 0; k < io_hidden_dim/2; k++){
        #pragma HLS PIPELINE II=1
            T sin_val = PE_sin[seq_id][k];
            T cos_val = PE_cos[seq_id][k];
            hls::vector<T, head_parallel> temp_pack_0 = reg_0[k];
            hls::vector<T, head_parallel> temp_pack_1 = input_stream.read();
            hls::vector<T, head_parallel> out_pack_0;
            hls::vector<T, head_parallel> out_pack_1;
            for(int m = 0; m < head_parallel; m++){
                T temp_0 = temp_pack_0[m];
                T temp_1 = temp_pack_1[m];
                T out_0 = temp_0 * cos_val - temp_1 * sin_val;
                T out_1 = temp_0 * sin_val + temp_1 * cos_val;
                out_pack_0[m] = out_0;
                out_pack_1[m] = out_1;
            }
            output_stream.write(out_pack_0);
            reg_1[k] = out_pack_1;
        }

        output_second_half_loop: for(int k = 0; k < io_hidden_dim/2; k++){
        #pragma HLS PIPELINE II=1
            output_stream.write(reg_1[k]);
        }
    }    
}

#endif