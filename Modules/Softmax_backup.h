#ifndef _SOFTMAX_H
#define _SOFTMAX_H
#include "config.h"


template <typename T, int io_parallel, int max_hidden_dim=HIDDEN_DIM, int max_seq_len=MAX_PRE_SEQ_LEN, int head_num=1, bool enable_scale=false, bool enable_sub_max=true, bool is_versal=false>
void pref_Softmax(
    tapa::istream<hls::vector<T, io_parallel>>& input_stream,
    tapa::ostream<hls::vector<T, io_parallel>>& output_stream,
    int seq_len = max_seq_len,
    int io_hidden_dim = max_hidden_dim,
    const T scale_factor = 1.0
){
    // ultrascale+ FPGA (U280, u250)
    if(!is_versal){
        T attn_exp[io_parallel][max_hidden_dim];
        #pragma HLS ARRAY_PARTITION variable=attn_exp dim=1 complete
        T attn_exp_sum[io_parallel][4];
        #pragma HLS ARRAY_PARTITION variable=attn_exp_sum complete
        T cd_attn_exp_sum[io_parallel];
        #pragma HLS ARRAY_PARTITION variable=cd_attn_exp_sum complete
        T attn_max[io_parallel];
        #pragma HLS ARRAY_PARTITION variable=attn_max complete

        io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
            #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
            attn_head_loop: for (int H = 0; H < head_num; H++){
                for(int i = 0; i < io_parallel; i++){
                #pragma HLS unroll
                    attn_exp_sum[i][0] = 0;
                    attn_exp_sum[i][1] = 0;
                    attn_exp_sum[i][2] = 0;
                    attn_exp_sum[i][3] = 0;
                    if(enable_sub_max) {
                        attn_max[i] = -1e32;
                    }
                }

                if(enable_sub_max){
                    max_loop: for (int k = 0; k < io_hidden_dim; k++) {
                    #pragma HLS loop_tripcount min=1 max=max_hidden_dim
                    #pragma HLS pipeline II=1
                        hls::vector<T, io_parallel> temp_pack = input_stream.read();
                        for(int i = 0; i < io_parallel; i++){
                            attn_max[i] = temp_pack[i] > attn_max[i] ? temp_pack[i] : attn_max[i];
                            attn_exp[i][k] = temp_pack[i];
                        }

                        if(M==0 && H == 0 && k == 0) cout << "softmax input data: ";
                        if(M==0 && H == 0 && k < 8) cout << temp_pack[0]<< " ";
                        if(M==0 && H == 0 && k == 7) cout << endl;
                    }

                    exp_loop_max_t: for (int k = 0; k < io_hidden_dim/4; k++) { // seq_len need to be padded to multiple of 4 in MHA
                    #pragma HLS loop_tripcount min=1 max=max_hidden_dim/4
                    #pragma HLS pipeline II=4
                    // #pragma HLS dependence variable=attn_exp_sum inter false
                        // for(int i = 0; i < io_parallel; i++){
                        //     T temp;
                        //     if(enable_scale) temp = exp((attn_exp[i][k] - attn_max[i])/scale_factor);
                        //     else temp = exp((attn_exp[i][k] - attn_max[i]));
                        //     attn_exp[i][k] = temp;
                        //     attn_exp_sum[i][k % 4] += temp;
                        //     #pragma HLS bind_op variable=attn_exp_sum op=fadd impl=fulldsp
                        // }
                        for(int i = 0; i < io_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp((attn_exp[i][4 * k] - attn_max[i])/scale_factor);
                            else temp = exp((attn_exp[i][4 * k] - attn_max[i]));
                            attn_exp[i][4 * k] = temp;
                            attn_exp_sum[i][0] += temp;
                        }
                        for(int i = 0; i < io_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp((attn_exp[i][4 * k + 1] - attn_max[i])/scale_factor);
                            else temp = exp((attn_exp[i][4 * k + 1] - attn_max[i]));
                            attn_exp[i][4 * k + 1] = temp;
                            attn_exp_sum[i][1] += temp;
                        }
                        for(int i = 0; i < io_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp((attn_exp[i][4 * k + 2] - attn_max[i])/scale_factor);
                            else temp = exp((attn_exp[i][4 * k + 2] - attn_max[i]));
                            attn_exp[i][4 * k + 2] = temp;
                            attn_exp_sum[i][2] += temp;
                        }
                        for(int i = 0; i < io_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp((attn_exp[i][4 * k + 3] - attn_max[i])/scale_factor);
                            else temp = exp((attn_exp[i][4 * k + 3] - attn_max[i]));
                            attn_exp[i][4 * k + 3] = temp;
                            attn_exp_sum[i][3] += temp;
                        }
                    }
                }

                else{
                    exp_loop_max_f: for (int k = 0; k < io_hidden_dim/4; k++) {
                    #pragma HLS loop_tripcount min=1 max=max_hidden_dim/4
                    #pragma HLS pipeline II=4
                    // #pragma HLS dependence variable=attn_exp_sum inter false
                        // hls::vector<T, io_parallel> temp_pack = input_stream.read();
                        // for(int i = 0; i < io_parallel; i++){
                        //     T temp;
                        //     if(enable_scale) temp = exp(temp_pack[i]/scale_factor);
                        //     else temp = exp(temp_pack[i]);
                        //     attn_exp[i][k] = temp;
                        //     attn_exp_sum[i][k % 4] += temp;
                        // }
                        hls::vector<T, io_parallel> temp_pack_0 = input_stream.read();
                        for(int i = 0; i < io_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp(temp_pack_0[i]/scale_factor);
                            else temp = exp(temp_pack_0[i]);
                            attn_exp[i][4 * k] = temp;
                            attn_exp_sum[i][0] += temp;
                        }
                        hls::vector<T, io_parallel> temp_pack_1 = input_stream.read();
                        for(int i = 0; i < io_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp(temp_pack_1[i]/scale_factor);
                            else temp = exp(temp_pack_1[i]);
                            attn_exp[i][4 * k + 1] = temp;
                            attn_exp_sum[i][1] += temp; 
                        }
                        hls::vector<T, io_parallel> temp_pack_2 = input_stream.read();
                        for(int i = 0; i < io_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp(temp_pack_2[i]/scale_factor);
                            else temp = exp(temp_pack_2[i]);
                            attn_exp[i][4 * k + 2] = temp;
                            attn_exp_sum[i][2] += temp;
                        }
                        hls::vector<T, io_parallel> temp_pack_3 = input_stream.read();
                        for(int i = 0; i < io_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp(temp_pack_3[i]/scale_factor);
                            else temp = exp(temp_pack_3[i]);
                            attn_exp[i][4 * k + 3] = temp;
                            attn_exp_sum[i][3] += temp;
                        }
                    }
                }

                countdown_exp_sum_loop: for (int i = 0; i < io_parallel; i++) {
                #pragma HLS pipeline II=1
                    cd_attn_exp_sum[i] = 1.0f / 
                        ((attn_exp_sum[i][0] + attn_exp_sum[i][1]) + 
                            (attn_exp_sum[i][2] + attn_exp_sum[i][3]));
                }
                output_scale_loop: for (int k = 0; k < io_hidden_dim; k++) {
                #pragma HLS loop_tripcount min=1 max=max_hidden_dim
                #pragma HLS pipeline II=1
                    hls::vector<T, io_parallel> outp_pack;
                    for(int i = 0; i < io_parallel; i++){
                        T temp = attn_exp[i][k] * cd_attn_exp_sum[i];
                        outp_pack[i] = temp;
                    }
                    output_stream.write(outp_pack);
                }
            }
        }
    }
    
    // versal FPGA (v80)
    else{
        T attn_exp[io_parallel][max_hidden_dim];
        #pragma HLS ARRAY_PARTITION variable=attn_exp dim=1 complete
        T attn_exp_sum[io_parallel];
        #pragma HLS ARRAY_PARTITION variable=attn_exp_sum complete
        T cd_attn_exp_sum[io_parallel];
        #pragma HLS ARRAY_PARTITION variable=cd_attn_exp_sum complete
        T attn_max[io_parallel];
        #pragma HLS ARRAY_PARTITION variable=attn_max complete

        io_block_loop_versal: for (int M = 0; M < seq_len/io_parallel; M++){
            #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
            attn_head_loop_versal: for (int H = 0; H < head_num; H++){
                for(int i = 0; i < io_parallel; i++){
                #pragma HLS unroll
                    attn_exp_sum[i] = 0;
                    if(enable_sub_max) {
                        attn_max[i] = -1e32;
                    }
                }

                if(enable_sub_max){
                    max_loop_versal: for (int k = 0; k < io_hidden_dim; k++) {
                    #pragma HLS loop_tripcount min=1 max=max_hidden_dim
                    #pragma HLS pipeline II=1
                        hls::vector<T, io_parallel> temp_pack = input_stream.read();
                        for(int i = 0; i < io_parallel; i++){
                            attn_max[i] = temp_pack[i] > attn_max[i] ? temp_pack[i] : attn_max[i];
                            attn_exp[i][k] = temp_pack[i];
                        }
                    }

                    exp_loop_max_t_versal: for (int k = 0; k < io_hidden_dim; k++) {
                    #pragma HLS loop_tripcount min=1 max=max_hidden_dim
                    #pragma HLS pipeline II=1
                        for(int i = 0; i < io_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp((attn_exp[i][k] - attn_max[i])/scale_factor);
                            else temp = exp((attn_exp[i][k] - attn_max[i]));
                            attn_exp[i][k] = temp;
                            attn_exp_sum[i] += temp;
                        }
                    }
                }

                else{
                    exp_loop_max_f_versal: for (int k = 0; k < io_hidden_dim; k++) {
                    #pragma HLS loop_tripcount min=1 max=max_hidden_dim
                    #pragma HLS pipeline II=1
                    #pragma HLS dependence variable=attn_exp_sum inter false
                        hls::vector<T, io_parallel> temp_pack = input_stream.read();
                        for(int i = 0; i < io_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp(temp_pack[i]/scale_factor);
                            else temp = exp(temp_pack[i]);
                            attn_exp[i][k] = temp;
                            attn_exp_sum[i] += temp;
                        }
                    }
                }

                countdown_exp_sum_loop_versal: for (int i = 0; i < io_parallel; i++) {
                #pragma HLS pipeline II=1
                    cd_attn_exp_sum[i] = 1.0f / attn_exp_sum[i];
                }
                output_scale_loop_versal: for (int k = 0; k < io_hidden_dim; k++) {
                #pragma HLS loop_tripcount min=1 max=max_hidden_dim
                #pragma HLS pipeline II=1
                    hls::vector<T, io_parallel> outp_pack;
                    for(int i = 0; i < io_parallel; i++){
                        T temp = attn_exp[i][k] * cd_attn_exp_sum[i];
                        outp_pack[i] = temp;
                    }
                    output_stream.write(outp_pack);
                }
            }
        }
    }
}


template <typename T, int block_parallel, int max_hidden_dim=HIDDEN_DIM, bool enable_scale=false, bool enable_sub_max=true, bool is_versal=false>
void dec_Softmax(
    tapa::istream<hls::vector<T, block_parallel>>& input_stream,
    tapa::ostream<hls::vector<T, block_parallel>>& output_stream,
    int io_hidden_dim = max_hidden_dim,
    const T scale_factor = 1.0
){
    // ultrascale+ FPGA (U280, u250)
    if(!is_versal){
        T attn_exp[block_parallel][max_hidden_dim/block_parallel];
        #pragma HLS ARRAY_PARTITION variable=attn_exp dim=1 complete
        T attn_exp_sum[block_parallel][4];
        #pragma HLS ARRAY_PARTITION variable=attn_exp_sum complete
        T attn_exp_block_sum[block_parallel];
        #pragma HLS ARRAY_PARTITION variable=attn_exp_block_sum complete
        T attn_max[block_parallel];
        #pragma HLS ARRAY_PARTITION variable=attn_max complete


        for(int i = 0; i < block_parallel; i++){
        #pragma HLS unroll
            attn_exp_sum[i][0] = 0;
            attn_exp_sum[i][1] = 0;
            attn_exp_sum[i][2] = 0;
            attn_exp_sum[i][3] = 0;
            if(enable_sub_max) {
                attn_max[i] = -1e32;
            }
        }

        if(enable_sub_max){
            max_loop: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
            #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel
            #pragma HLS pipeline II=1
                hls::vector<T, block_parallel> temp_pack = input_stream.read();
                for(int i = 0; i < block_parallel; i++){
                    attn_max[i] = temp_pack[i] > attn_max[i] ? temp_pack[i] : attn_max[i];
                    attn_exp[i][k] = temp_pack[i];
                }
            }

            T attn_max_token = attn_max[0];
            for(int i = 1; i < block_parallel; i++){
                attn_max_token = attn_max[i] > attn_max_token ? attn_max[i] : attn_max_token;
            }

            exp_loop_max_t: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
            #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel
            #pragma HLS pipeline II=1
            #pragma HLS dependence variable=attn_exp_sum inter false
                for(int i = 0; i < block_parallel; i++){
                    T temp;
                    if(enable_scale) temp = exp((attn_exp[i][k] - attn_max_token)/scale_factor);
                    else temp = exp((attn_exp[i][k] - attn_max_token));
                    attn_exp[i][k] = temp;
                    attn_exp_sum[i][k % 4] += temp;
                }
            }
        }

        else{
            exp_loop_max_f: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
            #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel
            #pragma HLS pipeline II=1
            #pragma HLS dependence variable=attn_exp_sum inter false
                hls::vector<T, block_parallel> temp_pack = input_stream.read();
                for(int i = 0; i < block_parallel; i++){
                    T temp;
                    if(enable_scale) temp = exp(temp_pack[i]/scale_factor);
                    else temp = exp(temp_pack[i]);
                    attn_exp[i][k] = temp;
                    attn_exp_sum[i][k % 4] += temp;
                }
            }
        }

        
        exp_sum_loop_block: for (int i = 0; i < block_parallel; i++) {
        #pragma HLS unroll
            attn_exp_block_sum[i] = 
                (attn_exp_sum[i][0] + attn_exp_sum[i][1]) + 
                (attn_exp_sum[i][2] + attn_exp_sum[i][3]);
        }

        T attn_exp_sum_total = 0;
        exp_sum_loop: for (int i = 0; i < block_parallel; i++) {
            attn_exp_sum_total += attn_exp_block_sum[i];
        }

        T cd_attn_exp_sum_total = 1.0f / attn_exp_sum_total;

        output_scale_loop: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
        #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel
        #pragma HLS pipeline II=1
            hls::vector<T, block_parallel> outp_pack;
            for(int i = 0; i < block_parallel; i++){
                T temp = attn_exp[i][k] * cd_attn_exp_sum_total;
                outp_pack[i] = temp;
            }
            output_stream.write(outp_pack);
        }
    }

    // versal FPGA (v80)
    else{
        T attn_exp[block_parallel][max_hidden_dim/block_parallel];
        #pragma HLS ARRAY_PARTITION variable=attn_exp dim=1 complete
        T attn_exp_sum[block_parallel];
        #pragma HLS ARRAY_PARTITION variable=attn_exp_sum complete
        T attn_max[block_parallel];
        #pragma HLS ARRAY_PARTITION variable=attn_max complete


        for(int i = 0; i < block_parallel; i++){
        #pragma HLS unroll
            attn_exp_sum[i] = 0;
            if(enable_sub_max) {
                attn_max[i] = -1e6;
            }
        }

        if(enable_sub_max){
            max_loop_versal: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
            #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel
            #pragma HLS pipeline II=1
                hls::vector<T, block_parallel> temp_pack = input_stream.read();
                for(int i = 0; i < block_parallel; i++){
                    attn_max[i] = temp_pack[i] > attn_max[i] ? temp_pack[i] : attn_max[i];
                    attn_exp[i][k] = temp_pack[i];
                }
            }

            T attn_max_token = attn_max[0];
            for(int i = 1; i < block_parallel; i++){
                attn_max_token = attn_max[i] > attn_max_token ? attn_max[i] : attn_max_token;
            }

            exp_loop_max_t_versal: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
            #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel
            #pragma HLS pipeline II=1
                for(int i = 0; i < block_parallel; i++){
                    T temp;
                    if(enable_scale) temp = exp((attn_exp[i][k] - attn_max_token)/scale_factor);
                    else temp = exp((attn_exp[i][k] - attn_max_token));
                    attn_exp[i][k] = temp;
                    attn_exp_sum[i] += temp;
                }
            }
        }

        else{
            exp_loop_max_f_versal: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
            #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel
            #pragma HLS pipeline II=1
                hls::vector<T, block_parallel> temp_pack = input_stream.read();
                for(int i = 0; i < block_parallel; i++){
                    T temp;
                    if(enable_scale) temp = exp(temp_pack[i]/scale_factor);
                    else temp = exp(temp_pack[i]);
                    attn_exp[i][k] = temp;
                    attn_exp_sum[i] += temp;
                }
            }
        }

        T attn_exp_sum_total = 0;
        exp_sum_loop_versal: for (int i = 0; i < block_parallel; i++) {
            attn_exp_sum_total += attn_exp_sum[i];
        }

        T cd_attn_exp_sum_total = 1.0f / attn_exp_sum_total;

        output_scale_loop_versal: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
        #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel
        #pragma HLS pipeline II=1
            hls::vector<T, block_parallel> outp_pack;
            for(int i = 0; i < block_parallel; i++){
                T temp = attn_exp[i][k] * cd_attn_exp_sum_total;
                outp_pack[i] = temp;
            }
            output_stream.write(outp_pack);
        }
    }
}

template <typename T, int head_parallel, int max_hidden_dim=HIDDEN_DIM, int head_num=1, bool enable_scale=false, bool enable_sub_max=true, bool is_versal=false>
void dec_MHA_Softmax(
    tapa::istream<hls::vector<T, head_parallel>>& input_stream,
    tapa::ostream<hls::vector<T, head_parallel>>& output_stream,
    int io_hidden_dim = max_hidden_dim,
    const T scale_factor = 1.0
){
    // ultrascale+ FPGA (U280, u250)
    if(!is_versal){
        T attn_exp[head_parallel][max_hidden_dim];
        #pragma HLS ARRAY_PARTITION variable=attn_exp dim=1 complete
        T attn_exp_sum[head_parallel][4];
        #pragma HLS ARRAY_PARTITION variable=attn_exp_sum complete
        T cd_attn_exp_sum[head_parallel];
        #pragma HLS ARRAY_PARTITION variable=cd_attn_exp_sum complete
        T attn_max[head_parallel];
        #pragma HLS ARRAY_PARTITION variable=attn_max complete

        attn_head_loop: for (int H = 0; H < head_num/head_parallel; H++){
            for(int i = 0; i < head_parallel; i++){
            #pragma HLS unroll
                attn_exp_sum[i][0] = 0;
                attn_exp_sum[i][1] = 0;
                attn_exp_sum[i][2] = 0;
                attn_exp_sum[i][3] = 0;
                if(enable_sub_max) {
                    attn_max[i] = -1e32;
                }
            }

            if(enable_sub_max){
                max_loop: for (int k = 0; k < io_hidden_dim; k++) {
                #pragma HLS loop_tripcount min=1 max=max_hidden_dim
                #pragma HLS pipeline II=1
                    hls::vector<T, head_parallel> temp_pack = input_stream.read();
                    for(int i = 0; i < head_parallel; i++){
                        attn_max[i] = temp_pack[i] > attn_max[i] ? temp_pack[i] : attn_max[i];
                        attn_exp[i][k] = temp_pack[i];
                    }

                    if(H == 0 && k == 0) cout << "softmax input data: ";
                    if(H == 0 && k < 8) cout << temp_pack[0]<< " ";
                    if(H == 0 && k == 7) cout << endl;
                }

                // exp_loop_max_t: for (int k = 0; k < io_hidden_dim; k++) {
                // #pragma HLS loop_tripcount min=1 max=max_hidden_dim
                // #pragma HLS pipeline II=1
                // #pragma HLS dependence variable=attn_exp_sum inter false
                //     for(int i = 0; i < head_parallel; i++){
                //         T temp;
                //         if(enable_scale) temp = exp((attn_exp[i][k] - attn_max[i])/scale_factor);
                //         else temp = exp((attn_exp[i][k] - attn_max[i]));
                //         attn_exp[i][k] = temp;
                //         attn_exp_sum[i][k % 4] += temp;
                //     }
                // }

                exp_loop_max_t: for (int k = 0; k < (io_hidden_dim + 3) / 4; k++) {
                #pragma HLS loop_tripcount min=1 max= (max_hidden_dim + 3) /4
                #pragma HLS pipeline II=4
                    if(4 * k < io_hidden_dim){
                        for(int i = 0; i < head_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp((attn_exp[i][4 * k] - attn_max[i])/scale_factor);
                            else temp = exp((attn_exp[i][4 * k] - attn_max[i]));
                            attn_exp[i][4 * k] = temp;
                            attn_exp_sum[i][0] += temp;
                        }
                    }
                    if(4 * k + 1 < io_hidden_dim){
                        for(int i = 0; i < head_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp((attn_exp[i][4 * k + 1] - attn_max[i])/scale_factor);
                            else temp = exp((attn_exp[i][4 * k + 1] - attn_max[i]));
                            attn_exp[i][4 * k + 1] = temp;
                            attn_exp_sum[i][1] += temp; 
                        }
                    }
                    if(4 * k + 2 < io_hidden_dim){
                        for(int i = 0; i < head_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp((attn_exp[i][4 * k + 2] - attn_max[i])/scale_factor);
                            else temp = exp((attn_exp[i][4 * k + 2] - attn_max[i]));
                            attn_exp[i][4 * k + 2] = temp;
                            attn_exp_sum[i][2] += temp;
                        }   
                    }
                    if(4 * k + 3 < io_hidden_dim){
                        for(int i = 0; i < head_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp((attn_exp[i][4 * k + 3] - attn_max[i])/scale_factor);
                            else temp = exp((attn_exp[i][4 * k + 3] - attn_max[i]));
                            attn_exp[i][4 * k + 3] = temp;
                            attn_exp_sum[i][3] += temp;
                        }
                    }
                }
            }

            else{
                // exp_loop_max_f: for (int k = 0; k < io_hidden_dim; k++) {
                // #pragma HLS loop_tripcount min=1 max=max_hidden_dim
                // #pragma HLS pipeline II=1
                // #pragma HLS dependence variable=attn_exp_sum inter false
                //     hls::vector<T, head_parallel> temp_pack = input_stream.read();
                //     for(int i = 0; i < head_parallel; i++){
                //         T temp;
                //         if(enable_scale) temp = exp(temp_pack[i]/scale_factor);
                //         else temp = exp(temp_pack[i]);
                //         attn_exp[i][k] = temp;
                //         attn_exp_sum[i][k % 4] += temp;
                //     }
                // }

                exp_loop_max_f: for (int k = 0; k < (io_hidden_dim + 3) / 4; k++) {
                #pragma HLS loop_tripcount min=1 max=(max_hidden_dim + 3) /4
                #pragma HLS pipeline II=4
                    if(4 * k < io_hidden_dim){
                        hls::vector<T, head_parallel> temp_pack_0 = input_stream.read();
                        for(int i = 0; i < head_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp(temp_pack_0[i]/scale_factor);
                            else temp = exp(temp_pack_0[i]);
                            attn_exp[i][4 * k] = temp;
                            attn_exp_sum[i][0] += temp;
                        }
                    }
                    if(4 * k + 1 < io_hidden_dim){
                        hls::vector<T, head_parallel> temp_pack_1 = input_stream.read();
                        for(int i = 0; i < head_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp(temp_pack_1[i]/scale_factor);
                            else temp = exp(temp_pack_1[i]);
                            attn_exp[i][4 * k + 1] = temp;
                            attn_exp_sum[i][1] += temp; 
                        }
                    }
                    if(4 * k + 2 < io_hidden_dim){
                        hls::vector<T, head_parallel> temp_pack_2 = input_stream.read();
                        for(int i = 0; i < head_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp(temp_pack_2[i]/scale_factor);
                            else temp = exp(temp_pack_2[i]);
                            attn_exp[i][4 * k + 2] = temp;
                            attn_exp_sum[i][2] += temp; 
                        }
                    }
                    if(4 * k + 3 < io_hidden_dim){
                        hls::vector<T, head_parallel> temp_pack_3 = input_stream.read();
                        for(int i = 0; i < head_parallel; i++){
                            T temp;
                            if(enable_scale) temp = exp(temp_pack_3[i]/scale_factor);
                            else temp = exp(temp_pack_3[i]);
                            attn_exp[i][4 * k + 3] = temp;
                            attn_exp_sum[i][3] += temp; 
                        }
                    }   
                }
            }

            
            countdown_exp_sum_loop: for (int i = 0; i < head_parallel; i++) {
            #pragma HLS pipeline II=1
                cd_attn_exp_sum[i] = 1.0f / 
                    ((attn_exp_sum[i][0] + attn_exp_sum[i][1]) + 
                        (attn_exp_sum[i][2] + attn_exp_sum[i][3]));
            }
            output_scale_loop: for (int k = 0; k < io_hidden_dim; k++) {
            #pragma HLS loop_tripcount min=1 max=max_hidden_dim
            #pragma HLS pipeline II=1
                hls::vector<T, head_parallel> outp_pack;
                for(int i = 0; i < head_parallel; i++){
                    T temp = attn_exp[i][k] * cd_attn_exp_sum[i];
                    outp_pack[i] = temp;
                }
                output_stream.write(outp_pack);
            }
        }
    }

    // versal FPGA (v80)
    else{
        T attn_exp[head_parallel][max_hidden_dim];
        #pragma HLS ARRAY_PARTITION variable=attn_exp dim=1 complete
        T attn_exp_sum[head_parallel];
        #pragma HLS ARRAY_PARTITION variable=attn_exp_sum complete
        T cd_attn_exp_sum[head_parallel];
        #pragma HLS ARRAY_PARTITION variable=cd_attn_exp_sum complete
        T attn_max[head_parallel];
        #pragma HLS ARRAY_PARTITION variable=attn_max complete

        attn_head_loop_versal: for (int H = 0; H < head_num/head_parallel; H++){
            for(int i = 0; i < head_parallel; i++){
            #pragma HLS unroll
                attn_exp_sum[i] = 0;
                if(enable_sub_max) {
                    attn_max[i] = -1e32;
                }
            }

            if(enable_sub_max){
                max_loop_versal: for (int k = 0; k < io_hidden_dim; k++) {
                #pragma HLS loop_tripcount min=1 max=max_hidden_dim
                #pragma HLS pipeline II=1
                    hls::vector<T, head_parallel> temp_pack = input_stream.read();
                    for(int i = 0; i < head_parallel; i++){
                        attn_max[i] = temp_pack[i] > attn_max[i] ? temp_pack[i] : attn_max[i];
                        attn_exp[i][k] = temp_pack[i];
                    }
                }

                exp_loop_max_t_versal: for (int k = 0; k < io_hidden_dim; k++) {
                #pragma HLS loop_tripcount min=1 max=max_hidden_dim
                #pragma HLS pipeline II=1
                    for(int i = 0; i < head_parallel; i++){
                        T temp;
                        if(enable_scale) temp = exp((attn_exp[i][k] - attn_max[i])/scale_factor);
                        else temp = exp((attn_exp[i][k] - attn_max[i]));
                        attn_exp[i][k] = temp;
                        attn_exp_sum[i] += temp;
                    }
                }
            }

            else{
                exp_loop_max_f_versal: for (int k = 0; k < io_hidden_dim; k++) {
                #pragma HLS loop_tripcount min=1 max=max_hidden_dim
                #pragma HLS pipeline II=1
                    hls::vector<T, head_parallel> temp_pack = input_stream.read();
                    for(int i = 0; i < head_parallel; i++){
                        T temp;
                        if(enable_scale) temp = exp(temp_pack[i]/scale_factor);
                        else temp = exp(temp_pack[i]);
                        attn_exp[i][k] = temp;
                        attn_exp_sum[i] += temp;
                    }
                }
            }

            countdown_exp_sum_loop_versal: for (int i = 0; i < head_parallel; i++) {
            #pragma HLS pipeline II=1
                cd_attn_exp_sum[i] = 1.0f / attn_exp_sum[i];
            }
            output_scale_loop_versal: for (int k = 0; k < io_hidden_dim; k++) {
            #pragma HLS loop_tripcount min=1 max=max_hidden_dim
            #pragma HLS pipeline II=1
                hls::vector<T, head_parallel> outp_pack;
                for(int i = 0; i < head_parallel; i++){
                    T temp = attn_exp[i][k] * cd_attn_exp_sum[i];
                    outp_pack[i] = temp;
                }
                output_stream.write(outp_pack);
            }
        }
    }
    
}




#endif