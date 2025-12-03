#ifndef _LAYER_NORM_H_
#define _LAYER_NORM_H_
#include "config.h"


template <typename T, int io_hidden_dim = HIDDEN_DIM, bool enble_beta = false, int decoder_layer_num = DECODER_LAYER_NUM>
void pref_Layer_Norm_gamma_beta_loader(
    tapa::mmap<T> gamma_beta_mmap,
    tapa::ostream<T>& gamma_beta_stream,
    int block_id
){
    gamma_loop: for(int i = 0; i < io_hidden_dim; i++){
    #pragma HLS pipeline II=1
        T gamma = gamma_beta_mmap[block_id * io_hidden_dim + i];
        gamma_beta_stream.write(gamma);
    }
    if(enble_beta){
        beta_loop: for(int i = 0; i < io_hidden_dim; i++){
        #pragma HLS pipeline II=1
            T beta = gamma_beta_mmap[(decoder_layer_num + block_id) * io_hidden_dim + i];
            gamma_beta_stream.write(beta);
        }
    }
}



template <typename T, int io_parallel, int max_hidden_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN, bool enble_beta = false>
void pref_Layer_Norm(
    tapa::istream<hls::vector<T, io_parallel>>& input_stream,
    tapa::istream<T>& gamma_beta_stream,
    tapa::ostream<hls::vector<T, io_parallel>>& output_stream,
    int seq_len = max_seq_len,
    int io_hidden_dim = max_hidden_dim,
    const T eps = 0.00001
){
    T gamma[max_hidden_dim];
    T beta[max_hidden_dim];

    gamma_load_loop: for (int k = 0; k < io_hidden_dim; k++) {
    #pragma HLS pipeline II=1
        gamma[k] = gamma_beta_stream.read();
    }
    

    if(enble_beta){
        beta_load_loop: for (int k = 0; k < io_hidden_dim; k++) {
        #pragma HLS pipeline II=1
            beta[k] = gamma_beta_stream.read();
        }
    }

    // ultrascale+ FPGA (U280, u250)
    T A[io_parallel][max_hidden_dim];
    #pragma HLS ARRAY_PARTITION variable=A dim=1 complete
    T A_square_sum[io_parallel][4];
    #pragma HLS ARRAY_PARTITION variable=A_square_sum complete
    T A_RMS_cd[io_parallel];
    #pragma HLS ARRAY_PARTITION variable=A_RMS_cd complete

    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
        #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        init_sum_loop: for (int i = 0; i < io_parallel; i++) {
        #pragma HLS unroll
            A_square_sum[i][0] = 0;
            A_square_sum[i][1] = 0;
            A_square_sum[i][2] = 0;
            A_square_sum[i][3] = 0;
        }

        // in_buf_loop: for (int k = 0; k < io_hidden_dim; k++) {
        // #pragma HLS pipeline II=1
        //     hls::vector<T, io_parallel> temp_pack = input_stream.read();
        //     for(int i = 0; i < io_parallel; i++){
        //         T temp = temp_pack[i];
        //         A[i][k] = temp;
        //         A_square_sum[i][k % 4] += temp * temp;
        //     }
        // }

        in_buf_loop: for (int k = 0; k < io_hidden_dim/4; k++) {
        #pragma HLS pipeline II=4
            hls::vector<T, io_parallel> temp_pack_0 = input_stream.read();
            for(int i = 0; i < io_parallel; i++){
                T temp = temp_pack_0[i];
                A[i][4 * k] = temp;
                A_square_sum[i][0] += temp * temp;
            }
            hls::vector<T, io_parallel> temp_pack_1 = input_stream.read();
            for(int i = 0; i < io_parallel; i++){
                T temp = temp_pack_1[i];
                A[i][4 * k + 1] = temp;
                A_square_sum[i][1] += temp * temp;
            }
            hls::vector<T, io_parallel> temp_pack_2 = input_stream.read();
            for(int i = 0; i < io_parallel; i++){
                T temp = temp_pack_2[i];
                A[i][4 * k + 2] = temp;
                A_square_sum[i][2] += temp * temp; 
            }
            hls::vector<T, io_parallel> temp_pack_3 = input_stream.read();
            for(int i = 0; i < io_parallel; i++){
                T temp = temp_pack_3[i];
                A[i][4 * k + 3] = temp;
                A_square_sum[i][3] += temp * temp;
            }
        }

        Countdown_RMS_loop: for (int i = 0; i < io_parallel; i++) {
        #pragma HLS pipeline II=1
            A_RMS_cd[i] = 1.0f / sqrt(
                ((A_square_sum[i][0] + A_square_sum[i][1]) + (A_square_sum[i][2] + A_square_sum[i][3])) / io_hidden_dim + eps
            );
        }

        output_scale_loop: for (int k = 0; k < io_hidden_dim; k++) {
        #pragma HLS pipeline II=1
            hls::vector<T, io_parallel> outp_pack;
            for(int i = 0; i < io_parallel; i++){
                T temp = A[i][k] * A_RMS_cd[i] * gamma[k];
                if(enble_beta){
                    T temp_beta = temp + beta[k];
                    outp_pack[i] = temp_beta;
                }
                else{
                    outp_pack[i] = temp;
                }
            }
            output_stream.write(outp_pack);
        }
    }


    // // versal FPGA (v80)
    // T A[io_parallel][max_hidden_dim];
    // #pragma HLS ARRAY_PARTITION variable=A dim=1 complete
    // T A_square_sum[io_parallel];
    // #pragma HLS ARRAY_PARTITION variable=A_square_sum complete
    // T A_RMS_cd[io_parallel];
    // #pragma HLS ARRAY_PARTITION variable=A_RMS_cd complete

    // io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    //     #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
    //     init_sum_loop: for (int i = 0; i < io_parallel; i++) {
    //     #pragma HLS unroll
    //         A_square_sum[i] = 0;
    //     }

    //     in_buf_loop: for (int k = 0; k < io_hidden_dim; k++) {
    //     #pragma HLS pipeline II=1
    //         hls::vector<T, io_parallel> temp_pack = input_stream.read();
    //         for(int i = 0; i < io_parallel; i++){
    //             T temp = temp_pack[i];
    //             A[i][k] = temp;
    //             A_square_sum[i] += temp * temp;
    //         }
    //     }

    //     Countdown_RMS_loop: for (int i = 0; i < io_parallel; i++) {
    //     #pragma HLS pipeline II=1
    //         A_RMS_cd[i] = 1.0f / sqrt(A_square_sum[i] / io_hidden_dim + eps);
    //     }

    //     output_scale_loop: for (int k = 0; k < io_hidden_dim; k++) {
    //     #pragma HLS pipeline II=1
    //         hls::vector<T, io_parallel> outp_pack;
    //         for(int i = 0; i < io_parallel; i++){
    //             T temp = A[i][k] * A_RMS_cd[i] * gamma[k];
    //             if(enble_beta){
    //                 T temp_beta = temp + beta[k];
    //                 outp_pack[i] = temp_beta;
    //             }
    //             else{
    //                 outp_pack[i] = temp;
    //             }
    //         }
    //         output_stream.write(outp_pack);
    //     }
    // }
}


template <typename T, int block_parallel, int io_hidden_dim = HIDDEN_DIM, bool enble_beta = false, int decoder_layer_num = DECODER_LAYER_NUM>
void dec_Layer_Norm_gamma_beta_loader(
    tapa::mmap<hls::vector<T, block_parallel>> gamma_beta_mmap,
    tapa::ostream<hls::vector<T, block_parallel>>& gamma_beta_stream,
    int block_id,
    int addr_bias = 0
){
    int bias = addr_bias + block_id * io_hidden_dim/block_parallel;
    gamma_loop: for(int i = 0; i < io_hidden_dim/block_parallel; i++) {
    #pragma HLS pipeline II=1
        hls::vector<T, block_parallel> gamma_beta_pack = gamma_beta_mmap[bias + i];
        gamma_beta_stream.write(gamma_beta_pack);
    }
    if(enble_beta){
        int bias = addr_bias + (decoder_layer_num + block_id) * io_hidden_dim/block_parallel;
        beta_loop: for(int i = 0; i < io_hidden_dim/block_parallel; i++) {
        #pragma HLS pipeline II=1
            hls::vector<T, block_parallel> beta_pack = gamma_beta_mmap[bias + i];
            gamma_beta_stream.write(beta_pack);
        }
    }
}


template <typename T, int block_parallel, int max_hidden_dim = HIDDEN_DIM, bool enble_beta = false>
void dec_Layer_Norm(
    tapa::istream<hls::vector<T, block_parallel>>& input_stream,
    tapa::istream<hls::vector<T, block_parallel>>& gamma_beta_stream,
    tapa::ostream<hls::vector<T, block_parallel>>& output_stream,
    int io_hidden_dim = max_hidden_dim,
    const T eps = 0.00001
){
    hls::vector<T, block_parallel> gamma[max_hidden_dim/block_parallel];
    hls::vector<T, block_parallel> beta[max_hidden_dim/block_parallel];

    gamma_load_loop: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
    #pragma HLS pipeline II=1
        gamma[k] = gamma_beta_stream.read();
    }

    if(enble_beta){
        beta_load_loop: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
        #pragma HLS pipeline II=1
            beta[k] = gamma_beta_stream.read();
        }
    }

    // ultrascale+ FPGA (U280, u250)
    T A[block_parallel][max_hidden_dim/block_parallel];
    #pragma HLS ARRAY_PARTITION variable=A dim=1 complete
    T A_square_sum[block_parallel][4];
    #pragma HLS ARRAY_PARTITION variable=A_square_sum complete
    T A_square_block_sum[block_parallel];
    #pragma HLS ARRAY_PARTITION variable=A_square_block_sum complete

    init_sum_loop: for (int i = 0; i < block_parallel; i++) {
    #pragma HLS unroll
        A_square_sum[i][0] = 0;
        A_square_sum[i][1] = 0;
        A_square_sum[i][2] = 0;
        A_square_sum[i][3] = 0;
    }

    // in_buf_loop: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
    // #pragma HLS pipeline II=1
    //     hls::vector<T, block_parallel> temp_pack = input_stream.read();
    //     for(int i = 0; i < block_parallel; i++){
    //         T temp = temp_pack[i];
    //         A[i][k] = temp;
    //         A_square_sum[i][k % 4] += temp * temp;
    //     }
    //     if(k == 0) cout << "LN input data: ";
    //     if(k < 16) cout << temp_pack[0] << " ";
    //     if(k == 15) cout << endl;
    // }

    in_buf_loop: for (int k = 0; k < io_hidden_dim/block_parallel/4; k++) {
    #pragma HLS pipeline II=4
        hls::vector<T, block_parallel> temp_pack_0 = input_stream.read();
        for(int i = 0; i < block_parallel; i++){
            T temp = temp_pack_0[i];
            A[i][4 * k] = temp;
            A_square_sum[i][0] += temp * temp;
        }
        hls::vector<T, block_parallel> temp_pack_1 = input_stream.read();
        for(int i = 0; i < block_parallel; i++){
            T temp = temp_pack_1[i];
            A[i][4 * k + 1] = temp;
            A_square_sum[i][1] += temp * temp;
        }
        hls::vector<T, block_parallel> temp_pack_2 = input_stream.read();
        for(int i = 0; i < block_parallel; i++){
            T temp = temp_pack_2[i];
            A[i][4 * k + 2] = temp;
            A_square_sum[i][2] += temp * temp; 
        }
        hls::vector<T, block_parallel> temp_pack_3 = input_stream.read();
        for(int i = 0; i < block_parallel; i++){
            T temp = temp_pack_3[i];
            A[i][4 * k + 3] = temp;
            A_square_sum[i][3] += temp * temp;
        }
    }

    sum_loop_block: for (int i = 0; i < block_parallel; i++) {
        A_square_block_sum[i] =
            (A_square_sum[i][0] + A_square_sum[i][1]) + 
            (A_square_sum[i][2] + A_square_sum[i][3]);
    }

    T A_square_sum_total = 0;
    sum_loop: for (int i = 0; i < block_parallel; i++) {
        A_square_sum_total += A_square_block_sum[i];
    }

    T A_RMS_cd = 1.0f / sqrt(A_square_sum_total / io_hidden_dim + eps);

    output_scale_loop: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
    #pragma HLS pipeline II=1
        hls::vector<T, block_parallel> outp_pack;
        for(int i = 0; i < block_parallel; i++){
            T temp = A[i][k] * A_RMS_cd * gamma[k][i];
            if(enble_beta){
                T temp_beta = temp + beta[k][i];
                outp_pack[i] = temp_beta;
            }
            else{
                outp_pack[i] = temp;
            }
        }
        output_stream.write(outp_pack);
        if(k == 0) cout << "LN output data: ";
        if(k < 16) cout << outp_pack[0] << " ";
        if(k == 15) cout << endl;
    }


    // // versal FPGA (v80)
    // T A[block_parallel][max_hidden_dim/block_parallel];
    // #pragma HLS ARRAY_PARTITION variable=A dim=1 complete
    // T A_square_sum[block_parallel];
    // #pragma HLS ARRAY_PARTITION variable=A_square_sum complete

    // init_sum_loop: for (int i = 0; i < block_parallel; i++) {
    // #pragma HLS unroll
    //     A_square_sum[i] = 0;
    // }

    // in_buf_loop: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
    // #pragma HLS pipeline II=1
    //     hls::vector<T, block_parallel> temp_pack = input_stream.read();
    //     for(int i = 0; i < block_parallel; i++){
    //         T temp = temp_pack[i];
    //         A[i][k] = temp;
    //         A_square_sum[i] += temp * temp;
    //     }
    // }

    // T A_square_sum_total = 0;
    // sum_loop: for (int i = 0; i < block_parallel; i++) {
    //     A_square_sum_total += A_square_sum[i];
    // }

    // T A_RMS_cd = 1.0f / sqrt(A_square_sum_total / io_hidden_dim + eps);

    // output_scale_loop: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
    // #pragma HLS pipeline II=1
    //     hls::vector<T, block_parallel> outp_pack;
    //     for(int i = 0; i < block_parallel; i++){
    //         T temp = A[i][k] * A_RMS_cd * gamma[k][i];
    //         if(enble_beta){
    //             T temp_beta = temp + beta[k][i];
    //             outp_pack[i] = temp_beta;
    //         }
    //         else{
    //             outp_pack[i] = temp;
    //         }
    //     }
    //     output_stream.write(outp_pack);
    // }
}

#endif