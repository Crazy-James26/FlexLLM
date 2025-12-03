#ifndef _SWISH_H
#define _SWISH_H
#include "config.h"

template <typename T, int io_parallel, int max_hidden_dim=INTER_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_Swish(
    tapa::istream<hls::vector<T, io_parallel>>& input_stream,
    tapa::ostream<hls::vector<T, io_parallel>>& output_stream,
    int seq_len = max_seq_len,
    int io_hidden_dim = max_hidden_dim
){
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        in_buf_loop: for (int k = 0; k < io_hidden_dim; k++) {
        #pragma HLS pipeline II=1
            hls::vector<T, io_parallel> temp_pack = input_stream.read();
            hls::vector<T, io_parallel> out_pack;
            for(int i = 0; i < io_parallel; i++){
                out_pack[i] = temp_pack[i] / (1 + exp(-temp_pack[i]));
            }
            output_stream.write(out_pack);
        }
    }
}

template <typename T, int block_parallel, int max_hidden_dim=INTER_DIM>
void dec_Swish(
    tapa::istream<hls::vector<T, block_parallel>>& input_stream,
    tapa::ostream<hls::vector<T, block_parallel>>& output_stream,
    int io_hidden_dim = max_hidden_dim
){
    in_buf_loop: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
    #pragma HLS pipeline II=1
        hls::vector<T, block_parallel> temp_pack = input_stream.read();
        hls::vector<T, block_parallel> out_pack;
        for(int i = 0; i < block_parallel; i++){
           out_pack[i] = temp_pack[i] / (1 + exp(-temp_pack[i]));
        }
        output_stream.write(out_pack);
    }
}

#endif