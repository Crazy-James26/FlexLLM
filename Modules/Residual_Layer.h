#ifndef _RESIDUAL_H_
#define _RESIDUAL_H_
#include "config.h"


template <typename T, int io_parallel, int io_hidden_dim=HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_Residual_storer(
    tapa::istream<hls::vector<T, io_parallel>>& residual_stream,
    tapa::mmap<hls::vector<T, io_parallel>> residual_mmap, // residual_mmap[max_seq_len/io_parallel * io_hidden_dim]
    tapa::ostream<bool>& finish_stream,
    int block_id = 0,
    int seq_len = max_seq_len
){
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++) {
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        int bias = block_id * max_seq_len/io_parallel * io_hidden_dim + M * io_hidden_dim;
        out_buf_loop: for (int k = 0; k < io_hidden_dim; k++) {
        #pragma HLS pipeline II=1
            hls::vector<T, io_parallel> out_pack = residual_stream.read();
            residual_mmap[bias + k] = out_pack;
        }
        finish_stream.write(true);
    }
}

template <typename T, int io_parallel, int io_hidden_dim=HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_Residual_loader(
    tapa::istream<bool>& ready_stream,
    tapa::mmap<hls::vector<T, io_parallel>> residual_mmap, // residual_mmap[max_seq_len/io_parallel * io_hidden_dim]
    tapa::ostream<hls::vector<T, io_parallel>>& residual_stream,
    int block_id = 0,
    int seq_len = max_seq_len
) {
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++) {
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        int bias = block_id * max_seq_len/io_parallel * io_hidden_dim + M * io_hidden_dim;
        bool ready = ready_stream.read();
        out_buf_loop: for (int k = 0; k < io_hidden_dim; k++) {
        #pragma HLS pipeline II=1
            hls::vector<T, io_parallel> out_pack = residual_mmap[bias + k];
            residual_stream.write(out_pack);
        }
    }
}

template <typename T, int io_parallel, int io_hidden_dim=HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_Residual_Layer(
    tapa::istream<hls::vector<T, io_parallel>>& input_stream,
    tapa::istream<hls::vector<T, io_parallel>>& residual_stream,
    tapa::ostream<hls::vector<T, io_parallel>>& output_stream,
    int seq_len = max_seq_len
) {
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++) {
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        out_buf_loop: for (int k = 0; k < io_hidden_dim; k++) {
        #pragma HLS pipeline II=1
            hls::vector<T, io_parallel> out_pack;
            hls::vector<T, io_parallel> input_pack = input_stream.read();
            hls::vector<T, io_parallel> residual_pack = residual_stream.read();
            for(int i = 0; i < io_parallel; i++){
                out_pack[i] = input_pack[i] + residual_pack[i];
            }
            output_stream.write(out_pack);
        }
    }
}


template <typename T, int block_parallel, int max_hidden_dim=HIDDEN_DIM>
void dec_Residual_Layer(
    tapa::istream<hls::vector<T, block_parallel>>& input_stream,
    tapa::istream<hls::vector<T, block_parallel>>& residual_stream,
    tapa::ostream<hls::vector<T, block_parallel>>& output_stream,
    int io_hidden_dim = max_hidden_dim
) {
    residual_loop: for (int k = 0; k < io_hidden_dim/block_parallel; k++) {
    #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel
    #pragma HLS pipeline II=1
        hls::vector<T, block_parallel> out_pack;
        hls::vector<T, block_parallel> input_pack = input_stream.read();
        hls::vector<T, block_parallel> residual_pack = residual_stream.read();
        for(int i = 0; i < block_parallel; i++){
            out_pack[i] = input_pack[i] + residual_pack[i];
        }
        output_stream.write(out_pack);
    }
}





#endif