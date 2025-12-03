#ifndef _MHA_Flatten_H_
#define _MHA_Flatten_H_
#include "config.h"
#include "PE.h"
#include "data_io.h"


template <int block_size_b, int max_log2_k_size = log2_HIDDEN_DIM, bool is_uint_A = false>
void systolic_array_i8xi8_pack_1x2_flatten_1D(
    tapa::istream<ap_int<8>>& A_loader,
    tapa::istream<hls::vector<ap_int<8>, block_size_b>>& B_loader, 
    tapa::ostream<ap_int<max_log2_k_size + 16>>& C_drainer,
    int k_size
  ) {
    hls::stream<ap_int<8>> A_fifo[block_size_b/2 + 1];
    #pragma HLS STREAM variable=A_fifo depth=4
    #pragma HLS BIND_STORAGE variable=A_fifo type=fifo impl=srl

    hls::stream<ap_uint<16>> B_fifo[block_size_b/2];
    #pragma HLS STREAM variable=B_fifo depth=block_size_b/2 + 1
    #pragma HLS BIND_STORAGE variable=B_fifo type=fifo impl=srl
  
    hls::stream<ap_int<max_log2_k_size + 16>> C_fifo[block_size_b/2];
    #pragma HLS STREAM variable=C_fifo depth=block_size_b/2
    #pragma HLS BIND_STORAGE variable=A_fifo type=fifo impl=srl
  
    #pragma HLS DATAFLOW
    data_load_AB:for (int k = 0; k < k_size; k++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=1 max=(1<<max_log2_k_size)
        ap_int<8> A_temp = A_loader.read();
        hls::vector<ap_int<8>, block_size_b> B_temp = B_loader.read();
  
        A_fifo[0].write(A_temp);
        
        for (int n = 0; n < block_size_b/2; n++) {
            B_fifo[n].write(ap_uint<16>((B_temp[2*n + 1], B_temp[2*n])));
        }
    }
    
    for (int n = 0; n < block_size_b/2; n++) {
    #pragma HLS UNROLL
        if(n == block_size_b/2 - 1)
            PE_i8xi8_pack_1x2_1xDSP_1D<is_uint_A, max_log2_k_size, true>(A_fifo[n], A_fifo[n+1], B_fifo[n], C_fifo[n], k_size);
        else
            PE_i8xi8_pack_1x2_1xDSP_1D<is_uint_A, max_log2_k_size>(A_fifo[n], A_fifo[n+1], B_fifo[n], C_fifo[n], k_size);
    }
    
  
    data_drain_C: for (int n = 0; n < block_size_b/2; n++) {
    #pragma HLS PIPELINE II=2
        C_drainer.write(C_fifo[n].read());
        C_drainer.write(C_fifo[n].read());
    }
}


template <typename T, int head_parallel, int K_parallel, int mha_head_num, int mha_head_dim = HEAD_DIM, int max_sum_seq_len=MAX_SUM_SEQ_LEN>
void dec_MHA_qxk_input_broadcastor_template(
    tapa::istream<hls::vector<T, head_parallel>>& input_seq,
    tapa::ostreams<T, head_parallel>& input_loaders,
    int seq_len
){
    hls::vector<T, head_parallel> A[mha_head_num/head_parallel * mha_head_dim];

    in_buf_loop: for (int k = 0; k < mha_head_num/head_parallel * mha_head_dim; k++) {    // L19
    #pragma HLS pipeline II=1
        A[k] = input_seq.read();
    }

    attn_head_loop: for (int H = 0; H < mha_head_num/head_parallel; H++){
        k_weight_block_loop: for(int N = 0; N < (seq_len + K_parallel - 1) / K_parallel; N++){
        #pragma HLS loop_tripcount min=1 max=max_sum_seq_len/K_parallel
            init_block_AB: for(int k = 0; k < mha_head_dim; k++){
            #pragma HLS PIPELINE II=1
                for (int i = 0; i < head_parallel; i++) {
                    input_loaders[i].write(A[H * mha_head_dim + k][i]);
                }
            }
        }
    }
}


template <int head_parallel, int K_parallel, int mha_head_num, int mha_head_dim = HEAD_DIM, int log2_mha_head_dim = log2_HEAD_DIM, int max_sum_seq_len=MAX_SUM_SEQ_LEN, bool is_uint_input=false>
void dec_MHA_i8xi8_qxk_flatten_template(
    tapa::istream<ap_int<8>>& input_loader,
    tapa::istream<hls::vector<ap_int<8>, K_parallel>>& weight_loader,
    tapa::ostream<ap_int<log2_mha_head_dim + 16>>& output_drainer,
    int seq_len
){
    attn_head_loop: for (int H = 0; H < mha_head_num/head_parallel; H++){
        k_weight_block_loop: for(int N = 0; N < (seq_len + K_parallel - 1) / K_parallel; N++){
        #pragma HLS loop_tripcount min=1 max=max_sum_seq_len/K_parallel
            systolic_array_i8xi8_pack_1x2_flatten_1D<K_parallel, log2_mha_head_dim, is_uint_input>(
                input_loader, weight_loader, output_drainer, mha_head_dim
            );
        }
    }
}

template <typename T, int head_parallel, int K_parallel, int mha_head_num, int max_sum_seq_len=MAX_SUM_SEQ_LEN>
void dec_MHA_qxk_output_merger_template(
    tapa::istreams<T, head_parallel>& output_drainers,
    tapa::ostream<hls::vector<T, head_parallel>>& output_seq,
    int seq_len
){
    attn_head_loop: for (int H = 0; H < mha_head_num/head_parallel; H++){
        k_weight_block_loop: for(int N = 0; N < (seq_len + K_parallel - 1) / K_parallel; N++){
        #pragma HLS loop_tripcount min=1 max=max_sum_seq_len/K_parallel
        output_scale_loop: for (int n = 0; n < K_parallel; n++) {    // L41
            #pragma HLS pipeline II=1
                hls::vector<T, head_parallel> outp_pack;
                for (int i = 0; i < head_parallel; i++) {
                    outp_pack[i] = output_drainers[i].read();
                }
                output_seq.write(outp_pack);
            }
        }
    }
}


template <typename T, int head_parallel, int V_parallel, int mha_head_num, int mha_head_dim = HEAD_DIM, int max_sum_seq_len = MAX_SUM_SEQ_LEN>
void dec_MHA_axv_input_broadcastor_template(
    tapa::istream<hls::vector<T, head_parallel>>& input_seq,
    tapa::ostreams<T, head_parallel>& input_loaders,
    int seq_len
){
    T A[head_parallel][max_sum_seq_len];
    #pragma HLS ARRAY_PARTITION variable=A complete dim=1

    attn_head_loop: for (int H = 0; H < mha_head_num/head_parallel; H++){
        in_buf_loop: for (int k = 0; k < seq_len; k++) {    // L19
        #pragma HLS loop_tripcount min=1 max=max_sum_seq_len
        #pragma HLS pipeline II=1
            hls::vector<T, head_parallel> A_pack = input_seq.read();
            for (int i = 0; i < head_parallel; i++) {
                A[i][k] = A_pack[i];
            }
        }

        v_weight_block_loop: for(int N = 0; N < mha_head_dim/V_parallel; N++){
            init_block_AB: for(int k = 0; k < seq_len; k++){
            #pragma HLS loop_tripcount min=1 max=max_sum_seq_len
            #pragma HLS PIPELINE II=1
                for (int i = 0; i < head_parallel; i++) {
                    input_loaders[i].write(A[i][k]);
                }
            }
        }
    }
}



template <int head_parallel, int V_parallel, int mha_head_num, int mha_head_dim = HEAD_DIM, int max_sum_seq_len = MAX_SUM_SEQ_LEN, int log2_max_sum_seq_len = log2_MAX_SUM_SEQ_LEN, bool is_uint_input=false>
void dec_MHA_i8xi8_axv_flatten_template(
    tapa::istream<ap_int<8>>& input_loader,
    tapa::istream<hls::vector<ap_int<8>, V_parallel>>& weight_loader,
    tapa::ostream<ap_int<log2_max_sum_seq_len + 16>>& output_drainer,
    int seq_len
){
    attn_head_loop: for (int H = 0; H < mha_head_num/head_parallel; H++){
        v_weight_block_loop: for(int N = 0; N < mha_head_dim/V_parallel; N++){
            systolic_array_i8xi8_pack_1x2_flatten_1D<V_parallel, log2_max_sum_seq_len, is_uint_input>(
                input_loader, weight_loader, output_drainer, seq_len
            );
        }
    }
}



template <typename T, int head_parallel, int V_parallel, int mha_head_num, int mha_head_dim = HEAD_DIM>
void dec_MHA_axv_output_merger_template(
    tapa::istreams<T, head_parallel>& output_drainers,
    tapa::ostream<hls::vector<T, head_parallel>>& output_seq
){
    attn_head_loop: for (int H = 0; H < mha_head_num/head_parallel; H++){
        v_weight_block_loop: for(int N = 0; N < mha_head_dim/V_parallel; N++){
            output_scale_loop: for (int n = 0; n < V_parallel; n++) {    // L41
            #pragma HLS pipeline II=1
                hls::vector<T, head_parallel> outp_pack;
                for (int i = 0; i < head_parallel; i++) {
                    outp_pack[i] = output_drainers[i].read();
                }
                output_seq.write(outp_pack);
            }
        }
    }
}

#endif