#ifndef _LL_H_
#define _LL_H_
#include "config.h"
#include "PE.h"
#include "data_io.h"


template <int block_size_a, int block_size_b, int max_log2_k_size = log2_HIDDEN_DIM, bool is_uint_A = false>
void systolic_array_i4xi4_pack_2x2_2D(
    hls::stream<hls::vector<ap_int<4>, block_size_a>>& A_loader,
    hls::stream<hls::vector<ap_int<4>, block_size_b>>& B_loader, 
    hls::stream<hls::vector<ap_int<max_log2_k_size + 8>, block_size_a>>& C_drainer,
    int k_size
  ) {
    hls::stream<ap_uint<8>> A_fifo[block_size_a/2][block_size_b/2 + 1];
    hls::stream<ap_uint<8>> B_fifo[block_size_b/2][block_size_a/2 + 1];
    #pragma HLS STREAM variable=A_fifo depth=block_size_a/2 + 1
    #pragma HLS BIND_STORAGE variable=A_fifo type=fifo impl=srl
    #pragma HLS STREAM variable=B_fifo depth=block_size_b/2 + 1
    #pragma HLS BIND_STORAGE variable=B_fifo type=fifo impl=srl
  
    hls::stream<ap_uint<2*max_log2_k_size + 16>> C_fifo[block_size_a/2][block_size_b/2];
    #pragma HLS STREAM variable=C_fifo depth=block_size_a/2 + 1
    #pragma HLS BIND_STORAGE variable=A_fifo type=fifo impl=srl
  
    #pragma HLS DATAFLOW
    data_load_AB:for (int k = 0; k < k_size; k++) {
    #pragma HLS PIPELINE II=1
        hls::vector<ap_int<4>, block_size_a> A_temp = A_loader.read();
        hls::vector<ap_int<4>, block_size_b> B_temp = B_loader.read();
  
        for (int m = 0; m < block_size_a/2; m++) {
            A_fifo[m][0].write(ap_uint<8>((A_temp[2*m + 1], A_temp[2*m])));
        }
        
        for (int n = 0; n < block_size_b/2; n++) {
            B_fifo[n][0].write(ap_uint<8>((B_temp[2*n + 1], B_temp[2*n])));
        }
    }
  
    systolic_array: for (int m = 0; m < block_size_a/2; m++) {
    #pragma HLS UNROLL
        for (int n = 0; n < block_size_b/2; n++) {
        #pragma HLS UNROLL
            if(m == block_size_a/2 - 1 && n == block_size_b/2 - 1)
                PE_i4xi4_pack_2x2_2D<is_uint_A, max_log2_k_size, true, true>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
                // PE_i4xi4_pack_2x2_1xDSP<is_uint_A, max_log2_k_size>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
            else if(m == block_size_a/2 - 1)
                PE_i4xi4_pack_2x2_2D<is_uint_A, max_log2_k_size, false, true>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
                // PE_i4xi4_pack_2x2_1xDSP<is_uint_A, max_log2_k_size>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
            else if(n == block_size_b/2 - 1)
                PE_i4xi4_pack_2x2_2D<is_uint_A, max_log2_k_size, true, false>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
                // PE_i4xi4_pack_2x2_1xDSP<is_uint_A, max_log2_k_size>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
            else
                PE_i4xi4_pack_2x2_2D<is_uint_A, max_log2_k_size, false, false>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
                // PE_i4xi4_pack_2x2_1xDSP<is_uint_A, max_log2_k_size>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
        }
    }
  
    // data_drain_AB:for (int k = 0; k < k_size; k++) {
    // #pragma HLS PIPELINE II=1
    //     for (int m = 0; m < block_size_a/2; m++) {
    //         A_fifo[m][block_size_b/2].read();
    //     }
    //     for (int n = 0; n < block_size_b/2; n++) {
    //         B_fifo[n][block_size_a/2].read();
    //     }
    // }
  
    data_drain_C: for (int n = 0; n < block_size_b/2; n++) {
    #pragma HLS PIPELINE II=2
        hls::vector<ap_int<max_log2_k_size + 8>, block_size_a> C_temp;
        for (int m = 0; m < block_size_a/2; m++) {
        #pragma HLS UNROLL
            (C_temp[2*m + 1], C_temp[2*m]) = C_fifo[m][n].read();
        }
        C_drainer.write(C_temp);
        for (int m = 0; m < block_size_a/2; m++) {
        #pragma HLS UNROLL
            (C_temp[2*m + 1], C_temp[2*m]) = C_fifo[m][n].read();
        }
        C_drainer.write(C_temp);
    }
}

template <typename T, int weight_parallel, int io_parallel, int max_input_dim = HIDDEN_DIM, int max_output_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_weight_loader(
    tapa::mmap<hls::vector<T, weight_parallel>> weight_mmap,
    tapa::ostream<hls::vector<T, weight_parallel>>& weight_stream,
    int block_id,
    int seq_len = max_seq_len,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim,
    int addr_bias = 0
){
    int bias = addr_bias + block_id * ((output_hidden_dim + weight_parallel - 1)/weight_parallel) * input_hidden_dim;
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
        read_weight_loop: for(int i = 0; i < ((output_hidden_dim + weight_parallel - 1)/weight_parallel)*input_hidden_dim; i++){
        #pragma HLS pipeline II=1
            weight_stream.write(weight_mmap[bias + i]);
        }
    }
}


template <int read_bit, int pack_num, int weight_parallel, int io_parallel, int max_input_dim = HIDDEN_DIM, int max_output_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_weight_loader_pack(
    tapa::mmap<hls::vector<ap_int<read_bit*pack_num>, weight_parallel/pack_num>> weight_mmap,
    tapa::ostream<hls::vector<ap_int<read_bit>, weight_parallel>>& weight_stream,
    int block_id,
    int seq_len = max_seq_len,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim,
    int addr_bias = 0
){
    int bias = addr_bias + block_id * ((output_hidden_dim + weight_parallel - 1)/weight_parallel) * input_hidden_dim;
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
        read_weight_loop: for(int i = 0; i < ((output_hidden_dim + weight_parallel - 1)/weight_parallel)*input_hidden_dim; i++){
        #pragma HLS pipeline II=1
            hls::vector<ap_int<read_bit*pack_num>, weight_parallel/pack_num> in_pack = weight_mmap[bias + i];
            hls::vector<ap_int<read_bit>, weight_parallel> out_pack;
            for(int j = 0; j < weight_parallel/pack_num; j++) {
            #pragma HLS UNROLL
                for(int k = 0; k < pack_num; k++) {
                    out_pack[j * pack_num + k] = in_pack[j].range((k+1)*read_bit-1, k*read_bit);
                }
            }
            weight_stream.write(out_pack);
            // weight_stream.write(weight_mmap[bias + i]);
        }
    }
}

template <int weight_parallel, int io_parallel, int max_input_dim = HIDDEN_DIM, int max_output_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_weight_loader_int4_pack_2(
    tapa::mmap<hls::vector<ap_int<8>, weight_parallel/2>> weight_mmap,
    tapa::ostream<hls::vector<ap_int<4>, weight_parallel>>& weight_stream,
    int block_id,
    int seq_len = max_seq_len,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim,
    int addr_bias = 0
){
    int bias = addr_bias + block_id * ((output_hidden_dim + weight_parallel - 1)/weight_parallel) * input_hidden_dim;
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
        read_weight_loop: for(int i = 0; i < ((output_hidden_dim + weight_parallel - 1)/weight_parallel)*input_hidden_dim; i++){
        #pragma HLS pipeline II=1
            hls::vector<ap_int<8>, weight_parallel/2> in_pack = weight_mmap[bias + i];
            hls::vector<ap_int<4>, weight_parallel> data_pack;
            for(int j = 0; j < weight_parallel/2; j++) {
            #pragma HLS UNROLL
                data_pack[j * 2] = in_pack[j].range(3, 0);
                data_pack[j * 2 + 1] = in_pack[j].range(7, 4);
            }
            weight_stream.write(data_pack);
        }
    }
}


template <int weight_parallel_read, int weight_parallel_write, int io_parallel, int max_input_dim = HIDDEN_DIM, int max_output_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_weight_loader_int4_pack_2_discard(
    tapa::mmap<hls::vector<ap_int<8>, weight_parallel_read/2>> weight_mmap,
    tapa::ostream<hls::vector<ap_int<4>, weight_parallel_write>>& weight_stream,
    int block_id,
    int seq_len = max_seq_len,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim,
    int addr_bias = 0
){
    int bias = addr_bias + block_id * ((output_hidden_dim + weight_parallel_write - 1)/weight_parallel_write) * input_hidden_dim;
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
        read_weight_loop: for(int i = 0; i < ((output_hidden_dim + weight_parallel_write - 1)/weight_parallel_write)*input_hidden_dim; i++){
        #pragma HLS pipeline II=1
            hls::vector<ap_int<8>, weight_parallel_read/2> in_pack = weight_mmap[bias + i];
            hls::vector<ap_int<4>, weight_parallel_write> data_pack;
            for(int j = 0; j < weight_parallel_write/2; j++) {
            #pragma HLS UNROLL
                data_pack[j * 2] = in_pack[j].range(3, 0);
                data_pack[j * 2 + 1] = in_pack[j].range(7, 4);
            }
            weight_stream.write(data_pack);
        }
    }
}


template <typename T, int weight_parallel_read, int weight_parallel_write, int io_parallel, int max_input_dim = HIDDEN_DIM, int max_output_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_weight_loader_discard(
    tapa::mmap<hls::vector<T, weight_parallel_read>> weight_mmap,
    tapa::ostream<hls::vector<T, weight_parallel_write>>& weight_stream,
    int block_id,
    int seq_len = max_seq_len,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim,
    int addr_bias = 0
){
    int bias = addr_bias + block_id * ((output_hidden_dim + weight_parallel_write - 1)/weight_parallel_write)*input_hidden_dim;
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
        read_weight_loop: for(int i = 0; i < ((output_hidden_dim + weight_parallel_write - 1)/weight_parallel_write)*input_hidden_dim; i++){
        #pragma HLS pipeline II=1
            hls::vector<T, weight_parallel_read> temp_pack = weight_mmap[bias + i];
            hls::vector<T, weight_parallel_write> weight_pack;
            for (int j = 0; j < weight_parallel_write; j++){
            #pragma HLS UNROLL
                weight_pack[j] = temp_pack[j];
            }
            weight_stream.write(weight_pack);
        }
    }
}


template <typename T, int weight_parallel, int io_parallel, int max_input_dim = HIDDEN_DIM, int max_output_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_weight_loader_buffer(
    tapa::istream<hls::vector<T, weight_parallel>>& weight_stream_in,
    tapa::ostream<hls::vector<T, weight_parallel>>& weight_stream_out,
    int block_id,
    int seq_len = max_seq_len,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim
){
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
        read_weight_loop: for(int i = 0; i < ((output_hidden_dim + weight_parallel - 1)/weight_parallel)*input_hidden_dim; i++){
        #pragma HLS pipeline II=1
            weight_stream_out.write(weight_stream_in.read());
        }
    }
}


template <typename T, int weight_parallel, int weight_port_merge_num, int io_parallel, int max_input_dim = HIDDEN_DIM, int max_output_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_weight_stream_merger(
    tapa::istreams<hls::vector<T, weight_parallel>, weight_port_merge_num>& weight_streams,
    tapa::ostream<hls::vector<T, weight_parallel * weight_port_merge_num>>& weight_stream_merged,
    int seq_len = max_seq_len,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim
){
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
        read_weight_loop: for(int k = 0; k < ((output_hidden_dim + weight_parallel - 1)/weight_parallel)*input_hidden_dim; k++){
        #pragma HLS pipeline II=1
            hls::vector<T, weight_parallel * weight_port_merge_num> weight_pack;
            for (int i = 0; i < weight_port_merge_num; i++){
            #pragma HLS UNROLL
                hls::vector<T, weight_parallel> temp_pack = weight_streams[i].read();
                for(int j = 0; j < weight_parallel; j++){
                #pragma HLS UNROLL
                    weight_pack[i * weight_parallel + j] = temp_pack[j];
                }
            }
            weight_stream_merged.write(weight_pack);
        }
    }
}

template <int io_parallel, bool is_act_asym=false, int max_output_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_weight_s_loader_fp32(
    tapa::mmap<hls::vector<float, 1+is_act_asym>> weight_s_sum_mmap,
    tapa::ostream<hls::vector<float, 1+is_act_asym>>& weight_s_sum_stream,
    int block_id,
    int seq_len = max_seq_len,
    int output_hidden_dim = max_output_dim,
    int addr_bias = 0
){
    int bias = addr_bias + block_id * output_hidden_dim;
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
        weight_sum_loop: for(int i = 0; i < output_hidden_dim; i++){
        #pragma HLS pipeline II=1
            hls::vector<float, 1+is_act_asym> w_pack = weight_s_sum_mmap[bias + i];
            weight_s_sum_stream.write(w_pack);
        }
    }
}


template <int io_parallel, bool is_act_asym=false, int max_output_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_weight_s_loader_buffer_fp32(
    tapa::istream<hls::vector<float, 1+is_act_asym>>& weight_s_sum_stream_in,
    tapa::ostream<hls::vector<float, 1+is_act_asym>>& weight_s_sum_stream_out,
    int block_id,
    int seq_len = max_seq_len,
    int output_hidden_dim = max_output_dim
){
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
        weight_sum_loop: for(int i = 0; i < output_hidden_dim; i++){
        #pragma HLS pipeline II=1
            weight_s_sum_stream_out.write(weight_s_sum_stream_in.read());
        }
    }
}

template <int io_parallel, int weight_parallel, int max_input_dim = HIDDEN_DIM, int max_log2_input_dim = log2_HIDDEN_DIM, int max_output_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN, bool is_uint_input=false>
void pref_Linear_Layer_i4xi4(
    tapa::istream<hls::vector<ap_int<4>, io_parallel>>& input_seq,
    tapa::istream<hls::vector<ap_int<4>, weight_parallel>>& weight_loader,
    tapa::ostream<hls::vector<ap_int<max_log2_input_dim + 8>, io_parallel>>& output_seq,
    int seq_len = max_seq_len,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim
){
    hls::vector<ap_int<4>, io_parallel> A[max_input_dim];
    #pragma HLS bind_storage variable=A type=ram_2p impl=uram
      
    hls::stream<hls::vector<ap_int<4>, io_parallel>> block_A_loader;
    #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

    hls::stream<hls::vector<ap_int<4>, weight_parallel>> block_B_loader;
    #pragma HLS BIND_STORAGE variable=block_B_loader type=fifo impl=srl

    hls::stream<hls::vector<ap_int<max_log2_input_dim + 8>, io_parallel>> block_C_drainer;
    #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl

    
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel 
        in_buf_loop: for (int k = 0; k < input_hidden_dim; k++) {    // L19
        #pragma HLS pipeline II=1
            A[k] = input_seq.read();
        }

        weight_block_loop: for(int N = 0; N < ((output_hidden_dim + weight_parallel - 1)/weight_parallel); N++){
        #pragma HLS DATAFLOW
            init_block_AB: for(int k = 0; k < input_hidden_dim; k++){
            #pragma HLS PIPELINE II=1
                block_A_loader.write(A[k]);
                block_B_loader.write(weight_loader.read());
            }
            // if acitivation is asymmetric quantized, then input datatype is ap_uint<4>
            // else input datatype is ap_int<4>
            systolic_array_i4xi4_pack_2x2_2D<io_parallel, weight_parallel, max_log2_input_dim, is_uint_input>(
                block_A_loader, block_B_loader, block_C_drainer, input_hidden_dim
            );

            bias_loop: for (int n = 0; n < weight_parallel; n++) {    // L41
            #pragma HLS pipeline II=1
                output_seq.write(block_C_drainer.read());
            }
        }
    }
}


template <int io_parallel, int weight_parallel, int block_num, int max_input_dim = HIDDEN_DIM, int max_log2_input_dim = log2_HIDDEN_DIM, int max_output_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN, bool is_uint_input=false>
void pref_Linear_Layer_i4xi4_blocked(
    tapa::istream<hls::vector<ap_int<4>, io_parallel>>& input_seq,
    tapa::istreams<hls::vector<ap_int<4>, weight_parallel/block_num>, block_num>& weight_loaders,
    tapa::ostream<hls::vector<ap_int<max_log2_input_dim + 8>, io_parallel>>& output_seq,
    int seq_len = max_seq_len,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim
){
    hls::vector<ap_int<4>, io_parallel> A[max_input_dim];
    #pragma HLS bind_storage variable=A type=ram_2p impl=uram
      
    hls::stream<hls::vector<ap_int<4>, io_parallel>> block_A_loader[block_num];
    #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

    hls::stream<hls::vector<ap_int<4>, weight_parallel/block_num>> block_B_loader[block_num];
    #pragma HLS BIND_STORAGE variable=block_B_loader type=fifo impl=srl

    hls::stream<hls::vector<ap_int<max_log2_input_dim + 8>, io_parallel>> block_C_drainer[block_num];
    #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl

    
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel 
        in_buf_loop: for (int k = 0; k < input_hidden_dim; k++) {    // L19
        #pragma HLS pipeline II=1
            auto temp_pack = input_seq.read();
            A[k] = temp_pack;
        }

        weight_block_loop: for(int N = 0; N < ((output_hidden_dim + weight_parallel - 1)/weight_parallel); N++){
        #pragma HLS DATAFLOW
            
            init_block_AB: for(int k = 0; k < input_hidden_dim; k++){
            #pragma HLS PIPELINE II=1
                for(int i = 0; i < block_num; i++){
                #pragma HLS UNROLL
                    block_A_loader[i].write(A[k]);
                    block_B_loader[i].write(weight_loaders[i].read());
                }
            }
            // if acitivation is asymmetric quantized, then input datatype is ap_uint<4>
            // else input datatype is ap_int<4>

            for(int i = 0; i < block_num; i++){
            #pragma HLS UNROLL
                systolic_array_i4xi4_pack_2x2_2D<io_parallel, weight_parallel/block_num, max_log2_input_dim, is_uint_input>(
                    block_A_loader[i], block_B_loader[i], block_C_drainer[i], input_hidden_dim
                );
            }

            
            bias_loop: for (int n = 0; n < weight_parallel/2/block_num; n++) {    // L41
                block_output_loop: for(int i = 0; i < block_num; i++){
                    double_loop: for(int j = 0; j < 2; j++){
                    #pragma HLS pipeline II=1
                        output_seq.write(block_C_drainer[i].read());
                    }
                }
            }
        }
    }
}


template <int block_size_a, int block_size_b>
void systolic_array_fp32xfp32_2D(
    hls::stream<hls::vector<float, block_size_a>>& A_loader,
    hls::stream<hls::vector<float, block_size_b>>& B_loader, 
    hls::stream<hls::vector<float, block_size_a>>& C_drainer,
    int k_size
  ) {
    hls::stream<float> A_fifo[block_size_a][block_size_b + 1];
    hls::stream<float> B_fifo[block_size_b][block_size_a + 1];
    #pragma HLS STREAM variable=A_fifo depth=block_size_a + 1
    #pragma HLS BIND_STORAGE variable=A_fifo type=fifo impl=srl
    #pragma HLS STREAM variable=B_fifo depth=block_size_b + 1
    #pragma HLS BIND_STORAGE variable=B_fifo type=fifo impl=srl
  
    hls::stream<float> C_fifo[block_size_a][block_size_b];
    #pragma HLS STREAM variable=C_fifo depth=block_size_a + 1
    #pragma HLS BIND_STORAGE variable=A_fifo type=fifo impl=srl
  
    #pragma HLS DATAFLOW
    data_load_AB:for (int k = 0; k < k_size; k++) {
    #pragma HLS PIPELINE II=1
        hls::vector<float, block_size_a> A_temp = A_loader.read();
        hls::vector<float, block_size_b> B_temp = B_loader.read();
  
        for (int m = 0; m < block_size_a; m++) {
            A_fifo[m][0].write(A_temp[m]);
        }
        
        for (int n = 0; n < block_size_b; n++) {
            B_fifo[n][0].write(B_temp[n]);
        }
    }
  
    systolic_array: for (int m = 0; m < block_size_a; m++) {
    #pragma HLS UNROLL
        for (int n = 0; n < block_size_b; n++) {
        #pragma HLS UNROLL
            if(m == block_size_a - 1 && n == block_size_b - 1)
                PE_fp32xfp32_2D<true, true>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
            else if(m == block_size_a - 1)
                PE_fp32xfp32_2D<false, true>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
            else if(n == block_size_b - 1)
                PE_fp32xfp32_2D<true, false>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
            else
                PE_fp32xfp32_2D<false, false>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
        }
    }
  
    // data_drain_AB:for (int k = 0; k < k_size; k++) {
    // #pragma HLS PIPELINE II=1
    //     for (int m = 0; m < block_size_a; m++) {
    //         A_fifo[m][block_size_b].read();
    //     }
    //     for (int n = 0; n < block_size_b; n++) {
    //         B_fifo[n][block_size_a].read();
    //     }
    // }
  
    data_drain_C: for (int n = 0; n < block_size_b; n++) {
    #pragma HLS PIPELINE II=1
        hls::vector<float, block_size_a> C_temp;
        for (int m = 0; m < block_size_a; m++) {
        #pragma HLS UNROLL
            C_temp[m] = C_fifo[m][n].read();
        }
        C_drainer.write(C_temp);
    }
}

template <int block_size_a, int block_size_b>
void systolic_array_fp32xfp32_pack_2x2_2D(
    hls::stream<hls::vector<float, block_size_a>>& A_loader,
    hls::stream<hls::vector<float, block_size_b>>& B_loader, 
    hls::stream<hls::vector<float, block_size_a>>& C_drainer,
    int k_size
  ) {
    hls::stream<ap_uint<64>> A_fifo[block_size_a/2][block_size_b/2 + 1];
    hls::stream<ap_uint<64>> B_fifo[block_size_b/2][block_size_a/2 + 1];
    #pragma HLS STREAM variable=A_fifo depth=block_size_a/2 + 1
    #pragma HLS BIND_STORAGE variable=A_fifo type=fifo impl=srl
    #pragma HLS STREAM variable=B_fifo depth=block_size_a/2 + 1
    #pragma HLS BIND_STORAGE variable=B_fifo type=fifo impl=srl
  
    hls::stream<ap_uint<64>> C_fifo[block_size_a/2][block_size_b/2];
    #pragma HLS STREAM variable=C_fifo depth=block_size_a/2 + 1
    #pragma HLS BIND_STORAGE variable=A_fifo type=fifo impl=srl
  
    #pragma HLS DATAFLOW
    data_load_AB:for (int k = 0; k < k_size; k++) {
    #pragma HLS PIPELINE II=1
        hls::vector<float, block_size_a> A_temp = A_loader.read();
        hls::vector<float, block_size_b> B_temp = B_loader.read();
  
        for (int m = 0; m < block_size_a/2; m++) {
            A_fifo[m][0].write((*(ap_uint<32>*)(&A_temp[2*m + 1]), *(ap_uint<32>*)(&A_temp[2*m])));
        }
        
        for (int n = 0; n < block_size_b/2; n++) {
            B_fifo[n][0].write((*(ap_uint<32>*)(&B_temp[2*n + 1]), *(ap_uint<32>*)(&B_temp[2*n])));
        }
    }
  
    systolic_array: for (int m = 0; m < block_size_a/2; m++) {
    #pragma HLS UNROLL
        for (int n = 0; n < block_size_b/2; n++) {
        #pragma HLS UNROLL
            if(m == block_size_a/2 - 1 && n == block_size_b/2 - 1)
                PE_fp32xfp32_pack_2x2_2D<true, true>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
            else if(m == block_size_a/2 - 1)
                PE_fp32xfp32_pack_2x2_2D<false, true>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
            else if(n == block_size_b/2 - 1)
                PE_fp32xfp32_pack_2x2_2D<true, false>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
            else
                PE_fp32xfp32_pack_2x2_2D<false, false>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
        }
    }

    // data_drain_AB:for (int k = 0; k < k_size; k++) {
    // #pragma HLS PIPELINE II=1
    //     for (int m = 0; m < block_size_a/2; m++) {
    //         A_fifo[m][block_size_b/2].read();
    //     }
    //     for (int n = 0; n < block_size_b/2; n++) {
    //         B_fifo[n][block_size_a/2].read();
    //     }
    // }
  
    data_drain_C: for (int n = 0; n < block_size_b/2; n++) {
    #pragma HLS PIPELINE II=2
        hls::vector<float, block_size_a> C_temp;
        ap_uint<32> C_temp_0, C_temp_1;
        for (int m = 0; m < block_size_a/2; m++) {
        #pragma HLS UNROLL
            (C_temp_1, C_temp_0) = C_fifo[m][n].read();
            C_temp[2*m + 1] = *(float*)(&C_temp_1);
            C_temp[2*m] = *(float*)(&C_temp_0);
        }
        C_drainer.write(C_temp);
        for (int m = 0; m < block_size_a/2; m++) {
        #pragma HLS UNROLL
            (C_temp_1, C_temp_0) = C_fifo[m][n].read();
            C_temp[2*m + 1] = *(float*)(&C_temp_1);
            C_temp[2*m] = *(float*)(&C_temp_0);
        }
        C_drainer.write(C_temp);
    }
}



template <int io_parallel, int weight_parallel, int max_input_dim = HIDDEN_DIM, int max_output_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_Linear_Layer_fp32xfp32(
    tapa::istream<hls::vector<float, io_parallel>>& input_seq,
    tapa::istream<hls::vector<float, weight_parallel>>& weight_loader,
    tapa::ostream<hls::vector<float, io_parallel>>& output_seq,
    int seq_len = max_seq_len,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim
){
    hls::vector<float, io_parallel> A[max_input_dim];
    #pragma HLS bind_storage variable=A type=ram_2p impl=uram
      
    hls::stream<hls::vector<float, io_parallel>> block_A_loader;
    #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

    hls::stream<hls::vector<float, weight_parallel>> block_B_loader;
    #pragma HLS BIND_STORAGE variable=block_B_loader type=fifo impl=srl

    hls::stream<hls::vector<float, io_parallel>> block_C_drainer;
    #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl

    
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel 
        in_buf_loop: for (int k = 0; k < input_hidden_dim; k++) {    // L19
        #pragma HLS pipeline II=1
            A[k] = input_seq.read();
        }

        weight_block_loop: for(int N = 0; N < ((output_hidden_dim + weight_parallel - 1)/weight_parallel); N++){
        #pragma HLS DATAFLOW
            init_block_AB: for(int k = 0; k < input_hidden_dim; k++){
            #pragma HLS PIPELINE II=1
                block_A_loader.write(A[k]);
                block_B_loader.write(weight_loader.read());
            }
            
            // systolic_array_fp32xfp32<io_parallel, weight_parallel>(
            //     block_A_loader, block_B_loader, block_C_drainer, input_hidden_dim
            // );
            systolic_array_fp32xfp32_pack_2x2_2D<io_parallel, weight_parallel>(
                block_A_loader, block_B_loader, block_C_drainer, input_hidden_dim
            );

            bias_loop: for (int n = 0; n < weight_parallel; n++) {    // L41
            #pragma HLS pipeline II=1
                output_seq.write(block_C_drainer.read());
            }
        }
    }
}


template <int io_parallel, int io_hidden_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_Gate_Layer_fp32xfp32(
    tapa::istream<hls::vector<float, io_parallel>>& input_seq,
    tapa::istream<hls::vector<float, io_parallel>>& gate_seq,
    tapa::ostream<hls::vector<float, io_parallel>>& output_seq,
    int seq_len = max_seq_len
){
    io_block_loop: for (int M = 0; M < seq_len/io_parallel * io_hidden_dim; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel * io_hidden_dim
    #pragma HLS pipeline II=1
        hls::vector<float, io_parallel> input_pack = input_seq.read();
        hls::vector<float, io_parallel> gate_pack = gate_seq.read();
        hls::vector<float, io_parallel> output_pack;
        for(int i = 0; i < io_parallel; i++){
            output_pack[i] = input_pack[i] * gate_pack[i];
        }
        output_seq.write(output_pack);
    }
}

template <int block_size_b, int max_log2_k_size = log2_HIDDEN_DIM, bool is_uint_A = false>
void systolic_array_i4xi4_pack_1x2_1D(
    hls::stream<ap_int<4>>& A_loader,
    hls::stream<hls::vector<ap_int<4>, block_size_b>>& B_loader, 
    hls::stream<ap_int<max_log2_k_size + 8>>& C_drainer,
    int k_size
  ) {
    hls::stream<ap_int<4>> A_fifo[block_size_b/2 + 1];
    #pragma HLS STREAM variable=A_fifo depth=4
    #pragma HLS BIND_STORAGE variable=A_fifo type=fifo impl=srl

    hls::stream<ap_uint<8>> B_fifo[block_size_b/2];
    #pragma HLS STREAM variable=B_fifo depth=block_size_b/2 + 1
    #pragma HLS BIND_STORAGE variable=B_fifo type=fifo impl=srl
  
    hls::stream<ap_int<max_log2_k_size + 8>> C_fifo[block_size_b/2];
    #pragma HLS STREAM variable=C_fifo depth=block_size_b/2
    #pragma HLS BIND_STORAGE variable=A_fifo type=fifo impl=srl
  
    #pragma HLS DATAFLOW
    data_load_AB:for (int k = 0; k < k_size; k++) {
    #pragma HLS PIPELINE II=1
        ap_int<4> A_temp = A_loader.read();
        hls::vector<ap_int<4>, block_size_b> B_temp = B_loader.read();
  
        A_fifo[0].write(A_temp);
        
        for (int n = 0; n < block_size_b/2; n++) {
            B_fifo[n].write(ap_uint<8>((B_temp[2*n + 1], B_temp[2*n])));
        }
    }
  
    for (int n = 0; n < block_size_b/2; n++) {
    #pragma HLS UNROLL
        if(n == block_size_b/2 - 1)
            PE_i4xi4_pack_1x2_1D<is_uint_A, max_log2_k_size, true>(A_fifo[n], A_fifo[n+1], B_fifo[n], C_fifo[n], k_size);
        else
            PE_i4xi4_pack_1x2_1D<is_uint_A, max_log2_k_size>(A_fifo[n], A_fifo[n+1], B_fifo[n], C_fifo[n], k_size);
    }
  
    data_drain_C: for (int n = 0; n < block_size_b/2; n++) {
    #pragma HLS PIPELINE II=2
        C_drainer.write(C_fifo[n].read());
        C_drainer.write(C_fifo[n].read());
    }
}


template <typename T, int block_parallel, int weight_parallel, int max_input_dim = HIDDEN_DIM, int max_output_dim = HIDDEN_DIM>
void dec_weight_loader(
    tapa::mmap<hls::vector<T, weight_parallel>> weight_mmap,
    tapa::ostream<hls::vector<T, weight_parallel>>& weight_stream,
    int block_id,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim,
    int addr_bias = 0
){
    int bias = addr_bias + block_id * output_hidden_dim/(block_parallel*weight_parallel)*input_hidden_dim;
    read_weight_loop: for(int i = 0; i < output_hidden_dim/(block_parallel*weight_parallel)*input_hidden_dim; i++) {
    #pragma HLS pipeline II=1
        hls::vector<T, weight_parallel> in_pack = weight_mmap[bias + i];
        weight_stream.write(in_pack);
    }
}

template <int read_bit, int pack_num, int block_parallel, int weight_parallel, int max_input_dim = HIDDEN_DIM, int max_output_dim = HIDDEN_DIM>
void dec_weight_loader_pack(
    tapa::mmap<hls::vector<ap_int<read_bit*pack_num>, weight_parallel/pack_num>> weight_mmap,
    tapa::ostream<hls::vector<ap_int<read_bit>, weight_parallel>>& weight_stream,
    int block_id,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim,
    int addr_bias = 0
){
    int bias = addr_bias + block_id * output_hidden_dim/(block_parallel*weight_parallel)*input_hidden_dim;
    read_weight_loop: for(int i = 0; i < output_hidden_dim/(block_parallel*weight_parallel)*input_hidden_dim; i++) {
    #pragma HLS pipeline II=1
        hls::vector<ap_int<read_bit*pack_num>, weight_parallel/pack_num> in_pack = weight_mmap[bias + i];
        hls::vector<ap_int<read_bit>, weight_parallel> data_pack;
        for(int j = 0; j < weight_parallel/pack_num; j++) {
        #pragma HLS UNROLL
            for(int k = 0; k < pack_num; k++) {
                data_pack[j * pack_num + k] = in_pack[j].range((k+1)*read_bit-1, k*read_bit);
            }
        }
        weight_stream.write(data_pack);
    }
}


template <int block_parallel, int weight_parallel, int max_input_dim = HIDDEN_DIM, int max_output_dim = HIDDEN_DIM>
void dec_weight_loader_int4_pack_2(
    tapa::mmap<hls::vector<ap_int<8>, weight_parallel/2>> weight_mmap,
    tapa::ostream<hls::vector<ap_int<4>, weight_parallel>>& weight_stream,
    int block_id,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim,
    int addr_bias = 0
){
    int bias = addr_bias + block_id * output_hidden_dim/(block_parallel*weight_parallel)*input_hidden_dim;
    read_weight_loop: for(int i = 0; i < output_hidden_dim/(block_parallel*weight_parallel)*input_hidden_dim; i++) {
    #pragma HLS pipeline II=1
        hls::vector<ap_int<8>, weight_parallel/2> in_pack = weight_mmap[bias + i];
        hls::vector<ap_int<4>, weight_parallel> data_pack;
        for(int j = 0; j < weight_parallel/2; j++) {
        #pragma HLS UNROLL
            data_pack[j * 2] = in_pack[j].range(3, 0);
            data_pack[j * 2 + 1] = in_pack[j].range(7, 4);
        }
        weight_stream.write(data_pack);
    }
}


template <int block_parallel, int weight_parallel, int max_input_dim = HIDDEN_DIM, int max_output_dim = HIDDEN_DIM>
void dec_weight_loader_int4_blockpack_2(
    tapa::mmap<hls::vector<ap_int<8>, weight_parallel>> weight_mmap,
    tapa::ostream<hls::vector<ap_int<4>, weight_parallel>>& weight_stream_0,
    tapa::ostream<hls::vector<ap_int<4>, weight_parallel>>& weight_stream_1,
    int block_id,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim,
    int addr_bias = 0
){
    int bias = addr_bias + block_id * output_hidden_dim/(block_parallel*weight_parallel)*input_hidden_dim;
    read_weight_loop: for(int i = 0; i < output_hidden_dim/(block_parallel*weight_parallel)*input_hidden_dim; i++) {
    #pragma HLS pipeline II=1
        hls::vector<ap_int<8>, weight_parallel> in_pack = weight_mmap[bias + i];
        hls::vector<ap_int<4>, weight_parallel> data_pack_0;
        hls::vector<ap_int<4>, weight_parallel> data_pack_1;
        for(int j = 0; j < weight_parallel; j++) {
        #pragma HLS UNROLL
            data_pack_0[j] = in_pack[j].range(3, 0);
            data_pack_1[j] = in_pack[j].range(7, 4);
        }
        weight_stream_0.write(data_pack_0);
        weight_stream_1.write(data_pack_1);
    }
}



template <typename T, int block_parallel, int weight_parallel, int max_input_dim = HIDDEN_DIM, int max_output_dim = HIDDEN_DIM>
void dec_weight_loader_buffer(
    tapa::istream<hls::vector<T, weight_parallel>>& weight_stream_in,
    tapa::ostream<hls::vector<T, weight_parallel>>& weight_stream_out,
    int block_id,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim
){
    read_weight_loop: for(int i = 0; i < output_hidden_dim/(block_parallel*weight_parallel)*input_hidden_dim; i++) {
    #pragma HLS pipeline II=1
        weight_stream_out.write(weight_stream_in.read());
    }
}


template <int block_parallel, bool is_act_asym=false, int max_output_dim = HIDDEN_DIM>
void dec_weight_s_loader_fp32(
    tapa::mmap<hls::vector<float, block_parallel * (1+is_act_asym)>> weight_s_sum_mmap,
    tapa::ostream<hls::vector<float, block_parallel * (1+is_act_asym)>>& weight_s_sum_stream,
    int block_id,
    int output_hidden_dim = max_output_dim,
    int addr_bias = 0
){
    int bias = addr_bias + block_id * output_hidden_dim/block_parallel;
    read_weight_loop: for(int i = 0; i < output_hidden_dim/block_parallel; i++) {
    #pragma HLS pipeline II=1
        hls::vector<float, block_parallel * (1+is_act_asym)> in_pack = weight_s_sum_mmap[bias + i];
        weight_s_sum_stream.write(in_pack);
    }
}

template <int block_parallel, bool is_act_asym=false, int max_output_dim = HIDDEN_DIM>
void dec_weight_s_loader_buffer_fp32(
    tapa::istream<hls::vector<float, block_parallel * (1+is_act_asym)>>& weight_s_sum_stream_in,
    tapa::ostream<hls::vector<float, block_parallel * (1+is_act_asym)>>& weight_s_sum_stream_out,
    int block_id,
    int output_hidden_dim = max_output_dim
){
    read_weight_loop: for(int i = 0; i < output_hidden_dim/block_parallel; i++) {
    #pragma HLS pipeline II=1
        weight_s_sum_stream_out.write(weight_s_sum_stream_in.read());
    }
}


template <int block_parallel, int weight_parallel, bool is_act_asym=false, int max_output_dim = HIDDEN_DIM>
void dec_weight_s_loader_fp32_bandwidth(
    tapa::mmap<hls::vector<float, 1+is_act_asym>> weight_s_sum_mmap,
    tapa::ostream<hls::vector<float, 1+is_act_asym>>& weight_s_sum_stream,
    int block_id,
    int output_hidden_dim = max_output_dim,
    int addr_bias = 0
){
    int bias = addr_bias + block_id * output_hidden_dim;
    // read_weight_loop: for(int i = 0; i < output_hidden_dim; i++){
    // #pragma HLS pipeline II=1
    //     hls::vector<float, 1+is_act_asym> in_pack = weight_s_sum_mmap[bias + i];
    //     weight_s_sum_stream.write(in_pack);
    // }

    read_weight_loop: for(int i = 0; i < output_hidden_dim/(block_parallel * weight_parallel); i++) {
        for(int j = 0; j < block_parallel; j++) {
        int block_bias = bias + j * (output_hidden_dim / block_parallel) + i * weight_parallel;
            for(int k = 0; k < weight_parallel; k++) {
            #pragma HLS pipeline II=1
                hls::vector<float, 1+is_act_asym> in_pack = weight_s_sum_mmap[block_bias + k];
                weight_s_sum_stream.write(in_pack);
            }
        }
    }

}


template <int block_parallel, bool is_act_asym=false, int max_output_dim = HIDDEN_DIM>
void dec_weight_s_loader_buffer_fp32_bandwidth(
    tapa::istream<hls::vector<float, 1+is_act_asym>>& weight_s_sum_stream_in,
    tapa::ostream<hls::vector<float, 1+is_act_asym>>& weight_s_sum_stream_out,
    int block_id,
    int output_hidden_dim = max_output_dim
){
    read_weight_loop: for(int i = 0; i < output_hidden_dim; i++){
    #pragma HLS pipeline II=1
        weight_s_sum_stream_out.write(weight_s_sum_stream_in.read());
    }
}


template <int block_parallel, int weight_parallel, int max_input_dim = HIDDEN_DIM, int max_log2_input_dim = log2_HIDDEN_DIM, int max_output_dim = HIDDEN_DIM, bool is_uint_input=false>
void dec_Linear_Layer_i4xi4(
    tapa::istream<hls::vector<ap_int<4>, block_parallel>>& input_seq,
    tapa::istreams<hls::vector<ap_int<4>, weight_parallel>, block_parallel>& weight_loaders,
    tapa::ostream<hls::vector<ap_int<max_log2_input_dim + 8>, block_parallel>>& output_seq,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim
){
    ap_int<4> A[max_input_dim];
    #pragma HLS bind_storage variable=A type=ram_2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=A type=block factor=block_parallel
    // #pragma HLS ARRAY_PARTITION variable=A type=cyclic factor=block_parallel
      
    hls::stream<ap_int<4>> block_A_loader[block_parallel];
    #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

    hls::stream<hls::vector<ap_int<4>, weight_parallel>> block_B_loader[block_parallel];
    #pragma HLS BIND_STORAGE variable=block_B_loader type=fifo impl=srl

    hls::stream<ap_int<max_log2_input_dim + 8>> block_C_drainer[block_parallel];
    #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl


    in_buf_loop: for (int k = 0; k < input_hidden_dim/block_parallel; k++) {    // L19
    #pragma HLS pipeline II=1
        hls::vector<ap_int<4>, block_parallel> A_pack = input_seq.read();
        for (int i = 0; i < block_parallel; i++) {
            A[i * max_input_dim/block_parallel + k] = A_pack[i]; //block partition
            // A[k * block_parallel + i] = A_pack[i]; //cyclic partition
        }
    }

    weight_block_loop: for(int N = 0; N < output_hidden_dim/(block_parallel * weight_parallel); N++){
    #pragma HLS DATAFLOW
        init_block_AB: for(int K = 0; K < block_parallel; K++){
            for(int k = 0; k < input_hidden_dim/block_parallel; k++){
            #pragma HLS PIPELINE II=1
                for (int i = 0; i < block_parallel; i++) {
                    block_A_loader[i].write(A[K * max_input_dim/block_parallel + k]);
                    block_B_loader[i].write(weight_loaders[i].read());
                }
            }
        }
        // if acitivation is asymmetric quantized, then input datatype is ap_uint<4>
        // else input datatype is ap_int<4>
        for (int i = 0; i < block_parallel; i++) {
        #pragma HLS UNROLL
            systolic_array_i4xi4_pack_1x2_1D<weight_parallel, max_log2_input_dim, is_uint_input>(
                block_A_loader[i], block_B_loader[i], block_C_drainer[i], input_hidden_dim
            );
        }

        bias_loop: for (int n = 0; n < weight_parallel; n++) {    // L41
        #pragma HLS pipeline II=1
            hls::vector<ap_int<max_log2_input_dim + 8>, block_parallel> output_pack;
            for (int i = 0; i < block_parallel; i++) {
                output_pack[i] = block_C_drainer[i].read();
            }
            output_seq.write(output_pack);
        }
    }
}


template <int block_parallel, int unroll_parallel, int weight_parallel, int max_input_dim = HIDDEN_DIM, int max_log2_input_dim = log2_HIDDEN_DIM, int max_output_dim = HIDDEN_DIM, bool is_uint_input=false>
void dec_Linear_Layer_i4xi4_unroll(
    tapa::istream<hls::vector<ap_int<4>, block_parallel>>& input_seq,
    tapa::istreams<hls::vector<ap_int<4>, weight_parallel>, block_parallel/unroll_parallel>& weight_loaders,
    tapa::ostream<hls::vector<ap_int<max_log2_input_dim + 8>, block_parallel/unroll_parallel>>& output_seq,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim
){
    ap_int<4> A[max_input_dim];
    #pragma HLS bind_storage variable=A type=ram_2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=A type=block factor=block_parallel
    // #pragma HLS ARRAY_PARTITION variable=A type=cyclic factor=block_parallel
      
    hls::stream<ap_int<4>> block_A_loader[block_parallel/unroll_parallel];
    #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

    hls::stream<hls::vector<ap_int<4>, weight_parallel>> block_B_loader[block_parallel/unroll_parallel];
    #pragma HLS BIND_STORAGE variable=block_B_loader type=fifo impl=srl

    hls::stream<ap_int<max_log2_input_dim + 8>> block_C_drainer[block_parallel/unroll_parallel];
    #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl


    in_buf_loop: for (int k = 0; k < input_hidden_dim/block_parallel; k++) {    // L19
    #pragma HLS pipeline II=1
        hls::vector<ap_int<4>, block_parallel> A_pack = input_seq.read();
        for (int i = 0; i < block_parallel; i++) {
            A[i * max_input_dim/block_parallel + k] = A_pack[i]; //block partition
        }
    }

    weight_block_loop: for(int N = 0; N < output_hidden_dim/(block_parallel * weight_parallel); N++){
    #pragma HLS DATAFLOW
        init_block_AB: for(int K = 0; K < block_parallel; K++){
            for(int k = 0; k < input_hidden_dim/block_parallel; k++){
            #pragma HLS PIPELINE II=1
                for (int i = 0; i < block_parallel/unroll_parallel; i++) {
                    block_A_loader[i].write(A[K * max_input_dim/block_parallel + k]);
                    block_B_loader[i].write(weight_loaders[i].read());
                }
            }
        }
        // if acitivation is asymmetric quantized, then input datatype is ap_uint<4>
        // else input datatype is ap_int<4>
        for (int i = 0; i < block_parallel/unroll_parallel; i++) {
        #pragma HLS UNROLL
            systolic_array_i4xi4_pack_1x2_1D<weight_parallel, max_log2_input_dim, is_uint_input>(
                block_A_loader[i], block_B_loader[i], block_C_drainer[i], input_hidden_dim
            );
        }

        bias_loop: for (int n = 0; n < weight_parallel; n++) {    // L41
        #pragma HLS pipeline II=1
            hls::vector<ap_int<max_log2_input_dim + 8>, block_parallel/unroll_parallel> output_pack;
            for (int i = 0; i < block_parallel/unroll_parallel; i++) {
                output_pack[i] = block_C_drainer[i].read();
            }
            output_seq.write(output_pack);
        }
    }
}





template <int block_size_b>
void systolic_array_fp32xfp32_pack_1x2_1D(
    hls::stream<float>& A_loader,
    hls::stream<hls::vector<float, block_size_b>>& B_loader, 
    hls::stream<float>& C_drainer,
    int k_size
  ) {
    hls::stream<float> A_fifo[block_size_b/2 + 1];
    #pragma HLS STREAM variable=A_fifo depth=4
    #pragma HLS BIND_STORAGE variable=A_fifo type=fifo impl=srl

    hls::stream<ap_uint<64>> B_fifo[block_size_b/2];
    #pragma HLS STREAM variable=B_fifo depth=block_size_b/2 + 1
    #pragma HLS BIND_STORAGE variable=B_fifo type=fifo impl=srl
  
    hls::stream<float> C_fifo[block_size_b/2];
    #pragma HLS STREAM variable=C_fifo depth=block_size_b/2
    #pragma HLS BIND_STORAGE variable=A_fifo type=fifo impl=srl
  
    #pragma HLS DATAFLOW
    data_load_AB:for (int k = 0; k < k_size; k++) {
    #pragma HLS PIPELINE II=1
        float A_temp = A_loader.read();
        hls::vector<float, block_size_b> B_temp = B_loader.read();
        
        A_fifo[0].write(A_temp);
        
        for (int n = 0; n < block_size_b/2; n++) {
            B_fifo[n].write((*(ap_uint<32>*)(&B_temp[2*n + 1]), *(ap_uint<32>*)(&B_temp[2*n])));
        }
    }
  
    systolic_array: for (int n = 0; n < block_size_b/2; n++) {
    #pragma HLS UNROLL
        if(n == block_size_b/2 - 1)
            PE_fp32xfp32_pack_1x2_1D<true>(A_fifo[n], A_fifo[n+1], B_fifo[n], C_fifo[n], k_size);
        else
            PE_fp32xfp32_pack_1x2_1D(A_fifo[n], A_fifo[n+1], B_fifo[n], C_fifo[n], k_size);
    }
  
    data_drain_C: for (int n = 0; n < block_size_b/2; n++) {
    #pragma HLS PIPELINE II=2
        C_drainer.write(C_fifo[n].read());
        C_drainer.write(C_fifo[n].read());
    }
}


template <int block_parallel, int weight_parallel, int max_input_dim = HIDDEN_DIM, int max_output_dim = HIDDEN_DIM>
void dec_Linear_Layer_fp32xfp32(
    tapa::istream<hls::vector<float, block_parallel>>& input_seq,
    tapa::istreams<hls::vector<float, weight_parallel>, block_parallel>& weight_loaders,
    tapa::ostream<hls::vector<float, block_parallel>>& output_seq,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim
){
    float A[max_input_dim];
    #pragma HLS bind_storage variable=A type=ram_2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=A type=block factor=block_parallel
    // #pragma HLS ARRAY_PARTITION variable=A type=cyclic factor=block_parallel
      
    hls::stream<float> block_A_loader[block_parallel];
    #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

    hls::stream<hls::vector<float, weight_parallel>> block_B_loader[block_parallel];
    #pragma HLS BIND_STORAGE variable=block_B_loader type=fifo impl=srl

    hls::stream<float> block_C_drainer[block_parallel];
    #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl

    
    in_buf_loop: for (int k = 0; k < input_hidden_dim/block_parallel; k++) {    // L19
    #pragma HLS pipeline II=1
        hls::vector<float, block_parallel> A_pack = input_seq.read();
        for (int i = 0; i < block_parallel; i++) {
            A[i * max_input_dim/block_parallel + k] = A_pack[i];
            // A[k * block_parallel + i] = A_pack[i];
        }
    }

    weight_block_loop: for(int N = 0; N < output_hidden_dim/(block_parallel * weight_parallel); N++){
    #pragma HLS DATAFLOW
        init_block_AB: for(int K = 0; K < block_parallel; K++){
            for(int k = 0; k < input_hidden_dim/block_parallel; k++){
            #pragma HLS PIPELINE II=1
                for (int i = 0; i < block_parallel; i++) {
                    block_A_loader[i].write(A[K * max_input_dim/block_parallel + k]);
                    block_B_loader[i].write(weight_loaders[i].read());
                }
            }
        }
        
        for (int i = 0; i < block_parallel; i++) {
        #pragma HLS UNROLL
            systolic_array_fp32xfp32_pack_1x2_1D<weight_parallel>(
                block_A_loader[i], block_B_loader[i], block_C_drainer[i], input_hidden_dim
            );
        }

        bias_loop: for (int n = 0; n < weight_parallel; n++) {    // L41
        #pragma HLS pipeline II=1
            hls::vector<float, block_parallel> output_pack;
            for (int i = 0; i < block_parallel; i++) {
                output_pack[i] = block_C_drainer[i].read();
            }
            output_seq.write(output_pack);
        }
    }
}


template <int block_parallel, int max_hidden_dim = HIDDEN_DIM>
void dec_Gate_Layer_fp32xfp32(
    tapa::istream<hls::vector<float, block_parallel>>& input_seq,
    tapa::istream<hls::vector<float, block_parallel>>& gate_seq,
    tapa::ostream<hls::vector<float, block_parallel>>& output_seq,
    int io_hidden_dim = max_hidden_dim
){
    io_block_loop: for (int M = 0; M < io_hidden_dim/block_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel
    #pragma HLS pipeline II=1
        hls::vector<float, block_parallel> input_pack = input_seq.read();
        hls::vector<float, block_parallel> gate_pack = gate_seq.read();
        hls::vector<float, block_parallel> output_pack;
        for(int i = 0; i < block_parallel; i++){
            output_pack[i] = input_pack[i] * gate_pack[i];
        }
        output_seq.write(output_pack);
    }
}




#endif