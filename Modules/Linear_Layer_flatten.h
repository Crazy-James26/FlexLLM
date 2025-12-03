#ifndef _LL_Flatten_H_
#define _LL_Flatten_H_
#include "config.h"
#include "PE.h"
#include "data_io.h"

template <typename T, int block_parallel, int weight_parallel, int max_input_dim = HIDDEN_DIM, int max_output_dim = HIDDEN_DIM>
void dec_Linear_Layer_input_broadcastor(
    tapa::istream<hls::vector<T, block_parallel>>& input_seq,
    tapa::ostreams<T, block_parallel>& input_loaders,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim
){
    T A[max_input_dim];
    #pragma HLS bind_storage variable=A type=ram_2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=A type=block factor=block_parallel
    // #pragma HLS ARRAY_PARTITION variable=A type=cyclic factor=block_parallel


    in_buf_loop: for (int k = 0; k < input_hidden_dim/block_parallel; k++) {    // L19
    #pragma HLS pipeline II=1
        hls::vector<T, block_parallel> A_pack = input_seq.read();
        for (int i = 0; i < block_parallel; i++) {
            A[i * input_hidden_dim/block_parallel + k] = A_pack[i]; //block partition
        }
    }

    weight_block_loop: for(int N = 0; N < output_hidden_dim/(block_parallel * weight_parallel); N++){
        init_block_AB: for(int k = 0; k < input_hidden_dim; k++){
        #pragma HLS PIPELINE II=1
            for (int i = 0; i < block_parallel; i++) {
                input_loaders[i].write(A[k]);
            }
        }
    }
}



template <int block_size_b, int max_log2_k_size = log2_HIDDEN_DIM, bool is_uint_A = false>
void systolic_array_i4xi4_pack_1x2_flatten_1D(
    tapa::istream<ap_int<4>>& A_loader,
    tapa::istream<hls::vector<ap_int<4>, block_size_b>>& B_loader, 
    tapa::ostream<ap_int<max_log2_k_size + 8>>& C_drainer,
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

template <int block_parallel, int weight_parallel, int max_input_dim = HIDDEN_DIM, int max_log2_input_dim = log2_HIDDEN_DIM, int max_output_dim = HIDDEN_DIM, bool is_uint_input=false>
void dec_Linear_Layer_i4xi4_flatten(
    tapa::istream<ap_int<4>>& input_loader,
    tapa::istream<hls::vector<ap_int<4>, weight_parallel>>& weight_loader,
    tapa::ostream<ap_int<max_log2_input_dim + 8>>& output_drainer,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim
){
    weight_block_loop: for(int N = 0; N < output_hidden_dim/(block_parallel * weight_parallel); N++){
        // if acitivation is asymmetric quantized, then input datatype is ap_uint<4>
        // else input datatype is ap_int<4>
        systolic_array_i4xi4_pack_1x2_flatten_1D<weight_parallel, max_log2_input_dim, is_uint_input>(
            input_loader, weight_loader, output_drainer, input_hidden_dim
        );
    }
}


template <int block_size_b>
void systolic_array_fp32xfp32_pack_1x2_flatten_1D(
    tapa::istream<float>& A_loader,
    tapa::istream<hls::vector<float, block_size_b>>& B_loader, 
    tapa::ostream<float>& C_drainer,
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
void dec_Linear_Layer_fp32xfp32_flatten(
    tapa::istream<float>& input_loader,
    tapa::istream<hls::vector<float, weight_parallel>>& weight_loader,
    tapa::ostream<float>& output_drainer,
    int input_hidden_dim = max_input_dim, 
    int output_hidden_dim = max_output_dim
){
    weight_block_loop: for(int N = 0; N < output_hidden_dim/(block_parallel * weight_parallel); N++){
        systolic_array_fp32xfp32_pack_1x2_flatten_1D<weight_parallel>(
            input_loader, weight_loader, output_drainer, input_hidden_dim
        );
    }
}

template <typename T, int block_parallel, int weight_parallel, int max_output_dim = HIDDEN_DIM>
void dec_Linear_Layer_output_merger(
    tapa::istreams<T, block_parallel>& output_drainers,
    tapa::ostream<hls::vector<T, block_parallel>>& output_seq,
    int output_hidden_dim = max_output_dim
){
    weight_block_loop: for(int N = 0; N < output_hidden_dim/(block_parallel * weight_parallel); N++){
        bias_loop: for (int n = 0; n < weight_parallel; n++) {    // L41
        #pragma HLS pipeline II=1
            hls::vector<T, block_parallel> output_pack;
            for (int i = 0; i < block_parallel; i++) {
                output_pack[i] = output_drainers[i].read();
            }
            output_seq.write(output_pack);
        }
    }
}





#endif