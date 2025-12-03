#ifndef _MHA_H_
#define _MHA_H_
#include "config.h"
#include "PE.h"
#include "data_io.h"



template <typename T, int K_s_parallel, int K_parallel, int K_head_num, int max_seq_len=MAX_PRE_SEQ_LEN>
void pref_K_s_cache_manager_template(
    tapa::istream<hls::vector<T, K_s_parallel>>& input_k_s_stream,
    tapa::mmap<hls::vector<T, K_s_parallel>> k_s_cache,
    tapa::ostream<hls::vector<T, K_s_parallel>>& output_k_s_stream,
    int block_id,
    int seq_len = max_seq_len
){
    input_k_block_loop: for(int N = 0; N < seq_len/K_parallel; N++){
        i_k_head_loop: for (int H = 0; H < K_head_num; H++){
            // store k_s
            int bias = (block_id * K_head_num + H) * max_seq_len/K_s_parallel + N * K_parallel/K_s_parallel;
            store_k_s_loop: for(int i = 0; i < K_parallel/K_s_parallel; i++){
            #pragma HLS pipeline II=1
                hls::vector<T, K_s_parallel> k_s_pack = input_k_s_stream.read();
                k_s_cache[bias + i] = k_s_pack;
            }
        }
    }


    // read k
    o_k_s_head_loop: for (int H = 0; H < K_head_num; H++){
        int bias = (block_id * K_head_num + H) * max_seq_len/K_s_parallel;
        read_k_s_loop: for(int i = 0; i < seq_len/K_s_parallel; i++){
        #pragma HLS pipeline II=1
            hls::vector<T, K_s_parallel> k_s_pack = k_s_cache[bias + i];
            output_k_s_stream.write(k_s_pack);
        }
    }
}


template <typename T, int K_s_parallel, int K_parallel, int K_head_num, int max_seq_len=MAX_PRE_SEQ_LEN>
void pref_K_s_cache_manager_static_template(
    tapa::mmap<hls::vector<T, K_s_parallel>> k_s_cache,
    tapa::ostream<hls::vector<T, K_s_parallel>>& output_k_s_stream,
    int block_id,
    int seq_len = max_seq_len
){
    input_k_block_loop: for(int N = 0; N < seq_len/K_parallel; N++){
        i_k_head_loop: for (int H = 0; H < K_head_num; H++){
            // store k_s
            int bias = (block_id * K_head_num + H) * max_seq_len/K_s_parallel + N * K_parallel/K_s_parallel;
            i_k_s_loop: for(int i = 0; i < K_parallel/K_s_parallel; i++){
            #pragma HLS pipeline II=1
                hls::vector<T, K_s_parallel> k_s_pack = k_s_cache[bias + i];
                output_k_s_stream.write(k_s_pack);
            }
        }
    }


    // read k for dequant
    dequant_k_s_head_loop: for (int H = 0; H < K_head_num; H++){
        int bias = (block_id * K_head_num + H) * max_seq_len/K_s_parallel;
        dequant_k_s_loop: for(int i = 0; i < seq_len/K_s_parallel; i++){
        #pragma HLS pipeline II=1
            hls::vector<T, K_s_parallel> k_s_pack = k_s_cache[bias + i];
            output_k_s_stream.write(k_s_pack);
        }
    }
}


template <typename T, int io_parallel, int K_parallel, int K_head_num, int K_head_dim=HEAD_DIM, int group_num=1, int max_seq_len=MAX_PRE_SEQ_LEN>
void pref_K_cache_manager_template(
    tapa::istream<hls::vector<T, K_parallel>>& input_k_stream,
    tapa::mmap<hls::vector<T, K_parallel>> k_cache,
    tapa::ostream<hls::vector<T, K_parallel>>& output_k_stream,
    int block_id,
    int seq_len = max_seq_len
){
    input_k_block_loop: for(int N = 0; N < seq_len/K_parallel; N++){
        i_k_head_loop: for (int H = 0; H < K_head_num; H++){
            // store k
            int bias = ((block_id * K_head_num + H) * max_seq_len/K_parallel + N) * K_head_dim;
            store_k_loop: for(int i = 0; i < K_head_dim; i++){
            #pragma HLS pipeline II=1
                hls::vector<T, K_parallel> k_pack = input_k_stream.read();
                k_cache[bias + i] = k_pack;
            }
        }
    }
    cout << "Block_id: " << block_id << " K_cache storing completed." << endl;

    // read k
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
        o_k_head_loop: for (int H = 0; H < K_head_num; H++){
            attn_group_loop: for (int G = 0; G < group_num; G++){
                int bias = (block_id * K_head_num + H) * max_seq_len/K_parallel * K_head_dim;
                read_k_loop: for(int i = 0; i < seq_len/K_parallel * K_head_dim; i++){
                #pragma HLS pipeline II=1
                    hls::vector<T, K_parallel> k_pack = k_cache[bias + i];
                    output_k_stream.write(k_pack);
                }
            }
        }
    }
    cout << "Block_id: " << block_id << " K_cache loading completed." << endl;
}



template <int block_size_a, int block_size_b, int max_log2_k_size = log2_HIDDEN_DIM, bool is_uint_A = false>
void systolic_array_i8xi8_pack_2x2(
    hls::stream<hls::vector<ap_int<8>, block_size_a>>& A_loader,
    hls::stream<hls::vector<ap_int<8>, block_size_b>>& B_loader, 
    hls::stream<hls::vector<ap_int<max_log2_k_size + 16>, block_size_a>>& C_drainer,
    int k_size
  ) {
    hls::stream<ap_uint<16>> A_fifo[block_size_a/2][block_size_b/2 + 1];
    hls::stream<ap_uint<16>> B_fifo[block_size_b/2][block_size_a/2 + 1];
    #pragma HLS STREAM variable=A_fifo depth=block_size_a/2 + 1
    #pragma HLS BIND_STORAGE variable=A_fifo type=fifo impl=srl
    #pragma HLS STREAM variable=B_fifo depth=block_size_b/2 + 1
    #pragma HLS BIND_STORAGE variable=B_fifo type=fifo impl=srl
  
    hls::stream<ap_uint<2*max_log2_k_size + 32>> C_fifo[block_size_a/2][block_size_b/2];
    #pragma HLS STREAM variable=C_fifo depth=block_size_a/2 + 1
    #pragma HLS BIND_STORAGE variable=A_fifo type=fifo impl=srl
  
    #pragma HLS DATAFLOW
    data_load_AB:for (int k = 0; k < k_size; k++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=1 max=(1<<max_log2_k_size)
        hls::vector<ap_int<8>, block_size_a> A_temp = A_loader.read();
        hls::vector<ap_int<8>, block_size_b> B_temp = B_loader.read();
  
        for (int m = 0; m < block_size_a/2; m++) {
            A_fifo[m][0].write(ap_uint<16>((A_temp[2*m + 1],A_temp[2*m])));
        }
        
        for (int n = 0; n < block_size_b/2; n++) {
            B_fifo[n][0].write(ap_uint<16>((B_temp[2*n + 1], B_temp[2*n])));
        }
    }
  
    systolic_array: for (int m = 0; m < block_size_a/2; m++) {
    #pragma HLS UNROLL
        for (int n = 0; n < block_size_b/2; n++) {
        #pragma HLS UNROLL
            if(m == block_size_a/2 - 1 && n == block_size_b/2 - 1)
                PE_i8xi8_pack_2x2_2xDSP_2D<is_uint_A, max_log2_k_size, true, true>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
            else if(m == block_size_a/2 - 1)
                PE_i8xi8_pack_2x2_2xDSP_2D<is_uint_A, max_log2_k_size, false, true>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
            else if(n == block_size_b/2 - 1)
                PE_i8xi8_pack_2x2_2xDSP_2D<is_uint_A, max_log2_k_size, true, false>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
            else
                PE_i8xi8_pack_2x2_2xDSP_2D<is_uint_A, max_log2_k_size, false, false>(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C_fifo[m][n], k_size);
        }
    }
  
    // data_drain_AB:for (int k = 0; k < k_size; k++) {
    // #pragma HLS PIPELINE II=1
    // #pragma HLS LOOP_TRIPCOUNT min=1 max=(1<<max_log2_k_size)
    //     for (int m = 0; m < block_size_a/2; m++) {
    //         A_fifo[m][block_size_b/2].read();
    //     }
    //     for (int n = 0; n < block_size_b/2; n++) {
    //         B_fifo[n][block_size_a/2].read();
    //     }
    // }
  
    data_drain_C: for (int n = 0; n < block_size_b/2; n++) {
    #pragma HLS PIPELINE II=2
        hls::vector<ap_int<max_log2_k_size + 16>, block_size_a> C_temp;
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

template <int io_parallel, int K_parallel, int mha_head_num, int mha_head_dim = HEAD_DIM, int log2_mha_head_dim = log2_HEAD_DIM, int max_seq_len = MAX_PRE_SEQ_LEN, bool is_uint_input=false>
void pref_MHA_i8xi8_qxk_template(
    tapa::istream<hls::vector<ap_int<8>, io_parallel>>& input_seq,
    tapa::istream<hls::vector<ap_int<8>, K_parallel>>& weight_loader,
    tapa::ostream<hls::vector<ap_int<log2_mha_head_dim + 16>, io_parallel>>& output_seq,
    int seq_len = max_seq_len
){
    // hls::vector<ap_int<8>, io_parallel> A[mha_head_num * mha_head_dim];
    hls::vector<ap_int<8>, io_parallel> A[mha_head_dim];

    hls::stream<hls::vector<ap_int<8>, io_parallel>> block_A_loader;
    #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

    hls::stream<hls::vector<ap_int<8>, K_parallel>> block_B_loader;
    #pragma HLS BIND_STORAGE variable=block_B_loader type=fifo impl=srl

    hls::stream<hls::vector<ap_int<log2_mha_head_dim + 16>, io_parallel>> block_C_drainer;
    #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl

    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        // in_buf_loop: for (int k = 0; k < mha_head_num * mha_head_dim; k++) {    // L19
        // #pragma HLS pipeline II=1
        //     A[k] = input_seq.read();
        // }

        attn_head_loop: for (int H = 0; H < mha_head_num; H++){
            in_buf_loop: for (int k = 0; k < mha_head_dim; k++) {    // L19
            #pragma HLS pipeline II=1
                A[k] = input_seq.read();
            }

            k_weight_block_loop: for(int N = 0; N < seq_len/K_parallel; N++){
            #pragma HLS loop_tripcount min=1 max=max_seq_len/K_parallel
            #pragma HLS DATAFLOW
                init_block_AB: for(int k = 0; k < mha_head_dim; k++){
                #pragma HLS PIPELINE II=1
                    // block_A_loader.write(A[H * mha_head_dim + k]);
                    block_A_loader.write(A[k]);
                    block_B_loader.write(weight_loader.read());
                }

                systolic_array_i8xi8_pack_2x2<io_parallel, K_parallel, log2_mha_head_dim, is_uint_input>(
                    block_A_loader, block_B_loader, block_C_drainer, mha_head_dim
                );

                output_scale_loop: for (int n = 0; n < K_parallel; n++) {    // L41
                #pragma HLS pipeline II=1
                    hls::vector<ap_int<log2_mha_head_dim + 16>, io_parallel> outp_pack = block_C_drainer.read();
                    output_seq.write(outp_pack);
                }
            }
        }
    }
}


template <typename T, int io_parallel, int max_hidden_dim=HIDDEN_DIM, int max_seq_len=MAX_PRE_SEQ_LEN, int head_num=1>
void pref_causal_mask_template(
    tapa::istream<hls::vector<T, io_parallel>>& input_stream,
    tapa::ostream<hls::vector<T, io_parallel>>& output_stream,
    int seq_len = max_seq_len,
    int io_hidden_dim = max_hidden_dim
){
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
        #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        attn_head_loop: for (int H = 0; H < head_num; H++){
            mask_loop: for (int k = 0; k < io_hidden_dim; k++) {
            #pragma HLS loop_tripcount min=1 max=max_hidden_dim
            #pragma HLS pipeline II=1
                hls::vector<T, io_parallel> input_pack = input_stream.read();
                hls::vector<T, io_parallel> output_pack;
                for(int i = 0; i < io_parallel; i++){
                    if(k <= M * io_parallel + i){
                        output_pack[i] = input_pack[i];
                    }
                    else {
                        output_pack[i] = -1e32;
                    }
                }
                output_stream.write(output_pack);
            }
        }
    }
}



template <typename T, int V_s_parallel, int V_hidden_dim>
void pref_V_s_cache_manager_static_template(
    tapa::mmap<hls::vector<T, V_s_parallel>> v_s_cache,
    tapa::ostream<hls::vector<T, V_s_parallel>>& output_v_s_stream,
    int block_id
){
    // store v_s
    int bias = block_id * V_hidden_dim / V_s_parallel;
    store_v_s_loop: for (int i = 0; i < V_hidden_dim / V_s_parallel; i++) {
    #pragma HLS pipeline II=1
        hls::vector<T, V_s_parallel> v_s_pack = v_s_cache[bias + i];
        output_v_s_stream.write(v_s_pack);
    }

    output_v_s_loop: for (int i = 0; i < V_hidden_dim / V_s_parallel; i++) {
    #pragma HLS pipeline II=1
        hls::vector<T, V_s_parallel> v_s_pack = v_s_cache[bias + i];
        output_v_s_stream.write(v_s_pack);
    }
}


template <typename T, int io_parallel, int V_parallel, int V_head_num, int V_head_dim=HEAD_DIM, int group_num=1, int max_seq_len=MAX_PRE_SEQ_LEN>
void pref_V_cache_manager_template(
    tapa::istream<hls::vector<T, V_parallel>>& input_v_stream,
    tapa::mmap<hls::vector<T, V_parallel>> v_cache,
    tapa::ostream<hls::vector<T, V_parallel>>& output_v_stream,
    int block_id,
    int seq_len = max_seq_len
){

    // when using "io_buffer_transpose" to transpose V
    i_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
        o_block_loop: for (int N = 0; N < (V_head_num * V_head_dim)/V_parallel; N++){
            int bias = (block_id * (V_head_num * V_head_dim) /V_parallel + N) * max_seq_len + M * io_parallel;
            store_v_loop: for(int i = 0; i < io_parallel; i++){
            #pragma HLS pipeline II=1
                hls::vector<T, V_parallel> v_pack = input_v_stream.read();
                v_cache[bias + i] = v_pack;
            }
        }
    }
    cout << "Block_id: " << block_id << " V_cache storing completed." << endl;
    
    output_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
        attn_head_loop: for (int H = 0; H < V_head_num; H++){
            attn_group_loop: for (int G = 0; G < group_num; G++){
                attn_seq_loop: for(int h_block = 0; h_block < V_head_dim/V_parallel; h_block++){
                    int bias = ((block_id * V_head_num + H) * V_head_dim/V_parallel + h_block) * max_seq_len;
                    read_v_loop: for(int i = 0; i < seq_len; i++){
                    #pragma HLS pipeline II=1
                        hls::vector<T, V_parallel> v_pack = v_cache[bias + i];
                        output_v_stream.write(v_pack);
                    }
                }
            }
        }
    }
    cout << "Block_id: " << block_id << " V_cache loading completed." << endl;
}


template <int io_parallel, int V_parallel, int mha_head_num, int mha_head_dim = HEAD_DIM, int max_seq_len = MAX_PRE_SEQ_LEN, int log2_max_seq_len = log2_MAX_PRE_SEQ_LEN, bool is_uint_input=false>
void pref_MHA_i8xi8_axv_template(
    tapa::istream<hls::vector<ap_int<8>, io_parallel>>& input_seq,
    tapa::istream<hls::vector<ap_int<8>, V_parallel>>& weight_loader,
    tapa::ostream<hls::vector<ap_int<log2_max_seq_len + 16>, io_parallel>>& output_seq,
    int seq_len = max_seq_len
){
    hls::vector<ap_int<8>, io_parallel> A[max_seq_len];
      
    hls::stream<hls::vector<ap_int<8>, io_parallel>> block_A_loader;
    #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

    hls::stream<hls::vector<ap_int<8>, V_parallel>> block_B_loader;
    #pragma HLS BIND_STORAGE variable=block_B_loader type=fifo impl=srl

    hls::stream<hls::vector<ap_int<log2_max_seq_len + 16>, io_parallel>> block_C_drainer;
    #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl

    
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        attn_head_loop: for (int H = 0; H < mha_head_num; H++){

            in_buf_loop: for (int k = 0; k < seq_len; k++) {    // L19
            #pragma HLS loop_tripcount min=1 max=max_seq_len
            #pragma HLS pipeline II=1
                A[k] = input_seq.read();
            }

            v_weight_block_loop: for(int N = 0; N < mha_head_dim/V_parallel; N++){
            #pragma HLS DATAFLOW
                init_block_AB: for(int k = 0; k < seq_len; k++){
                #pragma HLS loop_tripcount min=1 max=max_seq_len
                #pragma HLS PIPELINE II=1
                    block_A_loader.write(A[k]);
                    block_B_loader.write(weight_loader.read());
                }

                systolic_array_i8xi8_pack_2x2<io_parallel, V_parallel, log2_max_seq_len, is_uint_input>(
                    block_A_loader, block_B_loader, block_C_drainer, seq_len
                );

                output_scale_loop: for (int n = 0; n < V_parallel; n++) {    // L41
                #pragma HLS pipeline II=1
                    hls::vector<ap_int<log2_max_seq_len + 16>, io_parallel> outp_pack = block_C_drainer.read();
                    output_seq.write(outp_pack);
                }
            }
        }
    }
}



template <typename T, int head_parallel, int K_head_num, int group_num=ATTN_GROUP_NUM, int max_sum_seq_len=MAX_SUM_SEQ_LEN>
void dec_K_s_cache_manager_template(
    tapa::istream<hls::vector<T, head_parallel>>& input_k_s_stream,
    tapa::mmap<hls::vector<T, head_parallel>>& k_s_cache,
    tapa::ostream<hls::vector<T, head_parallel>>& output_k_s_stream,
    int block_id,
    int seq_id,
    int addr_bias = 0
){
    i_k_head_loop: for (int H = 0; H < K_head_num/head_parallel; H++){
        // store k_s
        int bias = addr_bias + (block_id * K_head_num/head_parallel + H) * max_sum_seq_len + seq_id;
        k_s_cache[bias] = input_k_s_stream.read();
    }
    
    // read k
    o_k_s_head_loop: for (int H = 0; H < K_head_num/head_parallel; H++){
        attn_group_loop: for (int G = 0; G < group_num; G++){
            int bias = addr_bias + (block_id * K_head_num/head_parallel + H) * max_sum_seq_len;
            read_k_s_loop: for(int i = 0; i <= seq_id; i++){
            #pragma HLS pipeline II=1
                output_k_s_stream.write(k_s_cache[bias + i]);
            }
        }
    }
}


template <typename T, int head_parallel, int K_head_num, int group_num=ATTN_GROUP_NUM, int max_sum_seq_len=MAX_SUM_SEQ_LEN>
void dec_K_s_cache_manager_static_template(
    tapa::mmap<hls::vector<T, head_parallel>> k_s_cache,
    tapa::ostream<hls::vector<T, head_parallel>>& output_k_s_stream,
    int block_id,
    int seq_id,
    int addr_bias = 0
){
    i_k_head_loop: for (int H = 0; H < K_head_num/head_parallel; H++){
        // store k_s
        int bias = addr_bias + (block_id * K_head_num/head_parallel + H) * max_sum_seq_len + seq_id;
        output_k_s_stream.write(k_s_cache[bias]);
    }
    
    // read k
    o_k_s_head_loop: for (int H = 0; H < K_head_num/head_parallel; H++){
        attn_group_loop: for (int G = 0; G < group_num; G++){
            int bias = addr_bias + (block_id * K_head_num/head_parallel + H) * max_sum_seq_len;
            read_k_s_loop: for(int i = 0; i <= seq_id; i++){
            #pragma HLS pipeline II=1
                output_k_s_stream.write(k_s_cache[bias + i]);
            }
        }
    }
}

template <typename T, int head_parallel, int K_parallel, int K_hidden_dim=KV_HIDDEN_DIM, int decoder_layer_num=DECODER_LAYER_NUM>
void dec_K_cache_buffer_template(
    tapa::istream<hls::vector<T, head_parallel>>& input_k_stream,
    tapa::ostreams<hls::vector<T, K_parallel>, head_parallel>& output_k_streams,
    int block_id,
    int seq_id
){
    static T k_reg[decoder_layer_num][head_parallel][K_hidden_dim/head_parallel][K_parallel];
    #pragma HLS ARRAY_PARTITION variable=k_reg type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=k_reg type=cyclic factor=K_parallel/2 dim=4

    reg_k_loop: for (int k = 0; k < K_hidden_dim/head_parallel; k++) {
    #pragma HLS pipeline II=1
        hls::vector<T, head_parallel> data_pack = input_k_stream.read();
        for (int i = 0; i < head_parallel; i++) {
            if(seq_id % K_parallel == 0) {
                for(int j = 0; j < K_parallel; j++){
                    k_reg[block_id][i][k][j] = 0;
                }
            }
            k_reg[block_id][i][k][seq_id % K_parallel] = data_pack[i];
        }
    }

    send_k_loop: for (int k = 0; k < K_hidden_dim/head_parallel; k++) {
    #pragma HLS pipeline II=1
        for (int i = 0; i < head_parallel; i++) {
            hls::vector<T, K_parallel> K_cache_pack;
            for(int j = 0; j < K_parallel; j++){
                K_cache_pack[j] = k_reg[block_id][i][k][j] ;
            }
            output_k_streams[i].write(K_cache_pack);
        }
    }
}


template <typename T, int head_parallel, int K_parallel, int K_head_num, int K_head_dim=HEAD_DIM, int group_num=ATTN_GROUP_NUM, int max_sum_seq_len=MAX_SUM_SEQ_LEN>
void dec_K_cache_manager_template(
    tapa::istream<hls::vector<T, K_parallel>>& input_k_stream,
    tapa::mmap<hls::vector<T, K_parallel>>& k_cache,
    tapa::ostream<hls::vector<T, K_parallel>>& output_k_stream,
    int block_id,
    int seq_id,
    int addr_bias = 0
){
    store_k_head_loop: for (int H = 0; H < K_head_num/head_parallel; H++){
        int bias = addr_bias + ((block_id * K_head_num/head_parallel + H) * max_sum_seq_len + seq_id)/K_parallel * K_head_dim;
        store_k_loop: for(int i = 0; i < K_head_dim; i++){
        #pragma HLS pipeline II=1
            k_cache[bias + i] = input_k_stream.read();
        }
    }
    printf("Sum_seq_id %d: Block_id %d: K_cache_manager storing completed.\n", seq_id, block_id);
    

    int seq_block_id = seq_id / K_parallel;
    // read k
    o_k_head_loop: for (int H = 0; H < K_head_num/head_parallel; H++){
        attn_group_loop: for (int G = 0; G < group_num; G++){
            int bias = addr_bias + (block_id * K_head_num/head_parallel + H) * max_sum_seq_len/K_parallel * K_head_dim;
            read_k_head_loop: for(int i = 0; i < (seq_block_id + 1) * K_head_dim; i++){
            #pragma HLS pipeline II=1
                    output_k_stream.write(k_cache[bias + i]);
            }
        }
    }
    printf("Sum_seq_id %d: Block_id %d: K_cache_manager loading completed.\n", seq_id, block_id);
}


template <typename T, int head_parallel, int K_parallel_read, int K_parallel, int K_head_num, int K_head_dim=HEAD_DIM, int group_num=ATTN_GROUP_NUM, int max_sum_seq_len=MAX_SUM_SEQ_LEN>
void dec_K_cache_manager_discard_template(
    tapa::istream<hls::vector<T, K_parallel>>& input_k_stream,
    tapa::mmap<hls::vector<T, K_parallel_read>>& k_cache,
    tapa::ostream<hls::vector<T, K_parallel>>& output_k_stream,
    int block_id,
    int seq_id,
    int addr_bias = 0
){
    store_k_head_loop: for (int H = 0; H < K_head_num/head_parallel; H++){
        int bias = addr_bias + ((block_id * K_head_num/head_parallel + H) * max_sum_seq_len + seq_id)/K_parallel * K_head_dim;
        store_k_loop: for(int i = 0; i < K_head_dim; i++){
        #pragma HLS pipeline II=1
            hls::vector<T, K_parallel> input_pack = input_k_stream.read();
            hls::vector<T, K_parallel_read> k_pack;
            for(int j = 0; j < K_parallel; j++){
                k_pack[j] = input_pack[j];
            }
            k_cache[bias + i] = k_pack;
        }
    }
    printf("Sum_seq_id %d: Block_id %d: K_cache_manager storing completed.\n", seq_id, block_id);
    

    int seq_block_id = seq_id / K_parallel;
    // read k
    o_k_head_loop: for (int H = 0; H < K_head_num/head_parallel; H++){
        attn_group_loop: for (int G = 0; G < group_num; G++){
            int bias = addr_bias + (block_id * K_head_num/head_parallel + H) * max_sum_seq_len/K_parallel * K_head_dim;
            read_k_head_loop: for(int i = 0; i < (seq_block_id + 1) * K_head_dim; i++){
            #pragma HLS pipeline II=1
                hls::vector<T, K_parallel_read> k_pack = k_cache[bias + i];
                hls::vector<T, K_parallel> output_pack;
                for(int j = 0; j < K_parallel; j++){
                    output_pack[j] = k_pack[j];
                }
                output_k_stream.write(output_pack);
            }
        }
    }
    printf("Sum_seq_id %d: Block_id %d: K_cache_manager loading completed.\n", seq_id, block_id);
}




template <int block_size_b, int max_log2_k_size = log2_HIDDEN_DIM, bool is_uint_A = false>
void systolic_array_i8xi8_pack_1x2_1D(
    hls::stream<ap_int<8>>& A_loader,
    hls::stream<hls::vector<ap_int<8>, block_size_b>>& B_loader, 
    hls::stream<ap_int<max_log2_k_size + 16>>& C_drainer,
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



template <int head_parallel, int K_parallel, int mha_head_num, int mha_head_dim = HEAD_DIM, int log2_mha_head_dim = log2_HEAD_DIM, int max_sum_seq_len=MAX_SUM_SEQ_LEN, bool is_uint_input=false>
void dec_MHA_i8xi8_qxk_template(
    tapa::istream<hls::vector<ap_int<8>, head_parallel>>& input_seq,
    tapa::istreams<hls::vector<ap_int<8>, K_parallel>, head_parallel>& weight_loaders,
    tapa::ostream<hls::vector<ap_int<log2_mha_head_dim + 16>, head_parallel>>& output_seq,
    int seq_len
){
    hls::vector<ap_int<8>, head_parallel> A[mha_head_num/head_parallel * mha_head_dim];

    hls::stream<ap_int<8>> block_A_loader[head_parallel];
    #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

    hls::stream<hls::vector<ap_int<8>, K_parallel>> block_B_loader[head_parallel];
    #pragma HLS BIND_STORAGE variable=block_B_loader type=fifo impl=srl

    hls::stream<ap_int<log2_mha_head_dim + 16>> block_C_drainer[head_parallel];
    #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl

    
    in_buf_loop: for (int k = 0; k < mha_head_num/head_parallel * mha_head_dim; k++) {    // L19
    #pragma HLS pipeline II=1
        A[k] = input_seq.read();
    }

    attn_head_loop: for (int H = 0; H < mha_head_num/head_parallel; H++){
        k_weight_block_loop: for(int N = 0; N < (seq_len + K_parallel - 1) / K_parallel; N++){
        #pragma HLS loop_tripcount min=1 max=max_sum_seq_len/K_parallel
        #pragma HLS DATAFLOW
            init_block_AB: for(int k = 0; k < mha_head_dim; k++){
            #pragma HLS PIPELINE II=1
                for (int i = 0; i < head_parallel; i++) {
                    block_A_loader[i].write(A[H * mha_head_dim + k][i]);
                    block_B_loader[i].write(weight_loaders[i].read());
                }
            }

            for (int i = 0; i < head_parallel; i++) {
            #pragma HLS UNROLL
                systolic_array_i8xi8_pack_1x2_1D<K_parallel, log2_mha_head_dim, is_uint_input>(
                    block_A_loader[i], block_B_loader[i], block_C_drainer[i], mha_head_dim
                );
            }

            output_scale_loop: for (int n = 0; n < K_parallel; n++) {    // L41
            #pragma HLS pipeline II=1
                hls::vector<ap_int<log2_mha_head_dim + 16>, head_parallel> outp_pack;
                for (int i = 0; i < head_parallel; i++) {
                    outp_pack[i] = block_C_drainer[i].read();
                }
                output_seq.write(outp_pack);
            }
        }
    }
    
}

template <typename T, int head_parallel, int K_parallel, int mha_head_num, int max_sum_seq_len=MAX_SUM_SEQ_LEN>
void dec_quant_a_discard_template(
    tapa::istream<hls::vector<T, head_parallel>>& input_seq,
    tapa::ostream<hls::vector<T, head_parallel>>& output_seq,
    int seq_len
){
    attn_head_loop: for (int H = 0; H < mha_head_num/head_parallel; H++){
        dec_io_discard<hls::vector<T, head_parallel>, ((max_sum_seq_len + K_parallel - 1) / K_parallel) * K_parallel, max_sum_seq_len>(
            input_seq, output_seq, ((seq_len + K_parallel - 1) / K_parallel) * K_parallel, seq_len
        );
    }
}


// template <typename T, int head_parallel, int max_hidden_dim=HIDDEN_DIM, int head_num=1>
// void dec_causal_mask_template(
//     tapa::istream<hls::vector<T, head_parallel>>& input_stream,
//     tapa::ostream<hls::vector<T, head_parallel>>& output_stream,
//     int io_hidden_dim,
//     int seq_len
// ){
//     attn_head_loop: for (int H = 0; H < head_num/head_parallel; H++){
//         max_loop: for (int k = 0; k < io_hidden_dim; k++) {
//         #pragma HLS loop_tripcount min=1 max=max_hidden_dim
//         #pragma HLS pipeline II=1
//             hls::vector<T, head_parallel> input_pack = input_stream.read();
//             hls::vector<T, head_parallel> output_pack;
//             for(int i = 0; i < head_parallel; i++){
//                 if(k < seq_len){
//                     output_pack[i] = input_pack[i];
//                 }
//                 else{
//                     output_pack[i] = -1e32;
//                 }
//             }
//             output_stream.write(output_pack);
//         }
//     }
// }
        


template <typename T, int head_parallel, int V_head_num, int V_head_dim=HEAD_DIM, int group_num=ATTN_GROUP_NUM>
void dec_V_s_cache_manager_static_template(
    tapa::mmap<hls::vector<T, head_parallel>> v_s_cache,
    tapa::ostream<hls::vector<T, head_parallel>>& output_v_s_stream,
    int block_id,
    int addr_bias = 0
){
    //V's scaling factor is unchanged during decoding stage
    head_loop_0: for (int H = 0; H < V_head_num/head_parallel; H++){
        int bias = addr_bias + (block_id * V_head_num/head_parallel + H) * V_head_dim;
        read_v_s_loop_0: for(int i = 0; i < V_head_dim; i++){
        #pragma HLS pipeline II=1
            output_v_s_stream.write(v_s_cache[bias + i]);
        }
    }

    head_loop_1: for (int H = 0; H < V_head_num/head_parallel; H++){
        attn_group_loop_1: for (int G = 0; G < group_num; G++){
            int bias = addr_bias + (block_id * V_head_num/head_parallel + H) * V_head_dim;
            read_v_s_loop_1: for(int i = 0; i < V_head_dim; i++){
            #pragma HLS pipeline II=1
                output_v_s_stream.write(v_s_cache[bias + i]);
            }
        }
    }
}


template <typename T, int head_parallel, int V_parallel, int V_hidden_dim=KV_HIDDEN_DIM, int decoder_layer_num=DECODER_LAYER_NUM>
void dec_V_cache_buffer_template(
    tapa::istream<hls::vector<T, head_parallel>>& input_v_stream,
    tapa::ostreams<hls::vector<T, V_parallel>, head_parallel>& output_v_streams,
    int block_id
){
    T v_reg[decoder_layer_num][head_parallel][V_hidden_dim/(head_parallel*V_parallel)][V_parallel];
    #pragma HLS ARRAY_PARTITION variable=v_reg type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=v_reg type=cyclic factor=V_parallel/2 dim=4

    reg_v_loop: for (int K = 0; K < V_hidden_dim/(head_parallel*V_parallel); K++) {
        for(int k = 0; k < V_parallel; k++){
        #pragma HLS pipeline II=1
            hls::vector<T, head_parallel> data_pack = input_v_stream.read();
            for (int i = 0; i < head_parallel; i++) {
                v_reg[block_id][i][K][k] = data_pack[i];
            }
        }
    }

    send_v_loop: for (int K = 0; K < V_hidden_dim/(head_parallel*V_parallel); K++) {
    #pragma HLS pipeline II=1
        for (int i = 0; i < head_parallel; i++) {
            hls::vector<T, V_parallel> V_cache_pack;
            for(int k = 0; k < V_parallel; k++){
                V_cache_pack[k] = v_reg[block_id][i][K][k] ;
            }
            output_v_streams[i].write(V_cache_pack);
        }
    }
}

template <typename T, int head_parallel, int V_parallel, int V_head_num, int V_head_dim=HEAD_DIM, int group_num=ATTN_GROUP_NUM, int max_sum_seq_len=MAX_SUM_SEQ_LEN>
void dec_V_cache_manager_template(
    tapa::istream<hls::vector<T, V_parallel>>& input_v_stream,
    tapa::mmap<hls::vector<T, V_parallel>>& v_cache,
    tapa::ostream<hls::vector<T, V_parallel>>& output_v_stream,
    int block_id,
    int seq_id,
    int addr_bias = 0
){
    input_block_loop: for (int N = 0; N < V_head_num/head_parallel * V_head_dim/V_parallel; N++){
    #pragma HLS pipeline II=1
        // store v
        int bias = addr_bias + (block_id * V_head_num/head_parallel * V_head_dim/V_parallel + N) * max_sum_seq_len + seq_id;
        v_cache[bias] = input_v_stream.read();
    }
    printf("Sum_seq_id %d: Block_id %d: V_cache_manager storing completed.\n", seq_id, block_id);
    
    
    attn_head_loop: for (int H = 0; H < V_head_num/head_parallel; H++){
        attn_group_loop: for (int G = 0; G < group_num; G++){
            attn_seq_loop: for(int h_block = 0; h_block < V_head_dim/V_parallel; h_block++){
                int bias = addr_bias + ((block_id * V_head_num/head_parallel + H) * V_head_dim/V_parallel + h_block) * max_sum_seq_len;
                read_v_loop: for(int i = 0; i <= seq_id; i++){
                #pragma HLS pipeline II=1
                    output_v_stream.write(v_cache[bias + i]);
                }
            }
        }
    }
    printf("Sum_seq_id %d: Block_id %d: V_cache_manager loading completed.\n", seq_id, block_id);
    
}


template <typename T, int head_parallel, int V_parallel_read, int V_parallel, int V_head_num, int V_head_dim=HEAD_DIM, int group_num=ATTN_GROUP_NUM, int max_sum_seq_len=MAX_SUM_SEQ_LEN>
void dec_V_cache_manager_discard_template(
    tapa::istream<hls::vector<T, V_parallel>>& input_v_stream,
    tapa::mmap<hls::vector<T, V_parallel_read>>& v_cache,
    tapa::ostream<hls::vector<T, V_parallel>>& output_v_stream,
    int block_id,
    int seq_id,
    int addr_bias = 0
){
    input_block_loop: for (int N = 0; N < V_head_num/head_parallel * V_head_dim/V_parallel; N++){
    #pragma HLS pipeline II=1
        // store v
        int bias = addr_bias + (block_id * V_head_num/head_parallel * V_head_dim/V_parallel + N) * max_sum_seq_len + seq_id;
        hls::vector<T, V_parallel> input_pack =  input_v_stream.read();
        hls::vector<T, V_parallel_read> v_pack;
        for(int j = 0; j < V_parallel; j++){
            v_pack[j] = input_pack[j];
        }
        v_cache[bias] = v_pack;
        
    }
    printf("Sum_seq_id %d: Block_id %d: V_cache_manager storing completed.\n", seq_id, block_id);
    
    
    attn_head_loop: for (int H = 0; H < V_head_num/head_parallel; H++){
        attn_group_loop: for (int G = 0; G < group_num; G++){
            attn_seq_loop: for(int h_block = 0; h_block < V_head_dim/V_parallel; h_block++){
                int bias = addr_bias + ((block_id * V_head_num/head_parallel + H) * V_head_dim/V_parallel + h_block) * max_sum_seq_len;
                read_v_loop: for(int i = 0; i <= seq_id; i++){
                #pragma HLS pipeline II=1
                    // output_v_stream.write(v_cache[bias + i]);
                    hls::vector<T, V_parallel_read> v_pack = v_cache[bias + i];
                    hls::vector<T, V_parallel> output_pack;
                    for(int j = 0; j < V_parallel; j++){
                        output_pack[j] = v_pack[j];
                    }
                    output_v_stream.write(output_pack);
                    
                }
            }
        }
    }
    printf("Sum_seq_id %d: Block_id %d: V_cache_manager loading completed.\n", seq_id, block_id);
    
}


template <int head_parallel, int V_parallel, int mha_head_num, int mha_head_dim = HEAD_DIM, int max_sum_seq_len = MAX_SUM_SEQ_LEN, int log2_max_sum_seq_len = log2_MAX_SUM_SEQ_LEN, bool is_uint_input=false>
void dec_MHA_i8xi8_axv_template(
    tapa::istream<hls::vector<ap_int<8>, head_parallel>>& input_seq,
    tapa::istreams<hls::vector<ap_int<8>, V_parallel>, head_parallel>& weight_loaders,
    tapa::ostream<hls::vector<ap_int<log2_max_sum_seq_len + 16>, head_parallel>>& output_seq,
    int seq_len
){
    hls::vector<ap_int<8>, head_parallel> A[max_sum_seq_len];
      
    hls::stream<ap_int<8>> block_A_loader[head_parallel];
    #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

    hls::stream<hls::vector<ap_int<8>, V_parallel>> block_B_loader[head_parallel];
    #pragma HLS BIND_STORAGE variable=block_B_loader type=fifo impl=srl

    hls::stream<ap_int<log2_max_sum_seq_len + 16>> block_C_drainer[head_parallel];
    #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl

    attn_head_loop: for (int H = 0; H < mha_head_num/head_parallel; H++){
        in_buf_loop: for (int k = 0; k < seq_len; k++) {    // L19
        #pragma HLS loop_tripcount min=1 max=max_sum_seq_len
        #pragma HLS pipeline II=1
            A[k] = input_seq.read();
        }

        v_weight_block_loop: for(int N = 0; N < mha_head_dim/V_parallel; N++){
        #pragma HLS DATAFLOW
            init_block_AB: for(int k = 0; k < seq_len; k++){
            #pragma HLS loop_tripcount min=1 max=max_sum_seq_len
            #pragma HLS PIPELINE II=1
                for (int i = 0; i < head_parallel; i++) {
                    block_A_loader[i].write(A[k][i]);
                    block_B_loader[i].write(weight_loaders[i].read());
                }
            }

            for (int i = 0; i < head_parallel; i++) {
            #pragma HLS UNROLL
                systolic_array_i8xi8_pack_1x2_1D<V_parallel, log2_max_sum_seq_len, is_uint_input>(
                    block_A_loader[i], block_B_loader[i], block_C_drainer[i], seq_len
                );
            }

            output_scale_loop: for (int n = 0; n < V_parallel; n++) {    // L41
            #pragma HLS pipeline II=1
                hls::vector<ap_int<log2_max_sum_seq_len + 16>, head_parallel> outp_pack;
                for (int i = 0; i < head_parallel; i++) {
                    outp_pack[i] = block_C_drainer[i].read();
                }
                output_seq.write(outp_pack);
            }
        }
    }
}




#endif