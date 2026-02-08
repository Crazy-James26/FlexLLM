#ifndef _LOGITS_H
#define _LOGITS_H
#include "config.h"

template <typename T, int top_k>
void sort_top_k(
    int max_logits_idx[top_k],
    T max_logits[top_k],
    int v_idx,
    T v
){
    int insert_pos = top_k;
    top_k_find_loop: for (int k = 0; k < top_k; k++) {
    #pragma HLS PIPELINE II=1
        if (v > max_logits[k] && insert_pos == top_k) {
            insert_pos = k;
        }
    }
    // insert and shift
    top_k_insert_loop: for (int k = top_k - 1; k >= 0; k--) {
    #pragma HLS PIPELINE II=2
        if(k > insert_pos){
            max_logits_idx[k] = max_logits_idx[k - 1];
            max_logits[k] = max_logits[k - 1];
        }
        else if(k == insert_pos){
            max_logits_idx[k] = v_idx;
            max_logits[k] = v;
        }
    }
}


template <typename T, int block_parallel_samp, int max_logits_num, int top_k, int max_hidden_dim=HIDDEN_DIM, int block_logits_num=HIDDEN_DIM, int inner_block_parallel=1, bool enable_softmax=true, bool enable_sub_max=true>
void dec_Logits_Max_K_Layer(
    tapa::istream<hls::vector<T, block_parallel_samp>>& logits_stream,
    tapa::ostream<hls::vector<T, 2>>& max_logits_stream,
    int logits_num = max_logits_num
){
    hls::vector<T, block_parallel_samp> logits_buffer[block_logits_num/block_parallel_samp];
    #pragma HLS ARRAY_PARTITION variable=logits_buffer type=cyclic factor=inner_block_parallel dim=1
    
    int max_logits_idx[block_parallel_samp][inner_block_parallel][top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits_idx type=complete
    T max_logits[block_parallel_samp][inner_block_parallel][top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits type=complete

    int max_logits_idx_final[top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits_idx_final complete
    T max_logits_final[top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits_final complete

    T max_logits_final_exp[top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits_final_exp complete

    init_loop: for (int k = 0; k < top_k; k++) {
    #pragma HLS unroll
        for(int i = 0; i < block_parallel_samp; i++){
        #pragma HLS unroll
            for(int j = 0;  j < inner_block_parallel; j++){
            #pragma HLS unroll
                max_logits_idx[i][j][k] = -1;
                max_logits[i][j][k] = -1e32;
            }
        }
        max_logits_idx_final[k] = -1;
        max_logits_final[k] = -1e32;
    }


    io_block_loop: for (int M = 0; M < (logits_num + block_logits_num - 1)/block_logits_num; M++) {
    #pragma HLS loop_tripcount min=1 max=(max_logits_num + block_logits_num - 1)/block_logits_num
        buffer_loop: for(int m = 0; m < block_logits_num/block_parallel_samp; m++){
        #pragma HLS PIPELINE II=1
            if(M * block_logits_num/block_parallel_samp + m < logits_num/block_parallel_samp){
                logits_buffer[m] = logits_stream.read();
            }
            else{
                logits_buffer[m] = -1e32;
            }
        }

        lane_top_k_loop: for(int m = 0; m < block_logits_num/(block_parallel_samp * inner_block_parallel); m++){
            // maintain Top-K in parallel for each lane
            for (int i = 0; i < block_parallel_samp; i++) {
            #pragma HLS unroll
                for (int j = 0; j < inner_block_parallel; j++) {
                #pragma HLS unroll
                    T v = logits_buffer[m * inner_block_parallel + j][i];
                    int v_idx = i * (logits_num/block_parallel_samp) + M * block_logits_num/block_parallel_samp + m * inner_block_parallel + j;
                    sort_top_k<T, top_k>(max_logits_idx[i][j], max_logits[i][j], v_idx, v);
                }
            }
        }
    }

    // 4) compute final global Top-K over all candidates
    merge_loop: for (int K = 0; K < top_k; K++) {
        top_k_block_loop_i: for (int i = 0; i < block_parallel_samp; i++) {
            top_k_block_loop_j: for (int j = 0; j < inner_block_parallel; j++) {
                T val = max_logits[i][j][K];
                int val_idx = max_logits_idx[i][j][K];
                int pos = top_k;
                sort_top_k<T, top_k>(max_logits_idx_final, max_logits_final, val_idx, val);
            }
        }
    }

    for(int i = 0; i < top_k; i++) {
        printf("max_logits_idx_final[%d] = %d, max_logits_final[%d] = %f\n", i, max_logits_idx_final[i], i, max_logits_final[i]);
    }

    // 5) apply softmax if enabled
    if (enable_softmax) {
        T sum = 0;
        exp_sum_loop: for (int k = 0; k < top_k; k++) {
        #pragma HLS PIPELINE II = 1
            T temp;
            if(enable_sub_max) 
                temp = exp(max_logits_final[k] - max_logits_final[0]);
            else
                temp = exp(max_logits_final[k]);
            max_logits_final_exp[k] = temp;
            sum += temp;
        }
        exp_scale_loop: for (int k = 0; k < top_k; k++) {
        #pragma HLS loop_tripcount max=top_k
        #pragma HLS PIPELINE II = 1
            max_logits_final_exp[k] /= sum;
        }
    }


    // 5) pack and write out the final Top-K
    hls::vector<T, 2> out_pack;
    write_loop: for (int k = 0; k < top_k; k++) {
    #pragma HLS PIPELINE II = 1
        out_pack[0] = max_logits_idx_final[k];
        out_pack[1] = max_logits_final_exp[k];
        max_logits_stream.write(out_pack);
    }
}


template <typename T, int block_parallel_samp, int max_logits_num, int top_k, int block_parallel_embed, int max_hidden_dim=HIDDEN_DIM, int block_logits_num=HIDDEN_DIM, int inner_block_parallel=1, bool enable_sub_max=true>
void dec_Sampling_Embedding_Layer(
    tapa::istream<hls::vector<T, block_parallel_samp>>& logits_stream,
    tapa::mmap<hls::vector<T, block_parallel_embed>> vocab_lib,
    tapa::ostream<hls::vector<T, block_parallel_embed>>& new_embedding_stream,
    T rand_seed, 
    int & sampled_idx,
    int logits_num = max_logits_num,
    int io_hidden_dim = max_hidden_dim
){
    hls::vector<T, block_parallel_samp> logits_buffer[block_logits_num/block_parallel_samp];
    #pragma HLS ARRAY_PARTITION variable=logits_buffer type=cyclic factor=inner_block_parallel dim=1
    
    int max_logits_idx[block_parallel_samp][inner_block_parallel][top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits_idx type=complete
    #pragma HLS bind_storage variable=max_logits_idx type=ram_s2p impl=LUTRAM
    T max_logits[block_parallel_samp][inner_block_parallel][top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits type=complete
    #pragma HLS bind_storage variable=max_logits type=ram_s2p impl=LUTRAM

    int max_logits_idx_final[top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits_idx_final complete
    T max_logits_final[top_k];
    #pragma HLS ARRAY_PARTITION variable=max_logits_final complete

    T max_vocab_probs[top_k];
    #pragma HLS ARRAY_PARTITION variable=max_vocab_probs complete

    init_loop: for (int k = 0; k < top_k; k++) {
    #pragma HLS unroll
        for(int i = 0; i < block_parallel_samp; i++){
        #pragma HLS unroll
            for(int j = 0;  j < inner_block_parallel; j++){
            #pragma HLS unroll
                max_logits_idx[i][j][k] = -1;
                max_logits[i][j][k] = -1e32;
            }
        }
        max_logits_idx_final[k] = -1;
        max_logits_final[k] = -1e32;
    }


    io_block_loop: for (int M = 0; M < (logits_num + block_logits_num - 1)/block_logits_num; M++) {
    #pragma HLS loop_tripcount min=1 max=(max_logits_num + block_logits_num - 1)/block_logits_num
        buffer_loop: for(int m = 0; m < block_logits_num/block_parallel_samp; m++){
        #pragma HLS PIPELINE II=1
            if(M * block_logits_num/block_parallel_samp + m < logits_num/block_parallel_samp){
                logits_buffer[m] = logits_stream.read();
            }
            else{
                logits_buffer[m] = -1e32;
            }
        }

        lane_top_k_loop: for(int m = 0; m < block_logits_num/(block_parallel_samp * inner_block_parallel); m++){
            // maintain Top-K in parallel for each lane
            for (int i = 0; i < block_parallel_samp; i++) {
            #pragma HLS unroll
                for (int j = 0; j < inner_block_parallel; j++) {
                #pragma HLS unroll
                    T v = logits_buffer[m * inner_block_parallel + j][i];
                    int v_idx = i * (logits_num/block_parallel_samp) + M * block_logits_num/block_parallel_samp + m * inner_block_parallel + j;
                    sort_top_k<T, top_k>(max_logits_idx[i][j], max_logits[i][j], v_idx, v);
                }
            }
            
        }
    }

    // 4) compute final global Top-K over all candidates
    merge_loop: for (int K = 0; K < top_k; K++) {
        top_k_block_loop_i: for (int i = 0; i < block_parallel_samp; i++) {
            top_k_block_loop_j: for (int j = 0; j < inner_block_parallel; j++) {
                T val = max_logits[i][j][K];
                int val_idx = max_logits_idx[i][j][K];
                sort_top_k<T, top_k>(max_logits_idx_final, max_logits_final, val_idx, val);
            }
        }
    }

    for(int i = 0; i < top_k; i++) {
        printf("max_logits_idx_final[%d] = %d, max_logits_final[%d] = %f\n", i, max_logits_idx_final[i], i, max_logits_final[i]);
    }

    // 5) apply softmax
    T sum_partial[2] = {0};
    exp_sum_loop: for (int k = 0; k < top_k; k++) {
    #pragma HLS PIPELINE II = 1
        T temp;
        if(enable_sub_max) 
            temp = exp(max_logits_final[k] - max_logits_final[0]);
        else
            temp = exp(max_logits_final[k]);
        max_vocab_probs[k] = temp;
        sum_partial[k % 2] += temp;
    }
    T sum = sum_partial[0] + sum_partial[1];

    exp_scale_loop: for (int k = 0; k < top_k; k++) {
    #pragma HLS PIPELINE II = 1
        max_vocab_probs[k] /= sum;
    }
    
    // 5) sample from Top-K according to probabilities
    // Build CDF
    T cdf[top_k];
    #pragma HLS ARRAY_PARTITION variable=cdf complete

    cdf[0] = max_vocab_probs[0];
    printf("k %d, prob %f, cdf %f\n", 0, (float)max_vocab_probs[0], cdf[0]);
    build_cdf: for (int k = 1; k < top_k; ++k) {
    #pragma HLS PIPELINE II = 2
        cdf[k] = max_vocab_probs[k] + cdf[k - 1];
        printf("k %d, prob %f, cdf %f\n", k, (float)max_vocab_probs[k], cdf[k]);
    }
    printf("rand_seed %f\n", (float)rand_seed);

    int selected_k;
    // Select smallest k with r < cdf[k]
    selected_k = top_k - 1;
    choose_k: for (int k = 0; k < top_k; ++k) {
        // priority-encode the first crossing
        if ((rand_seed < cdf[k]) && (selected_k == top_k - 1)) selected_k = k;
    }
    
    sampled_idx = max_logits_idx_final[selected_k];
    // sampled_idx = 271;

    // 6) retrieve embedding from vocab_lib and write to io_mmap
    dec_input_loader<T, block_parallel_embed, max_hidden_dim>(
        vocab_lib, new_embedding_stream, 0, sampled_idx, io_hidden_dim
    );
    printf("selected k %d, sampled token %d\n", selected_k, sampled_idx);
}



#endif