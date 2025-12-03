#ifndef _HMT_H_
#define _HMT_H_

#include "config.h"
#include "Linear_Layer.h"
#include "Softmax.h"

template <typename T, int hmt_t_block_parallel, int token_parallel, int hidden_dim=HIDDEN_DIM, int hmt_seg_len=HMT_SEG_LEN, int hmt_sum_seg_len=HMT_SUM_SEG_LEN, int hmt_rec_seg_len=HMT_REC_SEG_LEN>
void hmt_segment_loader_sync_template(
    tapa::mmap<hls::vector<T, hmt_t_block_parallel>> hmt_io_mmap, // the first embedding vector is Tn
    tapa::mmap<hls::vector<T, token_parallel>> pref_io_mmap,
    tapa::istream<bool>& hmt_stage_01_finish_stream,
    tapa::ostream<bool>& hmt_stage_01_ready_stream,
    tapa::ostream<hls::vector<T, hmt_t_block_parallel>>& hmt_Sn_stream, 
    tapa::istream<hls::vector<T, hmt_t_block_parallel>>& hmt_Pn_stream, 
    tapa::ostream<hls::vector<T, hmt_t_block_parallel>>& hmt_Mn_stream,
    tapa::ostream<int>& seg_len_stream,
    int seq_len,
    int seg_num
){
    T token_buffer[token_parallel][hidden_dim];
    #pragma HLS ARRAY_PARTITION variable=token_buffer dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=token_buffer dim=2 type=block factor=hmt_t_block_parallel
    
    T hmt_Sn_Mn_buffer[hidden_dim];
    #pragma HLS ARRAY_PARTITION variable=hmt_Sn_Mn_buffer type=block factor=hmt_t_block_parallel

    hls::vector<T, hmt_t_block_parallel> hmt_Pn_buffer[hidden_dim/hmt_t_block_parallel];

    init_buffer_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
    #pragma HLS PIPELINE II=1
        for(int t = 0; t < hmt_t_block_parallel; t++){
            hmt_Sn_Mn_buffer[t * (hidden_dim/hmt_t_block_parallel) + k] = 0;
            hmt_Pn_buffer[k][t] = 0;
        }
    }
    
    seg_loop: for(int seg_id = 0; seg_id < seg_num; seg_id++){
        int current_seg_len = seg_id < seg_num - 1 ? hmt_seg_len : (seq_len - seg_id * hmt_seg_len);
        int current_sum_seg_len = current_seg_len < hmt_sum_seg_len ? current_seg_len : hmt_sum_seg_len;
        int token_seg_offset = seg_id * hmt_seg_len;

        int stage_0_seq_len = (1 + current_sum_seg_len + 1 + token_parallel - 1)/token_parallel * token_parallel; 
        int stage_1_seq_len = (1 + hmt_rec_seg_len + current_seg_len + 1 + token_parallel - 1)/token_parallel * token_parallel;

        cout << "hmt segment id: " << seg_id << " stage_0_seq_len: " << stage_0_seq_len << " stage_1_seq_len: " << stage_1_seq_len << endl;

        //stage 0: seq_len = 1 + hmt_sum_seg_len + 1 = 1 + 495 + 1 = 497
        if(seg_id > 0){
            // seg_len_stream.write(stage_0_seq_len);
            while(! seg_len_stream.try_write(stage_0_seq_len)) {};
            for(int i = 0; i < (1 + current_sum_seg_len + 1 + token_parallel - 1)/token_parallel; i++){
                for(int j = 0; j < token_parallel; j++){
                    
                    int token_seg_id = i * token_parallel + j - 1;
                    
                    if(token_seg_id == -1 || token_seg_id == current_sum_seg_len){
                        stage_0_read_Sn_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
                        #pragma HLS PIPELINE II=1
                            hls::vector<T, hmt_t_block_parallel> temp_vec = hmt_io_mmap[k];
                            for(int t = 0; t < hmt_t_block_parallel; t++){
                                token_buffer[j][t * (hidden_dim/hmt_t_block_parallel) + k] = temp_vec[t];
                            }
                        }
                    }
                    else if(token_seg_id < current_sum_seg_len){
                        stage_0_read_token_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
                        #pragma HLS PIPELINE II=1
                            hls::vector<T, hmt_t_block_parallel> temp_vec = hmt_io_mmap[(1 + token_seg_offset + token_seg_id) * (hidden_dim/hmt_t_block_parallel) + k];
                            for(int t = 0; t < hmt_t_block_parallel; t++){
                                token_buffer[j][t * (hidden_dim/hmt_t_block_parallel) + k] = temp_vec[t];
                            }
                        }
                    }
                    else{
                        stage_0_read_pad_zero_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
                        #pragma HLS PIPELINE II=1
                            for(int t = 0; t < hmt_t_block_parallel; t++){
                                token_buffer[j][t * (hidden_dim/hmt_t_block_parallel) + k] = 0;
                            }
                        }
                    }
                }

                stage_0_write_pref_token_loop: for(int k = 0; k < hidden_dim; k++){
                #pragma HLS PIPELINE II=1
                    hls::vector<T, token_parallel> temp_vec;
                    for(int t = 0; t < token_parallel; t++){
                        temp_vec[t] = token_buffer[t][k];
                    }
                    pref_io_mmap[i * hidden_dim + k] = temp_vec;
                }
            }
            while(! hmt_stage_01_ready_stream.try_write(true)) {};
            cout << "hmt segment id: " << seg_id << " stage 0 ready!" << endl;

            // when stage 0 is done, read Sn from pref_io_mmap to do query calculation
            bool stage_0_finish = hmt_stage_01_finish_stream.read();
            
            read_Sn_loop: for(int k = 0; k < hidden_dim; k++){
            #pragma HLS PIPELINE II=1
                hls::vector<T, token_parallel> temp_vec = pref_io_mmap[(1 + current_sum_seg_len)/token_parallel * hidden_dim + k];
                hmt_Sn_Mn_buffer[k] = temp_vec[(1 + current_sum_seg_len) % token_parallel];
            }

            send_Sn_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
            #pragma HLS PIPELINE II=1
                hls::vector<T, hmt_t_block_parallel> temp_vec;
                for(int t = 0; t < hmt_t_block_parallel; t++){
                    temp_vec[t] = hmt_Sn_Mn_buffer[t * (hidden_dim/hmt_t_block_parallel) + k];
                }
                hmt_Sn_stream.write(temp_vec);
            }

            // receive Pn from cross attention and store in hmt_Pn_buffer
            receive_Pn_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
            #pragma HLS PIPELINE II=1
                hls::vector<T, hmt_t_block_parallel> temp_vec = hmt_Pn_stream.read();
                hmt_Pn_buffer[k] = temp_vec;
            }
            cout << "hmt segment id: " << seg_id << " stage 0 finished!" << endl;
        }

        
        //stage 1: : seq_len = 1 + hmt_rec_seg_len + hmt_seg_len + 1 = 1 + 32 + 990 + 1 = 1024
        // seg_len_stream.write(stage_1_seq_len);
        while(! seg_len_stream.try_write(stage_1_seq_len)) {};
        for(int i = 0; i < (1 + hmt_rec_seg_len + current_seg_len + 1 + token_parallel - 1)/token_parallel; i++){
            for(int j = 0; j < token_parallel; j++){
                
                int token_seg_id = i * token_parallel + j - 1 - hmt_rec_seg_len;

                if(token_seg_id == - 1 - hmt_rec_seg_len || (token_seg_id == current_seg_len && seg_id < seg_num - 1)){
                    stage_1_read_Sn_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
                    #pragma HLS PIPELINE II=1
                        hls::vector<T, hmt_t_block_parallel> temp_vec = hmt_Pn_buffer[k];
                        for(int t = 0; t < hmt_t_block_parallel; t++){
                            token_buffer[j][t * (hidden_dim/hmt_t_block_parallel) + k] = temp_vec[t];
                        }
                    }
                }
                else if(token_seg_id < current_seg_len && token_seg_offset + token_seg_id >= 0){
                    stage_1_read_token_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
                    #pragma HLS PIPELINE II=1
                        hls::vector<T, hmt_t_block_parallel> temp_vec = hmt_io_mmap[(1 + token_seg_offset + token_seg_id) * (hidden_dim/hmt_t_block_parallel) + k];
                        for(int t = 0; t < hmt_t_block_parallel; t++){
                            token_buffer[j][t * (hidden_dim/hmt_t_block_parallel) + k] = temp_vec[t];
                        }
                    }
                }
                else{
                    stage_1_read_pad_zero_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
                    #pragma HLS PIPELINE II=1
                        for(int t = 0; t < hmt_t_block_parallel; t++){
                            token_buffer[j][t * (hidden_dim/hmt_t_block_parallel) + k] = 0;
                        }
                    }
                }
            }

            stage_1_write_pref_token_loop: for(int k = 0; k < hidden_dim; k++){
            #pragma HLS PIPELINE II=1
                hls::vector<T, token_parallel> temp_vec;
                for(int t = 0; t < token_parallel; t++){
                    temp_vec[t] = token_buffer[t][k];
                }
                pref_io_mmap[i * hidden_dim + k] = temp_vec;
            }
        }
        while(! hmt_stage_01_ready_stream.try_write(true)) {};
        cout << "hmt segment id: " << seg_id << " stage 1 ready!" << endl;
        // when stage 1 is done, read Mn from pref_io_mmap to do key calculation and store in Memory Cache
        
        bool stage_1_finish = hmt_stage_01_finish_stream.read();
        if(seg_id < seg_num - 1){
            read_Mn_loop: for(int k = 0; k < hidden_dim; k++){
            #pragma HLS PIPELINE II=1
                hls::vector<T, token_parallel> temp_vec = pref_io_mmap[(1 + hmt_rec_seg_len + current_seg_len)/token_parallel * hidden_dim + k];
                hmt_Sn_Mn_buffer[k] = temp_vec[(1 + hmt_rec_seg_len + current_seg_len) % token_parallel];
            }   

            send_Mn_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
            #pragma HLS PIPELINE II=1
                hls::vector<T, hmt_t_block_parallel> temp_vec;
                for(int t = 0; t < hmt_t_block_parallel; t++){
                    temp_vec[t] = hmt_Sn_Mn_buffer[t * (hidden_dim/hmt_t_block_parallel) + k];
                }
                hmt_Mn_stream.write(temp_vec);
            }
        }
        cout << "hmt segment id: " << seg_id << " stage 1 finished!" << endl;
    }
    // seg_len_stream.write(0); // end signal
    while(! seg_len_stream.try_write(0)) {};
}


template <typename T, int hmt_t_block_parallel, int weight_parallel, int max_hidden_dim = HIDDEN_DIM, int max_mem_num=MEM_NUM, int hmt_seg_len=HMT_SEG_LEN>
void hmt_weight_loader_qk_attn_template(
    tapa::mmap<hls::vector<T, weight_parallel>> wq_wk_mmap,
    tapa::istream<hls::vector<T, weight_parallel>>& load_Kn_stream,
    tapa::istream<hls::vector<T, weight_parallel>>& load_Mn_stream,
    tapa::ostream<hls::vector<T, weight_parallel>>& w_qk_attn_stream,
    int seg_num,
    int hidden_dim = max_hidden_dim, 
    int mem_num = max_mem_num,
    int addr_bias = 0
){
    seg_loop: for(int seg_id = 0; seg_id < seg_num; seg_id++){
        // calculate query for the Sn of current segment
        if(seg_id > 0){
            dec_weight_loader<T, hmt_t_block_parallel, weight_parallel, max_hidden_dim, max_hidden_dim>(
                wq_wk_mmap, w_qk_attn_stream, 0, hidden_dim, hidden_dim, addr_bias
            ); // for Qn
            
            for(int i = 0; i < hidden_dim/(hmt_t_block_parallel*weight_parallel)*mem_num; i++){
            #pragma HLS PIPELINE II=1
                w_qk_attn_stream.write(load_Kn_stream.read());
            } // for Qn * Kn^T

            for(int i = 0; i < mem_num/(hmt_t_block_parallel*weight_parallel)*hidden_dim; i++){
            #pragma HLS PIPELINE II=1
                w_qk_attn_stream.write(load_Mn_stream.read());
            } // for An * Mn
        }

        // calculate key for the Mn of current segment
        if(seg_id < seg_num - 1){
            dec_weight_loader<T, hmt_t_block_parallel, weight_parallel, max_hidden_dim, max_hidden_dim>(
                wq_wk_mmap, w_qk_attn_stream, 0, hidden_dim, hidden_dim, addr_bias + hidden_dim/(hmt_t_block_parallel*weight_parallel)*hidden_dim
            );
        }
    }
}


template <typename T, int hmt_t_block_parallel, int max_hidden_dim = HIDDEN_DIM, int max_mem_num=MEM_NUM, int hmt_seg_len=HMT_SEG_LEN>
void hmt_Linear_Layer_fp32xfp32_qk_attn_input_merger_template(
    tapa::istream<hls::vector<T, hmt_t_block_parallel>>& hmt_Sn_stream, 
    tapa::istream<hls::vector<T, hmt_t_block_parallel>>& hmt_Qn_stream, 
    tapa::istream<hls::vector<T, hmt_t_block_parallel>>& hmt_sft_An_stream,
    tapa::istream<hls::vector<T, hmt_t_block_parallel>>& hmt_Mn_stream,
    tapa::ostream<hls::vector<T, hmt_t_block_parallel>>& hmt_Mn_cache_stream,
    tapa::ostream<hls::vector<T, hmt_t_block_parallel>>& hmt_ll_input_stream,
    int seg_num,
    int hidden_dim = max_hidden_dim,
    int mem_num = max_mem_num
){
    seg_loop: for(int seg_id = 0; seg_id < seg_num; seg_id++){
        if(seg_id > 0){
            send_Sn_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
            #pragma HLS PIPELINE II=1
                hmt_ll_input_stream.write(hmt_Sn_stream.read());
            }

            send_Qn_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
            #pragma HLS PIPELINE II=1
                hmt_ll_input_stream.write(hmt_Qn_stream.read());
            }

            send_sft_An_loop: for(int k = 0; k < mem_num/hmt_t_block_parallel; k++){
            #pragma HLS PIPELINE II=1
                hmt_ll_input_stream.write(hmt_sft_An_stream.read());
            }
        }
        if(seg_id < seg_num - 1){
            send_Mn_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
            #pragma HLS PIPELINE II=1
                hls::vector<T, hmt_t_block_parallel> temp_vec = hmt_Mn_stream.read();
                hmt_Mn_cache_stream.write(temp_vec);
                hmt_ll_input_stream.write(temp_vec);
            }
        }
    }
}


template <int hmt_t_block_parallel, int weight_parallel, int max_hidden_dim = HIDDEN_DIM, int max_mem_num=MEM_NUM, int hmt_seg_len=HMT_SEG_LEN>
void hmt_Linear_Layer_fp32xfp32_qk_attn_template(
    tapa::istream<hls::vector<float, hmt_t_block_parallel>>& input_seq,
    tapa::istreams<hls::vector<float, weight_parallel>, hmt_t_block_parallel>& weight_loaders,
    tapa::ostream<hls::vector<float, hmt_t_block_parallel>>& output_seq,
    int seg_num,
    int hidden_dim = max_hidden_dim, 
    int mem_num = max_mem_num
){
    seg_loop: for(int seg_id = 0; seg_id < seg_num; seg_id++){
        // calculate query for the Sn of current segment
        if(seg_id > 0){
            dec_Linear_Layer_fp32xfp32<hmt_t_block_parallel, weight_parallel, max_hidden_dim, max_hidden_dim>(
                input_seq, weight_loaders, output_seq, hidden_dim, hidden_dim
            ); // for Qn
            cout << "hmt segment id: " << seg_id << " stage 0 Qn done!" << endl;

            dec_Linear_Layer_fp32xfp32<hmt_t_block_parallel, weight_parallel, max_hidden_dim, max_hidden_dim>(
                input_seq, weight_loaders, output_seq, hidden_dim, mem_num
            ); // for Qn * Kn^T
            cout << "hmt segment id: " << seg_id << " stage 0 Qn*Kn done!" << endl;

            dec_Linear_Layer_fp32xfp32<hmt_t_block_parallel, weight_parallel, max_hidden_dim, max_hidden_dim>(
                input_seq, weight_loaders, output_seq, mem_num, hidden_dim
            ); // for An * Mn
            cout << "hmt segment id: " << seg_id << " stage 0 An*Mn done!" << endl;

        }

        // calculate key for the Mn of current segment
        if(seg_id < seg_num - 1){
            dec_Linear_Layer_fp32xfp32<hmt_t_block_parallel, weight_parallel, max_hidden_dim, max_hidden_dim>(
                input_seq, weight_loaders, output_seq, hidden_dim, hidden_dim
            );
            cout << "hmt segment id: " << seg_id << " stage 1 Kn done!" << endl;
        }
    }
}


template <typename T, int hmt_t_block_parallel, int max_hidden_dim = HIDDEN_DIM, int max_mem_num=MEM_NUM, int hmt_seg_len=HMT_SEG_LEN>
void hmt_Linear_Layer_fp32xfp32_qk_attn_output_merger_template(
    tapa::istream<hls::vector<T, hmt_t_block_parallel>>& hmt_ll_output_stream, 
    tapa::ostream<hls::vector<T, hmt_t_block_parallel>>& hmt_Qn_stream,
    tapa::ostream<hls::vector<T, hmt_t_block_parallel>>& hmt_An_stream,
    tapa::ostream<hls::vector<T, hmt_t_block_parallel>>& hmt_Pn_stream,
    tapa::ostream<hls::vector<T, hmt_t_block_parallel>>& hmt_Kn_stream,
    int seg_num,
    int hidden_dim = max_hidden_dim,
    int mem_num = max_mem_num
){
    seg_loop: for(int seg_id = 0; seg_id < seg_num; seg_id++){
        if(seg_id > 0){
            receive_Qn_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
            #pragma HLS PIPELINE II=1
                hmt_Qn_stream.write(hmt_ll_output_stream.read());
            }

            receive_An_loop: for(int k = 0; k < mem_num/hmt_t_block_parallel; k++){
            #pragma HLS PIPELINE II=1
                hmt_An_stream.write(hmt_ll_output_stream.read());
            }

            receive_Pn_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
            #pragma HLS PIPELINE II=1
                hmt_Pn_stream.write(hmt_ll_output_stream.read());
            }
        }
        if(seg_id < seg_num - 1){
            receive_Kn_loop: for(int k = 0; k < hidden_dim/hmt_t_block_parallel; k++){
            #pragma HLS PIPELINE II=1
                hmt_Kn_stream.write(hmt_ll_output_stream.read());
            }
        }
    }
}



template <typename T, int hmt_t_block_parallel, int w_attn_parallel, int hidden_dim=HIDDEN_DIM, int mem_num=MEM_NUM, int hmt_seg_len=HMT_SEG_LEN>
void hmt_memory_cache_manager_template(
    tapa::istream<hls::vector<T, hmt_t_block_parallel>>& input_Mn_stream,
    tapa::ostreams<hls::vector<T, w_attn_parallel>, hmt_t_block_parallel>& output_Mn_streams,
    int seg_num
){
    T mem_cache[mem_num][hmt_t_block_parallel][hidden_dim/hmt_t_block_parallel];
    #pragma HLS bind_storage variable=mem_cache type=ram_2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=mem_cache type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=mem_cache type=cyclic factor=w_attn_parallel/2 dim=3

    init_mem_loop: for(int M = 0; M < HIDDEN_DIM/(hmt_t_block_parallel * w_attn_parallel); M++){
        for(int k = 0; k < mem_num; k++){
            #pragma HLS PIPELINE II=1
            for(int t = 0; t < hmt_t_block_parallel; t++){
                for(int m = 0; m < w_attn_parallel; m++){
                    mem_cache[k][t][M * w_attn_parallel + m] = 0;
                }
            }
        }
    }

    seg_loop: for(int seg_id = 0; seg_id < seg_num; seg_id++){
        // send All Mn to do attention calculation
        if(seg_id > 0){
            load_mem_block_loop: for(int M = 0; M < HIDDEN_DIM/(hmt_t_block_parallel * w_attn_parallel); M++){
                load_mem_inner_loop: for(int k = 0; k < mem_num; k++){
                #pragma HLS PIPELINE II=1
                    for(int t = 0; t < hmt_t_block_parallel; t++){
                        hls::vector<T, w_attn_parallel> mem_pack;
                        for(int m = 0; m < w_attn_parallel; m++){
                            mem_pack[m] = mem_cache[k][t][M * w_attn_parallel + m];
                            // mem_pack[m] = 0; // for test
                        }
                        output_Mn_streams[t].write(mem_pack);
                    }
                }
            }
        }

        // store Mn of current segment to memory cache
        if(seg_id < seg_num - 1){
            receive_mem_loop: for(int i = 0; i < hidden_dim/hmt_t_block_parallel; i++){
            #pragma HLS PIPELINE II=1
                hls::vector<T, hmt_t_block_parallel> mem_pack = input_Mn_stream.read();
                for(int t = 0; t < hmt_t_block_parallel; t++){
                    mem_cache[seg_id % mem_num][t][i] = mem_pack[t];
                }
            }
        }
    }
}



template <typename T, int hmt_t_block_parallel, int w_attn_parallel, int hidden_dim=HIDDEN_DIM, int mem_num=MEM_NUM, int hmt_seg_len=HMT_SEG_LEN>
void hmt_k_mem_cache_manager_template(
    tapa::istream<hls::vector<T, hmt_t_block_parallel>>& input_Kn_stream,
    tapa::ostreams<hls::vector<T, w_attn_parallel>, hmt_t_block_parallel>& output_Kn_streams,
    int seg_num
){
    T k_mem_cache[hmt_t_block_parallel][mem_num/hmt_t_block_parallel][hidden_dim];
    #pragma HLS bind_storage variable=k_mem_cache type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=k_mem_cache type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=k_mem_cache type=cyclic factor=w_attn_parallel/2 dim=2

    T k_mem_buffer[hidden_dim];
    #pragma HLS ARRAY_PARTITION variable=k_mem_buffer type=block factor=hmt_t_block_parallel

    init_k_mem_loop: for(int N = 0; N < mem_num/(hmt_t_block_parallel * w_attn_parallel); N++){
        for(int k = 0; k < hidden_dim; k++){
        #pragma HLS PIPELINE II=1
            for(int t = 0; t < hmt_t_block_parallel; t++){
                for(int n = 0; n < w_attn_parallel; n++){
                    k_mem_cache[t][N * w_attn_parallel + n][k] = 0;
                }
            }
        }
    }

    seg_loop: for(int seg_id = 0; seg_id < seg_num; seg_id++){
        // send All Kn to do attention calculation
        if(seg_id > 0){
            load_k_mem_block_loop: for(int N = 0; N < mem_num/(hmt_t_block_parallel * w_attn_parallel); N++){
                load_k_mem_inner_loop: for(int k = 0; k < hidden_dim; k++){
                #pragma HLS PIPELINE II=1
                    for(int t = 0; t < hmt_t_block_parallel; t++){
                        hls::vector<T, w_attn_parallel> k_pack;
                        for(int n = 0; n < w_attn_parallel; n++){
                            k_pack[n] = k_mem_cache[t][N * w_attn_parallel + n][k];
                            // k_pack[n] = 1.0; // for test
                        }
                        output_Kn_streams[t].write(k_pack);
                    }
                }
            }
        }

        // store Kn of current segment to memory cache
        if(seg_id < seg_num - 1){
            receive_k_mem_loop: for(int i = 0; i < hidden_dim/hmt_t_block_parallel; i++){
            #pragma HLS PIPELINE II=1
                hls::vector<T, hmt_t_block_parallel> k_pack = input_Kn_stream.read();
                for(int t = 0; t < hmt_t_block_parallel; t++){
                    k_mem_buffer[t * (hidden_dim/hmt_t_block_parallel) + i] = k_pack[t];
                }
            }

            int t_idx = (seg_id % mem_num) / (mem_num/hmt_t_block_parallel);
            int n_idx = (seg_id % mem_num) % (mem_num/hmt_t_block_parallel);
            store_k_mem_loop: for(int i = 0; i < hidden_dim; i++){
            #pragma HLS PIPELINE II=1
                k_mem_cache[t_idx][n_idx][i] = k_mem_buffer[i];
            }
        }
    }
}


template <typename T, int hmt_t_block_parallel, int mem_num=MEM_NUM, int hmt_seg_len=HMT_SEG_LEN, bool enable_scale=false, bool enable_sub_max=true>
void hmt_attn_softmax_template(
    tapa::istream<hls::vector<T, hmt_t_block_parallel>>& input_stream,
    tapa::ostream<hls::vector<T, hmt_t_block_parallel>>& output_stream,
    int seg_num,
    int io_dim = mem_num,
    const T scale_factor = 1.0
){
    seg_loop: for(int seg_id = 0; seg_id < seg_num; seg_id++){
        if(seg_id > 0){
            dec_Softmax<T, hmt_t_block_parallel, mem_num, enable_scale, enable_sub_max>(
                input_stream, output_stream, io_dim, scale_factor
            );
        }
    }
}


template <typename T, int token_parallel, int hidden_dim=HIDDEN_DIM, int hmt_seg_len=HMT_SEG_LEN, int hmt_sum_seg_len=HMT_SUM_SEG_LEN, int hmt_rec_seg_len=HMT_REC_SEG_LEN, int max_pre_seq_len=MAX_PRE_SEQ_LEN>
void hmt_dummy_prefilling_template(
    tapa::mmap<hls::vector<T, token_parallel>> pref_io_mmap,
    tapa::istream<bool>& hmt_stage_01_ready_stream,
    tapa::ostream<bool>& hmt_stage_01_finish_stream,
    tapa::istream<int>& seg_len_stream
){
    hls::vector<T, token_parallel> buffer[hidden_dim];
    
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        bool stage_01_ready = hmt_stage_01_ready_stream.read();
        seg_block_loop: for(int i = 0; i < seg_len/token_parallel; i++){
            read_loop: for(int k = 0; k < hidden_dim; k++){
            #pragma HLS PIPELINE II=1
                hls::vector<T, token_parallel> temp_vec = pref_io_mmap[i * hidden_dim + k];
                buffer[k] = temp_vec + 1.0;
            }

            write_loop:for(int k = 0; k < hidden_dim; k++){
            #pragma HLS PIPELINE II=1
                pref_io_mmap[(max_pre_seq_len / token_parallel) * hidden_dim + i * hidden_dim + k] = buffer[k];
            }
        }
        hmt_stage_01_finish_stream.write(true);
    }
}



#endif

