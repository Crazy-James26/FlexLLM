#ifndef _MHA_88_H_
#define _MHA_88_H_
#include "config.h"
#include "PE.h"
#include "data_io.h"
#include "quant.h"
#include "Softmax.h"
#include "MHA.h"
#include "MHA_flatten.h"

#include "parameters/Q_s.h"
#include "parameters/K_s.h"
#include "parameters/V_s.h"
#include "parameters/A_s.h"

void pref_input_loader_q_fp32(
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> input_mmap,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& input_stream, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_input_loader<float, TOKEN_PARALLEL>(input_mmap, input_stream, block_id, seq_len, HIDDEN_DIM);
    }
}

void pref_quant_layer_q_fp32_int8(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    // tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& input_s_stream, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    // static float Q_s[DECODER_LAYER_NUM][Q_HEAD_NUM];
    // for(int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
    //     for(int h = 0; h < Q_HEAD_NUM; h++){
    //         Q_s[block_id][h] = 1.0f/HIDDEN_DIM;
    //     }
    // }

    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_static_sym_per_tensor_quant_layer_fp32_qint<8, TOKEN_PARALLEL, HEAD_DIM, Q_HEAD_NUM>(
            input_stream, output_stream, Q_s, block_id, seq_len, HEAD_DIM, sqrt_HEAD_DIM
        );
    }
}


void pref_input_loader_k_fp32(
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> input_mmap,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& input_stream, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_input_loader<float, TOKEN_PARALLEL>(input_mmap, input_stream, block_id, seq_len, KV_HIDDEN_DIM);
    }
}


void pref_quant_layer_k_fp32_int8(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    // static float K_s[DECODER_LAYER_NUM][KV_HEAD_NUM];
    // for(int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
    //     for(int h = 0; h < KV_HEAD_NUM; h++){
    //         K_s[block_id][h] = 1.0f/HIDDEN_DIM;
    //     }
    // }

    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_static_sym_per_tensor_quant_layer_fp32_qint<8, TOKEN_PARALLEL, HEAD_DIM, KV_HEAD_NUM>(
            input_stream, output_stream, K_s, block_id, seq_len, HEAD_DIM
        );
    }
}


void pref_K_buffer(
    tapa::istream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& input_k_stream,
    tapa::ostream<hls::vector<ap_int<8>, PRE_K_PARALLEL>>& output_k_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_io_buffer<ap_int<8>, TOKEN_PARALLEL, PRE_K_PARALLEL, KV_HIDDEN_DIM>(
            input_k_stream, output_k_stream, seq_len, KV_HIDDEN_DIM
        );
    }
}

// void pref_K_s_cache_manager(
//     tapa::istream<hls::vector<float, max_w_s_sum_parallel>>& input_k_s_stream,
//     tapa::mmap<hls::vector<float, max_w_s_sum_parallel>> k_s_cache,
//     tapa::ostream<hls::vector<float, max_w_s_sum_parallel>>& output_k_s_stream,
//     int seq_len
// ){
//     decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
//         K_s_cache_manager_template<float, max_w_s_sum_parallel, PRE_K_PARALLEL, KV_HEAD_NUM>(
//             input_k_s_stream, k_s_cache, output_k_s_stream, block_id, seq_len
//         );
//     }
// }



void pref_K_cache_manager(
    tapa::istream<hls::vector<ap_int<8>, PRE_K_PARALLEL>>& input_k_stream,
    tapa::mmap<hls::vector<ap_int<8>, PRE_K_PARALLEL>> k_cache,
    tapa::ostream<hls::vector<ap_int<8>, PRE_K_PARALLEL>>& output_k_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_K_cache_manager_template<ap_int<8>, TOKEN_PARALLEL, PRE_K_PARALLEL, KV_HEAD_NUM, HEAD_DIM, ATTN_GROUP_NUM>(
            input_k_stream, k_cache, output_k_stream, block_id, seq_len
        );
    }
}



void pref_MHA_i8xi8_qxk(
    tapa::istream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& input_seq,
    tapa::istream<hls::vector<ap_int<8>, PRE_K_PARALLEL>>& weight_loader,
    tapa::ostream<hls::vector<ap_int<log2_HEAD_DIM + 16>, TOKEN_PARALLEL>>& output_seq,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_MHA_i8xi8_qxk_template<TOKEN_PARALLEL, PRE_K_PARALLEL, Q_HEAD_NUM, HEAD_DIM, log2_HEAD_DIM>(
            input_seq, weight_loader, output_seq, seq_len
        );
        cout << "Block_id: " << block_id << " MHA_i8xi8_qxk completed." << endl;
    }
}

void pref_dequant_layer_a_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_HEAD_DIM + 16>, TOKEN_PARALLEL>>& input_stream,
    // tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_s_stream, //input's scale factor
    // tapa::istream<hls::vector<float, max_w_s_sum_parallel>>& weight_s_stream, //weight's scale factor 
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    // static float Q_s[DECODER_LAYER_NUM][Q_HEAD_NUM];
    // static float K_s[DECODER_LAYER_NUM][KV_HEAD_NUM];
    // for(int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
    //     for(int h = 0; h < Q_HEAD_NUM; h++){
    //         Q_s[block_id][h] = 1.0f/HIDDEN_DIM;
    //     }
    //     for(int h = 0; h < KV_HEAD_NUM; h++){
    //         K_s[block_id][h] = 1.0f/HIDDEN_DIM;
    //     }
    // }


    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_static_sym_per_tensor_dequant_layer_qint_fp32<log2_HEAD_DIM + 16, TOKEN_PARALLEL, MAX_PRE_SEQ_LEN, Q_HEAD_NUM, KV_HEAD_NUM, MAX_PRE_SEQ_LEN>(
            input_stream, output_stream, Q_s, K_s, block_id, seq_len, seq_len
        );
    }
}


void pref_output_drainer_a_fp32(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> output_mmap, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        io_block_loop: for (int M = 0; M < seq_len/TOKEN_PARALLEL; M++){
        #pragma HLS loop_tripcount min=1 max=1024/16
            attn_head_loop: for (int H = 0; H < Q_HEAD_NUM; H++){
                int bias = ((block_id * MAX_PRE_SEQ_LEN/TOKEN_PARALLEL + M) * Q_HEAD_NUM + H) * MAX_PRE_SEQ_LEN;
                for(int i = 0; i < seq_len; i++){
                    output_mmap[bias + i] = output_stream.read();
                }
            }
        }
    }
}


void MHA_i8xi8_qxk_Prefilling_tb(
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> q_mmap,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> k_mmap,
    tapa::mmap<hls::vector<ap_int<8>, PRE_K_PARALLEL>> k_cache,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> a_mmap,
    int seq_len
){
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> q_stream("q_stream");
    tapa::stream<hls::vector<ap_int<8>, TOKEN_PARALLEL>> quant_q_stream("quant_q_stream");

    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> k_stream("k_stream");
    tapa::stream<hls::vector<ap_int<8>, TOKEN_PARALLEL>> quant_k_stream("quant_k_stream");
    tapa::stream<hls::vector<ap_int<8>, PRE_K_PARALLEL>> cache_quant_k_stream("cache_quant_k_stream");
    tapa::stream<hls::vector<ap_int<8>, PRE_K_PARALLEL>> load_quant_k_stream("load_quant_k_stream");

    tapa::stream<hls::vector<ap_int<log2_HEAD_DIM + 16>, TOKEN_PARALLEL>> quant_a_stream("quant_a_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> a_stream("a_stream");

    tapa::task()
    .invoke(pref_input_loader_k_fp32, k_mmap, k_stream, seq_len)
    .invoke(pref_quant_layer_k_fp32_int8, k_stream, quant_k_stream, seq_len)
    .invoke(pref_K_buffer, quant_k_stream, cache_quant_k_stream, seq_len)
    .invoke(pref_K_cache_manager, cache_quant_k_stream, k_cache, load_quant_k_stream, seq_len)

    .invoke(pref_input_loader_q_fp32, q_mmap, q_stream, seq_len)
    .invoke(pref_quant_layer_q_fp32_int8, q_stream, quant_q_stream, seq_len)
    
    .invoke(pref_MHA_i8xi8_qxk, quant_q_stream, load_quant_k_stream, quant_a_stream, seq_len)
    .invoke(pref_dequant_layer_a_int_fp32, quant_a_stream, a_stream, seq_len)
    .invoke(pref_output_drainer_a_fp32, a_stream, a_mmap, seq_len);
}


void pref_causal_mask(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_causal_mask_template<float, TOKEN_PARALLEL, MAX_PRE_SEQ_LEN, MAX_PRE_SEQ_LEN, Q_HEAD_NUM>(
            input_stream, output_stream, seq_len, seq_len
        );
    }
}

void pref_Softmax_MHA(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_Softmax<float, TOKEN_PARALLEL, MAX_PRE_SEQ_LEN, MAX_PRE_SEQ_LEN, Q_HEAD_NUM>(
            input_stream, output_stream, seq_len, seq_len
        ); // scale input before softmax
        cout << "Block_id: " << block_id << " Softmax_MHA completed." << endl;
    }
}


void pref_quant_layer_sfm_a_fp32_int8(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    // tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& input_s_stream, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    // static float A_s[DECODER_LAYER_NUM][Q_HEAD_NUM];
    // for(int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
    //     for(int h = 0; h < Q_HEAD_NUM; h++){
    //         A_s[block_id][h] = 1.0f/HIDDEN_DIM;
    //     }
    // }


    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_static_sym_per_tensor_quant_layer_fp32_qint<8, TOKEN_PARALLEL, MAX_PRE_SEQ_LEN, Q_HEAD_NUM, false>(
            input_stream, output_stream, A_s, block_id, seq_len, seq_len
        );
    }
}


void pref_input_loader_v_fp32(
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> input_mmap,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& input_stream, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_input_loader<float, TOKEN_PARALLEL>(input_mmap, input_stream, block_id, seq_len, KV_HIDDEN_DIM);
    }
}


void pref_quant_layer_v_fp32_int8(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    // static float V_s[DECODER_LAYER_NUM][KV_HEAD_NUM];
    // for(int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
    //     for(int h = 0; h < KV_HEAD_NUM; h++){
    //         V_s[block_id][h] = 1.0f/HIDDEN_DIM;
    //     }
    // }

    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_static_sym_per_tensor_quant_layer_fp32_qint<8, TOKEN_PARALLEL, HEAD_DIM, KV_HEAD_NUM>(
            input_stream, output_stream, V_s, block_id, seq_len, HEAD_DIM
        );
    }
}


void pref_V_buffer_transpose(
    tapa::istream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& input_v_stream,
    tapa::ostream<hls::vector<ap_int<8>, PRE_V_PARALLEL>>& output_v_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_io_buffer_transpose<ap_int<8>, TOKEN_PARALLEL, PRE_V_PARALLEL, KV_HIDDEN_DIM>(
            input_v_stream, output_v_stream, seq_len, KV_HIDDEN_DIM
        );
    }
}

void pref_V_cache_manager(
    tapa::istream<hls::vector<ap_int<8>, PRE_V_PARALLEL>>& input_v_stream,
    tapa::mmap<hls::vector<ap_int<8>, PRE_V_PARALLEL>> v_cache,
    tapa::ostream<hls::vector<ap_int<8>, PRE_V_PARALLEL>>& output_v_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_V_cache_manager_template<ap_int<8>, TOKEN_PARALLEL, PRE_V_PARALLEL, KV_HEAD_NUM, HEAD_DIM, ATTN_GROUP_NUM>(
            input_v_stream, v_cache, output_v_stream, block_id, seq_len
        );
    }
}

void pref_MHA_i8xi8_axv(
    tapa::istream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& input_seq,
    tapa::istream<hls::vector<ap_int<8>, PRE_V_PARALLEL>>& weight_loader,
    tapa::ostream<hls::vector<ap_int<log2_MAX_PRE_SEQ_LEN + 16>, TOKEN_PARALLEL>>& output_seq,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_MHA_i8xi8_axv_template<TOKEN_PARALLEL, PRE_V_PARALLEL, Q_HEAD_NUM, HEAD_DIM, MAX_PRE_SEQ_LEN, log2_MAX_PRE_SEQ_LEN, true>(
            input_seq, weight_loader, output_seq, seq_len
        );
        cout << "Block_id: " << block_id << " MHA_i8xi8_axv completed." << endl;
    }
}


void pref_dequant_layer_o_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_MAX_PRE_SEQ_LEN + 16>, TOKEN_PARALLEL>>& input_stream,
    // tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_s_stream, //input's scale factor
    // tapa::istream<hls::vector<float, max_w_s_sum_parallel>>& weight_s_stream, //weight's scale factor 
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    // static float A_s[DECODER_LAYER_NUM][Q_HEAD_NUM];
    // static float V_s[DECODER_LAYER_NUM][KV_HEAD_NUM];
    // for(int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
    //     for(int h = 0; h < Q_HEAD_NUM; h++){
    //         A_s[block_id][h] = 1.0f/HIDDEN_DIM;
    //     }
    //     for(int h = 0; h < KV_HEAD_NUM; h++){
    //         V_s[block_id][h] = 1.0f/HIDDEN_DIM;
    //     }
    // }

    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_static_sym_per_tensor_dequant_layer_qint_fp32<log2_MAX_PRE_SEQ_LEN + 16, TOKEN_PARALLEL, HEAD_DIM, Q_HEAD_NUM, KV_HEAD_NUM, MAX_PRE_SEQ_LEN>(
            input_stream, output_stream, A_s, V_s, block_id, seq_len, HEAD_DIM
        );
    }
}


void pref_output_drainer_o_fp32(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> output_mmap, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_output_drainer<float, TOKEN_PARALLEL>(output_stream, output_mmap, block_id, seq_len, HIDDEN_DIM);
    }
}


void MHA_i8xi8_Prefilling_tb(
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> q_mmap,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> k_mmap,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> v_mmap,
    tapa::mmap<hls::vector<ap_int<8>, PRE_K_PARALLEL>> k_cache,
    tapa::mmap<hls::vector<ap_int<8>, PRE_V_PARALLEL>> v_cache,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> o_mmap,
    int seq_len
){
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> q_stream("q_stream");
    tapa::stream<hls::vector<ap_int<8>, TOKEN_PARALLEL>> quant_q_stream("quant_q_stream");

    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> k_stream("k_stream");
    tapa::stream<hls::vector<ap_int<8>, TOKEN_PARALLEL>> quant_k_stream("quant_k_stream");
    tapa::stream<hls::vector<ap_int<8>, PRE_K_PARALLEL>> cache_quant_k_stream("cache_quant_k_stream");
    tapa::stream<hls::vector<ap_int<8>, PRE_K_PARALLEL>> load_quant_k_stream("load_quant_k_stream");

    tapa::stream<hls::vector<ap_int<log2_HEAD_DIM + 16>, TOKEN_PARALLEL>> quant_a_stream("quant_a_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> a_stream("a_stream");

    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> masked_a_stream("masked_a_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> sfm_a_stream("sfm_a_stream");
    tapa::stream<hls::vector<ap_int<8>, TOKEN_PARALLEL>> quant_sfm_a_stream("quant_sfm_a_stream");

    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> v_stream("v_stream");
    tapa::stream<hls::vector<ap_int<8>, TOKEN_PARALLEL>> quant_v_stream("quant_v_stream");
    tapa::stream<hls::vector<ap_int<8>, PRE_V_PARALLEL>> cache_quant_v_stream("cache_quant_v_stream");
    tapa::stream<hls::vector<ap_int<8>, PRE_V_PARALLEL>> load_quant_v_stream("load_quant_v_stream");

    tapa::stream<hls::vector<ap_int<log2_MAX_PRE_SEQ_LEN + 16>, TOKEN_PARALLEL>> quant_o_stream("quant_o_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> o_stream("o_stream");

    tapa::task()
    .invoke(pref_input_loader_k_fp32, k_mmap, k_stream, seq_len)
    .invoke(pref_quant_layer_k_fp32_int8, k_stream, quant_k_stream, seq_len)
    .invoke(pref_K_buffer, quant_k_stream, cache_quant_k_stream, seq_len)
    .invoke(pref_K_cache_manager, cache_quant_k_stream, k_cache, load_quant_k_stream, seq_len)

    .invoke(pref_input_loader_v_fp32, v_mmap, v_stream, seq_len)
    .invoke(pref_quant_layer_v_fp32_int8, v_stream, quant_v_stream, seq_len)
    .invoke(pref_V_buffer_transpose, quant_v_stream, cache_quant_v_stream, seq_len)
    .invoke(pref_V_cache_manager, cache_quant_v_stream, v_cache, load_quant_v_stream, seq_len)


    .invoke(pref_input_loader_q_fp32, q_mmap, q_stream, seq_len)
    .invoke(pref_quant_layer_q_fp32_int8, q_stream, quant_q_stream, seq_len)
    
    .invoke(pref_MHA_i8xi8_qxk, quant_q_stream, load_quant_k_stream, quant_a_stream, seq_len)
    .invoke(pref_dequant_layer_a_int_fp32, quant_a_stream, a_stream, seq_len)

    .invoke(pref_causal_mask, a_stream, masked_a_stream, seq_len)
    .invoke(pref_Softmax_MHA, masked_a_stream, sfm_a_stream, seq_len)
    .invoke(pref_quant_layer_sfm_a_fp32_int8, sfm_a_stream, quant_sfm_a_stream, seq_len)

    .invoke(pref_MHA_i8xi8_axv, quant_sfm_a_stream, load_quant_v_stream, quant_o_stream, seq_len)
    .invoke(pref_dequant_layer_o_int_fp32, quant_o_stream, o_stream, seq_len)

    .invoke(pref_output_drainer_o_fp32, o_stream, o_mmap, seq_len);
}







void dec_input_loader_q_fp32(
    tapa::istream<bool>& block_q_ready_stream,
    tapa::mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> input_mmap,
    tapa::ostream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& input_stream, 
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_input_loader<float, T_QKVO_FFN_BLOCK_PARALLEL>(input_mmap, input_stream, block_id, dec_seq_id, HIDDEN_DIM);
            printf("Dec_seq_id %d: Block_id %d: input_loader_q completed.\n", dec_seq_id, block_id);
            bool q_ready = block_q_ready_stream.read();
        }
    }
}

void dec_Q_buffer(
    tapa::istream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& input_q_stream,
    tapa::ostream<hls::vector<float, DEC_HEAD_PARALLEL>>& output_q_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_io_buffer<float, T_QKVO_FFN_BLOCK_PARALLEL, DEC_HEAD_PARALLEL, HIDDEN_DIM>(input_q_stream, output_q_stream);
        }
    }
}

void dec_quant_layer_q_fp32_int8(
    tapa::istream<hls::vector<float, DEC_HEAD_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>>& output_stream,
    int dec_seq_len
){
    // static float Q_s[DECODER_LAYER_NUM][Q_HEAD_NUM];
    // for(int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
    //     for(int h = 0; h < Q_HEAD_NUM; h++){
    //         Q_s[block_id][h] = float(h+1)/HIDDEN_DIM;
    //     }
    // }
    
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_MHA_static_sym_per_tensor_quant_layer_fp32_qint<8, DEC_HEAD_PARALLEL, HEAD_DIM, Q_HEAD_NUM>(
                input_stream, output_stream, Q_s, block_id, HEAD_DIM, sqrt_HEAD_DIM
            );
        }
    }
}

void dec_input_loader_k_fp32(
    tapa::istream<bool>& block_k_ready_stream,
    tapa::mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> input_mmap,
    tapa::ostream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& input_stream, 
    int pre_seq_len,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_input_loader<float, T_QKVO_FFN_BLOCK_PARALLEL, KV_HIDDEN_DIM, MAX_SUM_SEQ_LEN>(
                input_mmap, input_stream, block_id, pre_seq_len + dec_seq_id, KV_HIDDEN_DIM
            );
            printf("Dec_seq_id %d: Block_id %d: input_loader_k completed.\n", dec_seq_id, block_id);
            bool k_ready = block_k_ready_stream.read();
        }
    }
}


void dec_K_buffer(
    tapa::istream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& input_k_stream,
    tapa::ostream<hls::vector<float, DEC_HEAD_PARALLEL>>& output_k_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_io_buffer<float, T_QKVO_FFN_BLOCK_PARALLEL, DEC_HEAD_PARALLEL, KV_HIDDEN_DIM>(input_k_stream, output_k_stream);
        }
    }
}


void dec_quant_layer_k_fp32_int8(
    tapa::istream<hls::vector<float, DEC_HEAD_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>>& output_stream,
    int dec_seq_len
){
    // static float K_s[DECODER_LAYER_NUM][KV_HEAD_NUM];
    // for(int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
    //     for(int h = 0; h < KV_HEAD_NUM; h++){
    //         K_s[block_id][h] = float(h+1)/HIDDEN_DIM;
    //     }
    // }
    
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_MHA_static_sym_per_tensor_quant_layer_fp32_qint<8, DEC_HEAD_PARALLEL, HEAD_DIM, KV_HEAD_NUM>(
                input_stream, output_stream, K_s, block_id, HEAD_DIM
            );
        }
    }
}


void dec_K_cache_buffer(
    tapa::istream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>>& input_k_stream,
    tapa::ostreams<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL>& output_k_streams,
    int pre_seq_len,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_K_cache_buffer_template<ap_int<8>, DEC_HEAD_PARALLEL, DEC_K_PARALLEL, KV_HIDDEN_DIM>(
                input_k_stream, output_k_streams, block_id, pre_seq_len + dec_seq_id
            );
        }
    }
}


void dec_K_cache_manager(
    tapa::istream<hls::vector<ap_int<8>, DEC_K_PARALLEL>>& input_k_stream,
    tapa::mmap<hls::vector<ap_int<8>, DEC_K_PARALLEL>> k_cache,
    tapa::ostream<hls::vector<ap_int<8>, DEC_K_PARALLEL>>& output_k_stream,
    int pre_seq_len,
    int dec_seq_len,
    int addr_bias = 0
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_K_cache_manager_template<ap_int<8>, DEC_HEAD_PARALLEL, DEC_K_PARALLEL, KV_HEAD_NUM, HEAD_DIM, ATTN_GROUP_NUM>(
                input_k_stream, k_cache, output_k_stream, block_id, pre_seq_len + dec_seq_id, addr_bias
            );
        }
    }
}



void dec_MHA_i8xi8_qxk(
    tapa::istream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>>& input_seq,
    tapa::istreams<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL>& weight_loaders,
    tapa::ostream<hls::vector<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL>>& output_seq,
    int pre_seq_len,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_MHA_i8xi8_qxk_template<DEC_HEAD_PARALLEL, DEC_K_PARALLEL, Q_HEAD_NUM, HEAD_DIM, log2_HEAD_DIM>(
                input_seq, weight_loaders, output_seq, pre_seq_len + dec_seq_id + 1
            );
            printf("Dec_seq_id %d: Block_id %d: MHA_i8xi8_qxk completed.\n", dec_seq_id, block_id);
        }
    }
}

void dec_MHA_i8xi8_qxk_input_broadcastor(
    tapa::istream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>>& input_seq,
    tapa::ostreams<ap_int<8>, DEC_HEAD_PARALLEL>& input_loaders,
    int pre_seq_len,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_MHA_qxk_input_broadcastor_template<ap_int<8>, DEC_HEAD_PARALLEL, DEC_K_PARALLEL, Q_HEAD_NUM, HEAD_DIM>(
                input_seq, input_loaders, pre_seq_len + dec_seq_id + 1
            );
        }
    }
}

void dec_MHA_i8xi8_qxk_flatten(
    tapa::istream<ap_int<8>>& input_loader,
    tapa::istream<hls::vector<ap_int<8>, DEC_K_PARALLEL>>& weight_loader,
    tapa::ostream<ap_int<log2_HEAD_DIM + 16>>& output_drainer,
    int pre_seq_len,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_MHA_i8xi8_qxk_flatten_template<DEC_HEAD_PARALLEL, DEC_K_PARALLEL, Q_HEAD_NUM, HEAD_DIM, log2_HEAD_DIM>(
                input_loader, weight_loader, output_drainer, pre_seq_len + dec_seq_id + 1
            );
            printf("Dec_seq_id %d: Block_id %d: MHA_i8xi8_qxk_flatten completed.\n", dec_seq_id, block_id);
        }
    }
}

void dec_MHA_i8xi8_qxk_output_merger(
    tapa::istreams<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL>& output_drainers,
    tapa::ostream<hls::vector<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL>>& output_seq,
    int pre_seq_len,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_MHA_qxk_output_merger_template<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL, DEC_K_PARALLEL, Q_HEAD_NUM>(
                output_drainers, output_seq, pre_seq_len + dec_seq_id + 1
            );
        }
    }
}



void dec_quant_a_discard(
    tapa::istream<hls::vector<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL>>& input_seq,
    tapa::ostream<hls::vector<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL>>& output_seq,
    int pre_seq_len,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_quant_a_discard_template<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL, DEC_K_PARALLEL, Q_HEAD_NUM>(
                input_seq, output_seq, pre_seq_len + dec_seq_id + 1
            );
        }
    }
}


void dec_dequant_layer_a_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, DEC_HEAD_PARALLEL>>& output_stream,
    int pre_seq_len,
    int dec_seq_len
){
    // static float Q_s[DECODER_LAYER_NUM][Q_HEAD_NUM];
    // static float K_s[DECODER_LAYER_NUM][KV_HEAD_NUM];
    // for(int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
    //     for(int h = 0; h < Q_HEAD_NUM; h++){
    //         Q_s[block_id][h] = float(h+1)/HIDDEN_DIM;
    //     }
    //     for(int h = 0; h < KV_HEAD_NUM; h++){
    //         K_s[block_id][h] = float(h+1)/HIDDEN_DIM;
    //     }
    // }

    
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_MHA_static_sym_per_tensor_dequant_layer_qint_fp32<log2_HEAD_DIM + 16, DEC_HEAD_PARALLEL, MAX_SUM_SEQ_LEN, Q_HEAD_NUM, KV_HEAD_NUM>(
                input_stream, output_stream, Q_s, K_s, block_id, pre_seq_len + dec_seq_id + 1
            );
        }
    }
}

void dec_output_drainer_a_fp32(
    tapa::istream<hls::vector<float, DEC_HEAD_PARALLEL>>& output_stream,
    tapa::mmap<hls::vector<float, DEC_HEAD_PARALLEL>> output_mmap,
    tapa::ostream<bool>& block_q_ready_stream,
    tapa::ostream<bool>& block_k_ready_stream,
    int pre_seq_len,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            attn_head_loop: for (int H = 0; H < Q_HEAD_NUM/DEC_HEAD_PARALLEL; H++){
                int bias = ((dec_seq_id * DECODER_LAYER_NUM + block_id) * Q_HEAD_NUM/DEC_HEAD_PARALLEL + H) * MAX_SUM_SEQ_LEN;
                write_output_loop: for(int i = 0; i < pre_seq_len + dec_seq_id + 1; i++){
                    #pragma HLS pipeline II=1
                    output_mmap[bias + i] = output_stream.read();
                }
            }
            printf("Dec_seq_id %d: Block_id %d: output_drainer_a completed.\n", dec_seq_id, block_id);
            block_q_ready_stream.write(true);
            block_k_ready_stream.write(true);
        }
    }
}


void MHA_i8xi8_qxk_decoding_tb(
    tapa::mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> q_mmap,
    tapa::mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> k_mmap,
    tapa::mmaps<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL> k_caches,
    tapa::mmap<hls::vector<float, DEC_HEAD_PARALLEL>> a_mmap,
    int pre_seq_len,
    int dec_seq_len
){
    tapa::stream<bool> block_q_ready_stream;
    tapa::stream<bool> block_k_ready_stream;

    tapa::stream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> q_in_stream("q_in_stream");
    tapa::stream<hls::vector<float, DEC_HEAD_PARALLEL>> q_stream("q_stream");
    tapa::stream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>> quant_q_stream("quant_q_stream");
    tapa::streams<ap_int<8>, DEC_HEAD_PARALLEL> quant_q_loaders("quant_q_loaders");

    tapa::stream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> k_in_stream("k_in_stream");
    tapa::stream<hls::vector<float, DEC_HEAD_PARALLEL>> k_stream("k_stream");
    tapa::stream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>> quant_k_stream("quant_k_stream");
    tapa::streams<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL> cache_quant_k_streams("cache_quant_k_streams");
    tapa::streams<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL> load_quant_k_streams("load_quant_k_streams");


    tapa::streams<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL> quant_a_drainers("quant_a_drainers");
    tapa::stream<hls::vector<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL>> quant_a_stream_redundant("quant_a_stream_redundant");
    tapa::stream<hls::vector<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL>> quant_a_stream("quant_a_stream");
    tapa::stream<hls::vector<float, DEC_HEAD_PARALLEL>> a_stream("a_stream");

    tapa::task()
    .invoke(dec_input_loader_k_fp32, block_k_ready_stream, k_mmap, k_in_stream, pre_seq_len,dec_seq_len)
    .invoke(dec_K_buffer, k_in_stream, k_stream, dec_seq_len)
    .invoke(dec_quant_layer_k_fp32_int8, k_stream, quant_k_stream, dec_seq_len)
    .invoke(dec_K_cache_buffer, quant_k_stream, cache_quant_k_streams, pre_seq_len, dec_seq_len)
    .invoke<tapa::detach, DEC_HEAD_PARALLEL>(dec_K_cache_manager, cache_quant_k_streams, k_caches, load_quant_k_streams, pre_seq_len, dec_seq_len, 0)

    .invoke(dec_input_loader_q_fp32, block_q_ready_stream, q_mmap, q_in_stream, dec_seq_len)
    .invoke(dec_Q_buffer, q_in_stream, q_stream, dec_seq_len)
    .invoke(dec_quant_layer_q_fp32_int8, q_stream, quant_q_stream, dec_seq_len)
    .invoke(dec_MHA_i8xi8_qxk_input_broadcastor, quant_q_stream, quant_q_loaders, pre_seq_len, dec_seq_len)
    
    // .invoke(MHA_i8xi8_qxk, quant_q_stream, load_quant_k_streams, quant_a_stream_redundant, pre_seq_len, dec_seq_len)
    .invoke<tapa::detach, DEC_HEAD_PARALLEL>(
        dec_MHA_i8xi8_qxk_flatten, quant_q_loaders, load_quant_k_streams, quant_a_drainers, pre_seq_len, dec_seq_len
    )

    .invoke(dec_MHA_i8xi8_qxk_output_merger, quant_a_drainers, quant_a_stream_redundant, pre_seq_len, dec_seq_len)
    .invoke(dec_quant_a_discard, quant_a_stream_redundant, quant_a_stream, pre_seq_len, dec_seq_len)
    .invoke(dec_dequant_layer_a_int_fp32, quant_a_stream, a_stream, pre_seq_len, dec_seq_len)
    .invoke(dec_output_drainer_a_fp32, a_stream, a_mmap, block_q_ready_stream, block_k_ready_stream, pre_seq_len, dec_seq_len);
}


//////////////////////

// void dec_causal_mask(
//     tapa::istream<hls::vector<float, DEC_HEAD_PARALLEL>>& input_stream,
//     tapa::ostream<hls::vector<float, DEC_HEAD_PARALLEL>>& output_stream,
//     int pre_seq_len,
//     int dec_seq_len
// ){
//     decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
//         decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
//             dec_causal_mask_template<float, DEC_HEAD_PARALLEL, MAX_SUM_SEQ_LEN, Q_HEAD_NUM>(
//                 input_stream, output_stream, pre_seq_len + dec_seq_id + 1
//             ); // scale input before softmax
//         }
//     }
// }



void dec_Softmax_MHA(
    tapa::istream<hls::vector<float, DEC_HEAD_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, DEC_HEAD_PARALLEL>>& output_stream,
    int pre_seq_len,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_MHA_Softmax<float, DEC_HEAD_PARALLEL, MAX_SUM_SEQ_LEN, Q_HEAD_NUM>(
                input_stream, output_stream, pre_seq_len + dec_seq_id + 1
            ); // scale input before softmax
            printf("Dec_seq_id %d: Block_id %d: Softmax_MHA completed.\n", dec_seq_id, block_id);
        }
    }
}


void dec_quant_layer_sfm_a_fp32_int8(
    tapa::istream<hls::vector<float, DEC_HEAD_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>>& output_stream,
    int pre_seq_len,
    int dec_seq_len
){
    // static float A_s[DECODER_LAYER_NUM][Q_HEAD_NUM];
    // for(int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
    //     for(int h = 0; h < Q_HEAD_NUM; h++){
    //         A_s[block_id][h] = float(h+1)/HIDDEN_DIM;
    //     }
    // }

    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_MHA_static_sym_per_tensor_quant_layer_fp32_qint<8, DEC_HEAD_PARALLEL, MAX_SUM_SEQ_LEN, Q_HEAD_NUM, false>(
                input_stream, output_stream, A_s, block_id, pre_seq_len + dec_seq_id + 1
            );
        }
    }
}


void dec_input_loader_v_fp32(
    tapa::istream<bool>& block_v_ready_stream,
    tapa::mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> input_mmap,
    tapa::ostream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& input_stream, 
    int pre_seq_len,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_input_loader<float, T_QKVO_FFN_BLOCK_PARALLEL, KV_HIDDEN_DIM, MAX_SUM_SEQ_LEN>(
                input_mmap, input_stream, block_id, pre_seq_len + dec_seq_id, KV_HIDDEN_DIM
            );
            printf("Dec_seq_id %d: Block_id %d: input_loader_v completed.\n", dec_seq_id, block_id);
            bool v_ready = block_v_ready_stream.read();
        }
    }
}


void dec_V_buffer(
    tapa::istream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& input_v_stream,
    tapa::ostream<hls::vector<float, DEC_HEAD_PARALLEL>>& output_v_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_io_buffer<float, T_QKVO_FFN_BLOCK_PARALLEL, DEC_HEAD_PARALLEL, KV_HIDDEN_DIM>(input_v_stream, output_v_stream);
        }
    }
}


void dec_quant_layer_v_fp32_int8(
    tapa::istream<hls::vector<float, DEC_HEAD_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>>& output_stream,
    int dec_seq_len
){
    // static float V_s[DECODER_LAYER_NUM][KV_HEAD_NUM];
    // for(int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
    //     for(int h = 0; h < KV_HEAD_NUM; h++){
    //         V_s[block_id][h] = float(h+1)/HIDDEN_DIM;
    //     }
    // }

    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_MHA_static_sym_per_tensor_quant_layer_fp32_qint<8, DEC_HEAD_PARALLEL, HEAD_DIM, KV_HEAD_NUM>(
                input_stream, output_stream, V_s, block_id, HEAD_DIM
            );
        }
    }
}


void dec_V_cache_buffer(
    tapa::istream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>>& input_v_stream,
    tapa::ostreams<hls::vector<ap_int<8>, DEC_V_PARALLEL>, DEC_HEAD_PARALLEL>& output_v_streams,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_V_cache_buffer_template<ap_int<8>, DEC_HEAD_PARALLEL, DEC_V_PARALLEL, KV_HIDDEN_DIM>(
                input_v_stream, output_v_streams, block_id
            );
        }
    }
}



void dec_V_cache_manager(
    tapa::istream<hls::vector<ap_int<8>, DEC_V_PARALLEL>>& input_v_stream,
    tapa::mmap<hls::vector<ap_int<8>, DEC_V_PARALLEL>> v_cache,
    tapa::ostream<hls::vector<ap_int<8>, DEC_V_PARALLEL>>& output_v_stream,
    int pre_seq_len,
    int dec_seq_len,
    int addr_bias = 0
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_V_cache_manager_template<ap_int<8>, DEC_HEAD_PARALLEL, DEC_V_PARALLEL, KV_HEAD_NUM, HEAD_DIM, ATTN_GROUP_NUM>(
                input_v_stream, v_cache, output_v_stream, block_id, pre_seq_len + dec_seq_id, addr_bias
            );
        }
    }
}

void dec_MHA_i8xi8_axv(
    tapa::istream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>>& input_seq,
    tapa::istreams<hls::vector<ap_int<8>, DEC_V_PARALLEL>, DEC_HEAD_PARALLEL>& weight_loaders,
    tapa::ostream<hls::vector<ap_int<log2_MAX_SUM_SEQ_LEN + 16>, DEC_HEAD_PARALLEL>>& output_seq,
    int pre_seq_len,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_MHA_i8xi8_axv_template<DEC_HEAD_PARALLEL, DEC_V_PARALLEL, Q_HEAD_NUM, HEAD_DIM, MAX_SUM_SEQ_LEN, log2_MAX_SUM_SEQ_LEN, true>(
                input_seq, weight_loaders, output_seq, pre_seq_len + dec_seq_id + 1
            );
            printf("Dec_seq_id %d: Block_id %d: MHA_i8xi8_axv completed.\n", dec_seq_id, block_id);
        }
    }
}

void dec_MHA_i8xi8_axv_input_broadcastor(
    tapa::istream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>>& input_seq,
    tapa::ostreams<ap_int<8>, DEC_HEAD_PARALLEL>& input_loaders,
    int pre_seq_len,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_MHA_axv_input_broadcastor_template<ap_int<8>, DEC_HEAD_PARALLEL, DEC_V_PARALLEL, Q_HEAD_NUM, HEAD_DIM>(
                input_seq, input_loaders, pre_seq_len + dec_seq_id + 1
            );
        }
    }
}

void dec_MHA_i8xi8_axv_flatten(
    tapa::istream<ap_int<8>>& input_loader,
    tapa::istream<hls::vector<ap_int<8>, DEC_V_PARALLEL>>& weight_loader,
    tapa::ostream<ap_int<log2_MAX_SUM_SEQ_LEN + 16>>& output_drainer,
    int pre_seq_len,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_MHA_i8xi8_axv_flatten_template<DEC_HEAD_PARALLEL, DEC_V_PARALLEL, Q_HEAD_NUM, HEAD_DIM, MAX_SUM_SEQ_LEN, log2_MAX_SUM_SEQ_LEN, true>(
                input_loader, weight_loader, output_drainer, pre_seq_len + dec_seq_id + 1
            );
            printf("Dec_seq_id %d: Block_id %d: MHA_i8xi8_axv_flatten completed.\n", dec_seq_id, block_id);
        }
    }
}

void dec_MHA_i8xi8_axv_output_merger(
    tapa::istreams<ap_int<log2_MAX_SUM_SEQ_LEN + 16>, DEC_HEAD_PARALLEL>& output_drainers,
    tapa::ostream<hls::vector<ap_int<log2_MAX_SUM_SEQ_LEN + 16>, DEC_HEAD_PARALLEL>>& output_seq,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_MHA_axv_output_merger_template<ap_int<log2_MAX_SUM_SEQ_LEN + 16>, DEC_HEAD_PARALLEL, DEC_V_PARALLEL, Q_HEAD_NUM, HEAD_DIM>(
                output_drainers, output_seq
            );
        }
    }
}


void dec_dequant_layer_o_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_MAX_SUM_SEQ_LEN + 16>, DEC_HEAD_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, DEC_HEAD_PARALLEL>>& output_stream,
    int dec_seq_len
){
    // static float A_s[DECODER_LAYER_NUM][Q_HEAD_NUM];
    // static float V_s[DECODER_LAYER_NUM][KV_HEAD_NUM];
    // for(int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
    //     for(int h = 0; h < Q_HEAD_NUM; h++){
    //         A_s[block_id][h] = float(h+1)/HIDDEN_DIM;
    //     }
    //     for(int h = 0; h < KV_HEAD_NUM; h++){
    //         V_s[block_id][h] = float(h+1)/HIDDEN_DIM;
    //     }
    // }


    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        dec_MHA_static_sym_per_tensor_dequant_layer_qint_fp32<log2_MAX_SUM_SEQ_LEN + 16, DEC_HEAD_PARALLEL, HEAD_DIM, Q_HEAD_NUM, KV_HEAD_NUM>(
            input_stream, output_stream, A_s, V_s, block_id, HEAD_DIM
        );
    }
    }
}


void dec_O_buffer(
    tapa::istream<hls::vector<float, DEC_HEAD_PARALLEL>>& input_o_stream,
    tapa::ostream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& output_o_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_io_buffer<float, DEC_HEAD_PARALLEL, T_QKVO_FFN_BLOCK_PARALLEL, HIDDEN_DIM>(input_o_stream, output_o_stream);
        }
    }
}

void dec_output_drainer_o_fp32(
    tapa::istream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& output_stream,
    tapa::mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> output_mmap, 
    tapa::ostream<bool>& block_q_ready_stream,
    tapa::ostream<bool>& block_k_ready_stream,
    tapa::ostream<bool>& block_v_ready_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_output_drainer<float, T_QKVO_FFN_BLOCK_PARALLEL>(output_stream, output_mmap, block_id, dec_seq_id, HIDDEN_DIM);
            printf("Dec_seq_id %d: Block_id %d: output_drainer_o completed.\n", dec_seq_id, block_id);
            block_q_ready_stream.write(true);
            block_k_ready_stream.write(true);
            block_v_ready_stream.write(true);
        }
    }
}


void MHA_i8xi8_decoding_tb(
    tapa::mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> q_mmap,
    tapa::mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> k_mmap,
    tapa::mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> v_mmap,
    tapa::mmaps<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL> k_caches,
    tapa::mmaps<hls::vector<ap_int<8>, DEC_V_PARALLEL>, DEC_HEAD_PARALLEL> v_caches,
    tapa::mmap<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> o_mmap,
    int pre_seq_len,
    int dec_seq_len
){
    tapa::stream<bool> block_q_ready_stream;
    tapa::stream<bool> block_k_ready_stream;
    tapa::stream<bool> block_v_ready_stream;

    tapa::stream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> q_in_stream("q_in_stream");
    tapa::stream<hls::vector<float, DEC_HEAD_PARALLEL>> q_stream("q_stream");
    tapa::stream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>> quant_q_stream("quant_q_stream");
    tapa::streams<ap_int<8>, DEC_HEAD_PARALLEL> quant_q_loaders("quant_q_loaders");
    
    tapa::stream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> k_in_stream("k_in_stream");
    tapa::stream<hls::vector<float, DEC_HEAD_PARALLEL>> k_stream("k_stream");
    tapa::stream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>> quant_k_stream("quant_k_stream");
    tapa::streams<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL> cache_quant_k_streams("cache_quant_k_streams");
    tapa::streams<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL> load_quant_k_streams("load_quant_k_streams");

    tapa::streams<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL> quant_a_drainers("quant_a_drainers");
    tapa::stream<hls::vector<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL>> quant_a_stream_redundant("quant_a_stream_redundant");
    tapa::stream<hls::vector<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL>> quant_a_stream("quant_a_stream");
    tapa::stream<hls::vector<float, DEC_HEAD_PARALLEL>> a_stream("a_stream");

    tapa::stream<hls::vector<float, DEC_HEAD_PARALLEL>> sfm_a_stream("sfm_a_stream");
    tapa::stream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>> quant_sfm_a_stream("quant_sfm_a_stream");
    tapa::streams<ap_int<8>, DEC_HEAD_PARALLEL> quant_sfm_a_loaders("quant_sfm_a_loaders");

    tapa::stream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> v_in_stream("v_in_stream");
    tapa::stream<hls::vector<float, DEC_HEAD_PARALLEL>> v_stream("v_stream");
    tapa::stream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>> quant_v_stream("quant_v_stream");
    tapa::streams<hls::vector<ap_int<8>, DEC_V_PARALLEL>, DEC_HEAD_PARALLEL> cache_quant_v_streams("cache_quant_v_streams");
    tapa::streams<hls::vector<ap_int<8>, DEC_V_PARALLEL>, DEC_HEAD_PARALLEL> load_quant_v_streams("load_quant_v_streams");

    tapa::streams<ap_int<log2_MAX_SUM_SEQ_LEN + 16>, DEC_HEAD_PARALLEL> quant_o_drainers("quant_o_drainers");
    tapa::stream<hls::vector<ap_int<log2_MAX_SUM_SEQ_LEN + 16>, DEC_HEAD_PARALLEL>> quant_o_stream("quant_o_stream");
    tapa::stream<hls::vector<float, DEC_HEAD_PARALLEL>> o_stream("o_stream");
    tapa::stream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> o_out_stream("o_out_stream");

    tapa::task()
    .invoke(dec_input_loader_k_fp32, block_k_ready_stream, k_mmap, k_in_stream, pre_seq_len, dec_seq_len)
    .invoke(dec_K_buffer, k_in_stream, k_stream, dec_seq_len)
    .invoke(dec_quant_layer_k_fp32_int8, k_stream, quant_k_stream, dec_seq_len)
    .invoke(dec_K_cache_buffer, quant_k_stream, cache_quant_k_streams, pre_seq_len, dec_seq_len)
    .invoke<tapa::detach, DEC_HEAD_PARALLEL>(dec_K_cache_manager, cache_quant_k_streams, k_caches, load_quant_k_streams, pre_seq_len, dec_seq_len, 0)

    .invoke(dec_input_loader_q_fp32, block_q_ready_stream, q_mmap, q_in_stream, dec_seq_len)
    .invoke(dec_Q_buffer, q_in_stream, q_stream, dec_seq_len)
    .invoke(dec_quant_layer_q_fp32_int8, q_stream, quant_q_stream, dec_seq_len)
    .invoke(dec_MHA_i8xi8_qxk_input_broadcastor, quant_q_stream, quant_q_loaders, pre_seq_len, dec_seq_len)
    
    // .invoke(dec_MHA_i8xi8_qxk, quant_q_stream, load_quant_k_streams, quant_a_stream_redundant, pre_seq_len, dec_seq_len)
    .invoke<tapa::detach, DEC_HEAD_PARALLEL>(
        dec_MHA_i8xi8_qxk_flatten, quant_q_loaders, load_quant_k_streams, quant_a_drainers, pre_seq_len, dec_seq_len
    )

    .invoke(dec_MHA_i8xi8_qxk_output_merger, quant_a_drainers, quant_a_stream_redundant, pre_seq_len, dec_seq_len)
    .invoke(dec_quant_a_discard, quant_a_stream_redundant, quant_a_stream, pre_seq_len, dec_seq_len)
    .invoke(dec_dequant_layer_a_int_fp32, quant_a_stream, a_stream, pre_seq_len, dec_seq_len)

    .invoke(dec_Softmax_MHA, a_stream, sfm_a_stream, pre_seq_len, dec_seq_len)
    .invoke(dec_quant_layer_sfm_a_fp32_int8, sfm_a_stream, quant_sfm_a_stream, pre_seq_len, dec_seq_len)
    .invoke(dec_MHA_i8xi8_axv_input_broadcastor, quant_sfm_a_stream, quant_sfm_a_loaders, pre_seq_len, dec_seq_len)

    .invoke(dec_input_loader_v_fp32, block_v_ready_stream, v_mmap, v_in_stream, pre_seq_len, dec_seq_len)
    .invoke(dec_V_buffer, v_in_stream, v_stream, dec_seq_len)
    .invoke(dec_quant_layer_v_fp32_int8, v_stream, quant_v_stream, dec_seq_len)
    .invoke(dec_V_cache_buffer, quant_v_stream, cache_quant_v_streams, dec_seq_len)
    .invoke<tapa::detach, DEC_HEAD_PARALLEL>(dec_V_cache_manager, cache_quant_v_streams, v_caches, load_quant_v_streams, pre_seq_len, dec_seq_len, 0)

    // .invoke(MHA_i8xi8_axv, quant_sfm_a_stream, load_quant_v_streams, quant_o_stream, pre_seq_len, dec_seq_len)
    .invoke<tapa::detach, DEC_HEAD_PARALLEL>(
        dec_MHA_i8xi8_axv_flatten, quant_sfm_a_loaders, load_quant_v_streams, quant_o_drainers, pre_seq_len, dec_seq_len
    )

    .invoke(dec_MHA_i8xi8_axv_output_merger, quant_o_drainers, quant_o_stream, dec_seq_len)
    .invoke(dec_dequant_layer_o_int_fp32, quant_o_stream, o_stream, dec_seq_len)
    .invoke(dec_O_buffer, o_stream,  o_out_stream, dec_seq_len)
    .invoke(dec_output_drainer_o_fp32, o_out_stream, o_mmap, block_q_ready_stream, block_k_ready_stream, block_v_ready_stream, dec_seq_len)
    
    ;
}


#endif