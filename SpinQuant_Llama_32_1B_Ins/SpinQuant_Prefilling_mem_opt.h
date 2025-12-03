#ifndef _SQ_PREF_H_
#define _SQ_PREF_H_
#include "config_u280_mem_opt.h"
#include "PE.h"
#include "data_io.h"
#include "quant.h"
#include "Linear_Layer.h"
#include "RoPE.h"
#include "parameters/RoPE_sin_cos.h"
#include "MHA_i8xi8.h"
#include "Residual_Layer.h"
#include "Swish.h"
#include "LayerNorm.h"
#include "Softmax.h"
#include "FHT.h"


void pref_block_input_loader_sync(
    tapa::istream<bool>& block_input_ready_stream,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> io_mmap,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& input_stream, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_input_loader<float, TOKEN_PARALLEL>(io_mmap, input_stream, block_id, seq_len, HIDDEN_DIM); // for k and v
        cout << "Block_id: " << block_id << " block_input_loader completed for k and v." << endl;
        
        pref_input_loader<float, TOKEN_PARALLEL>(io_mmap, input_stream, block_id, seq_len, HIDDEN_DIM); // for q
        cout << "Block_id: " << block_id << " block_input_loader completed for q." << endl;

        bool block_input_ready = block_input_ready_stream.read(); // signal that input embedding is ready for this decoder block
    }
}


// cache iembed for residual layer 0
void pref_iembed_distributor(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_ln,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_res0,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_stream_distributor<float, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream, output_stream_ln, output_stream_res0, 0, HIDDEN_DIM, seq_len
        );
        pref_stream_distributor<float, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream, output_stream_ln, output_stream_res0, 2, HIDDEN_DIM, seq_len
        );
    }
}

// void pref_Residual_storer_res0(
//     tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& res0_stream_cache,
//     tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> res0_cache_mmap,
//     tapa::ostream<bool>& res0_cache_finish_stream,
//     int seq_len
// ){
//     decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
//         pref_Residual_storer<float, TOKEN_PARALLEL, HIDDEN_DIM>(
//             res0_stream_cache, res0_cache_mmap, res0_cache_finish_stream, 0, seq_len
//         );
//     }
// }


// Layer Norm 0
void pref_Layer_Norm_0_gamma_beta_loader(
    tapa::mmap<float> gamma_beta_mmap,
    tapa::ostream<float>& gamma_beta_stream
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_Layer_Norm_gamma_beta_loader<float, HIDDEN_DIM>(gamma_beta_mmap, gamma_beta_stream, block_id); // for k and v
        pref_Layer_Norm_gamma_beta_loader<float, HIDDEN_DIM>(gamma_beta_mmap, gamma_beta_stream, block_id); // for q
    }
}

void pref_Layer_Norm_0(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::istream<float>& gamma_beta_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_Layer_Norm<float, TOKEN_PARALLEL, HIDDEN_DIM>(input_stream, gamma_beta_stream, output_stream, seq_len, HIDDEN_DIM); // for k and v
        cout << "Block_id: " << block_id << " Layer_Norm_0 for k and v completed." << endl;
        pref_Layer_Norm<float, TOKEN_PARALLEL, HIDDEN_DIM>(input_stream, gamma_beta_stream, output_stream, seq_len, HIDDEN_DIM); // for q
        cout << "Block_id: " << block_id << " Layer_Norm_0 for q completed." << endl;
    }
}


// Linear Layer QKVO
// reuse Linear Layer for k and q
void pref_LN_iembed_temporal_distributor(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_kq,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_v,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        // io_block_kv_loop: for (int M = 0; M < seq_len/TOKEN_PARALLEL*HIDDEN_DIM; M++){
        // #pragma HLS loop_tripcount min=1*2048 max=1024/16*2048
        //     hls::vector<float, TOKEN_PARALLEL> input_pack = input_stream.read();
        //     output_stream_kq.write(input_pack);
        //     output_stream_v.write(input_pack);
        // }
        // io_block_q_loop: for (int M = 0; M < seq_len/TOKEN_PARALLEL*HIDDEN_DIM; M++){
        // #pragma HLS loop_tripcount min=1*2048 max=1024/16*2048
        //     hls::vector<float, TOKEN_PARALLEL> input_pack = input_stream.read();
        //     output_stream_kq.write(input_pack);
        // }
        pref_stream_distributor<float, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream, output_stream_kq, output_stream_v, 2, HIDDEN_DIM, seq_len
        );
        pref_stream_distributor<float, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream, output_stream_kq, output_stream_v, 0, HIDDEN_DIM, seq_len
        );
    }
}


void pref_quant_layer_kq_fp32_int4(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_kq,
    tapa::ostream<hls::vector<float, 2>>& input_s_b_stream_kq, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& output_stream_kq,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_quant_layer_fp32_qint<4, true, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_kq, input_s_b_stream_kq, output_stream_kq, seq_len, HIDDEN_DIM 
        ); // for k
        pref_quant_layer_fp32_qint<4, true, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_kq, input_s_b_stream_kq, output_stream_kq, seq_len, HIDDEN_DIM
        ); // for q
    }
}


void pref_weight_loader_wk_wq(
    tapa::mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>> wk_wq_mmap,
    tapa::ostream<hls::vector<ap_int<4>, PRE_QKVO_W_PARALLEL>>& wk_wq_stream, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_weight_loader_int4_pack_2_discard<PRE_QKVO_W_PARALLEL_READ, PRE_QKVO_W_PARALLEL, TOKEN_PARALLEL>(
            wk_wq_mmap, wk_wq_stream, block_id, seq_len, HIDDEN_DIM, KV_HIDDEN_DIM
        );
        pref_weight_loader_int4_pack_2_discard<PRE_QKVO_W_PARALLEL_READ, PRE_QKVO_W_PARALLEL, TOKEN_PARALLEL>(
            wk_wq_mmap, wk_wq_stream, block_id, seq_len, HIDDEN_DIM, HIDDEN_DIM, 
            DECODER_LAYER_NUM * ((KV_HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL) * HIDDEN_DIM
        );
    }
}

void pref_weight_s_loader_wk_wq(
    tapa::mmap<hls::vector<float, 2>> wk_wq_s_sum_mmap,
    tapa::ostream<hls::vector<float, 2>>& wk_wq_s_sum_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_weight_s_loader_fp32<TOKEN_PARALLEL, true>(
            wk_wq_s_sum_mmap, wk_wq_s_sum_stream, block_id, seq_len, KV_HIDDEN_DIM
        );
        pref_weight_s_loader_fp32<TOKEN_PARALLEL, true>(
            wk_wq_s_sum_mmap, wk_wq_s_sum_stream, block_id, seq_len, HIDDEN_DIM, DECODER_LAYER_NUM * KV_HIDDEN_DIM
        );
    }
}


void pref_Linear_Layer_i4xi4_kq(
    tapa::istream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& input_stream_kq,
    tapa::istream<hls::vector<ap_int<4>, PRE_QKVO_W_PARALLEL>>& wk_wq_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream_kq,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_Linear_Layer_i4xi4<TOKEN_PARALLEL, PRE_QKVO_W_PARALLEL, HIDDEN_DIM, log2_HIDDEN_DIM, HIDDEN_DIM, MAX_PRE_SEQ_LEN, true>(
            input_stream_kq, wk_wq_stream, output_stream_kq, seq_len, HIDDEN_DIM, KV_HIDDEN_DIM
        );
        cout << "Block_id: " << block_id << " Linear_Layer_i4xi4_kq for k completed." << endl;
        pref_Linear_Layer_i4xi4<TOKEN_PARALLEL, PRE_QKVO_W_PARALLEL, HIDDEN_DIM, log2_HIDDEN_DIM, HIDDEN_DIM, MAX_PRE_SEQ_LEN, true>(
            input_stream_kq, wk_wq_stream, output_stream_kq, seq_len, HIDDEN_DIM, HIDDEN_DIM
        );
        cout << "Block_id: " << block_id << " Linear_Layer_i4xi4_kq for q completed." << endl;
    }
}

void pref_kq_discard(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_io_discard<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL, (HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL * PRE_QKVO_W_PARALLEL, HIDDEN_DIM>(
            input_stream, output_stream, seq_len, ((KV_HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL) * PRE_QKVO_W_PARALLEL, KV_HIDDEN_DIM
        );
        pref_io_discard<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL, (HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL * PRE_QKVO_W_PARALLEL, HIDDEN_DIM>(
            input_stream, output_stream, seq_len, ((HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL) * PRE_QKVO_W_PARALLEL, HIDDEN_DIM
        );
    }
}


void pref_dequant_layer_kq_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream_kq,
    tapa::istream<hls::vector<float, 2>>& input_s_b_stream_kq, //input's scale factor and zero point
    tapa::istream<hls::vector<float, 2>>& wk_wq_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_kq,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_dequant_layer_qint_fp32<log2_HIDDEN_DIM + 8, true, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_kq, input_s_b_stream_kq, wk_wq_s_sum_stream, output_stream_kq, seq_len, KV_HIDDEN_DIM
        );
        pref_dequant_layer_qint_fp32<log2_HIDDEN_DIM + 8, true, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_kq, input_s_b_stream_kq, wk_wq_s_sum_stream, output_stream_kq, seq_len, HIDDEN_DIM
        );
    }
}

void pref_RoPE_layer_kq(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_kq,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_kq,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_RoPE_layer<float, TOKEN_PARALLEL, HEAD_DIM>(input_stream_kq, output_stream_kq, PE_sin, PE_cos, KV_HEAD_NUM, seq_len);
        cout << "Block_id: " << block_id << " RoPE_layer for k completed." << endl;
        pref_RoPE_layer<float, TOKEN_PARALLEL, HEAD_DIM>(input_stream_kq, output_stream_kq, PE_sin, PE_cos, Q_HEAD_NUM, seq_len);
        cout << "Block_id: " << block_id << " RoPE_layer for q completed." << endl;
    }
}

void pref_qk_temporal_distributor(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_kq,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_k,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_q,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_stream_distributor<float, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_kq, output_stream_k, output_stream_q, 0, KV_HIDDEN_DIM, seq_len
        );
        pref_stream_distributor<float, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_kq, output_stream_k, output_stream_q, 1, HIDDEN_DIM, seq_len
        );
    }
}


// reuse Linear Layer for v and o
void pref_vo_temporal_merger(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_v,
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_o,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_vo,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_stream_merger<float, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_v, input_stream_o, output_stream_vo, 0, HIDDEN_DIM, seq_len
        );
        pref_stream_merger<float, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_v, input_stream_o, output_stream_vo, 1, HIDDEN_DIM, seq_len
        );
    }
}


void pref_quant_layer_vo_fp32_int4(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_vo,
    tapa::ostream<hls::vector<float, 2>>& input_s_b_stream_vo, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& output_stream_vo,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_quant_layer_fp32_qint<4, true, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_vo, input_s_b_stream_vo, output_stream_vo, seq_len, HIDDEN_DIM
        ); // for v
        pref_quant_layer_fp32_qint<4, true, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_vo, input_s_b_stream_vo, output_stream_vo, seq_len, HIDDEN_DIM
        ); // for o
    }
}


void pref_weight_loader_wv_wo(
    tapa::mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>> wv_wo_mmap,
    tapa::ostream<hls::vector<ap_int<4>, PRE_QKVO_W_PARALLEL>>& wv_wo_stream, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_weight_loader_int4_pack_2_discard<PRE_QKVO_W_PARALLEL_READ, PRE_QKVO_W_PARALLEL, TOKEN_PARALLEL>(
            wv_wo_mmap, wv_wo_stream, block_id, seq_len, HIDDEN_DIM, KV_HIDDEN_DIM
        );
        pref_weight_loader_int4_pack_2_discard<PRE_QKVO_W_PARALLEL_READ, PRE_QKVO_W_PARALLEL, TOKEN_PARALLEL>(
            wv_wo_mmap, wv_wo_stream, block_id, seq_len, HIDDEN_DIM, HIDDEN_DIM, 
            DECODER_LAYER_NUM * ((KV_HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL) * HIDDEN_DIM
        );
    }
}

void pref_weight_s_loader_wv_wo(
    tapa::mmap<hls::vector<float, 2>> wv_wo_s_sum_mmap,
    tapa::ostream<hls::vector<float, 2>>& wv_wo_s_sum_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_weight_s_loader_fp32<TOKEN_PARALLEL, true>(
            wv_wo_s_sum_mmap, wv_wo_s_sum_stream, block_id, seq_len, KV_HIDDEN_DIM
        );
        pref_weight_s_loader_fp32<TOKEN_PARALLEL, true>(
            wv_wo_s_sum_mmap, wv_wo_s_sum_stream, block_id, seq_len, HIDDEN_DIM, DECODER_LAYER_NUM * KV_HIDDEN_DIM
        );
    }
}


void pref_Linear_Layer_i4xi4_vo(
    tapa::istream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& input_stream_vo,
    tapa::istream<hls::vector<ap_int<4>, PRE_QKVO_W_PARALLEL>>& wv_wo_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream_vo,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_Linear_Layer_i4xi4<TOKEN_PARALLEL, PRE_QKVO_W_PARALLEL, HIDDEN_DIM, log2_HIDDEN_DIM, HIDDEN_DIM, MAX_PRE_SEQ_LEN, true>(
            input_stream_vo, wv_wo_stream, output_stream_vo, seq_len, HIDDEN_DIM, KV_HIDDEN_DIM
        );
        cout << "Block_id: " << block_id << " Linear_Layer_i4xi4_vo for v completed." << endl;
        pref_Linear_Layer_i4xi4<TOKEN_PARALLEL, PRE_QKVO_W_PARALLEL, HIDDEN_DIM, log2_HIDDEN_DIM, HIDDEN_DIM, MAX_PRE_SEQ_LEN, true>(
            input_stream_vo, wv_wo_stream, output_stream_vo, seq_len, HIDDEN_DIM, HIDDEN_DIM
        );
        cout << "Block_id: " << block_id << " Linear_Layer_i4xi4_vo for o completed." << endl;
    }
}

void pref_vo_discard(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_io_discard<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL, (HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL * PRE_QKVO_W_PARALLEL, HIDDEN_DIM>(
            input_stream, output_stream, seq_len, ((KV_HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL) * PRE_QKVO_W_PARALLEL, KV_HIDDEN_DIM
        );
        pref_io_discard<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL, (HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL * PRE_QKVO_W_PARALLEL, HIDDEN_DIM>(
            input_stream, output_stream, seq_len, ((HIDDEN_DIM + PRE_QKVO_W_PARALLEL - 1)/PRE_QKVO_W_PARALLEL) * PRE_QKVO_W_PARALLEL, HIDDEN_DIM
        );
    }
}

void pref_dequant_layer_vo_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream_vo,
    tapa::istream<hls::vector<float, 2>>& input_s_b_stream_vo, //input's scale factor and zero point
    tapa::istream<hls::vector<float, 2>>& wv_wo_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_vo,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_dequant_layer_qint_fp32<log2_HIDDEN_DIM + 8, true, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_vo, input_s_b_stream_vo, wv_wo_s_sum_stream, output_stream_vo, seq_len, KV_HIDDEN_DIM
        );
        pref_dequant_layer_qint_fp32<log2_HIDDEN_DIM + 8, true, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_vo, input_s_b_stream_vo, wv_wo_s_sum_stream, output_stream_vo, seq_len, HIDDEN_DIM
        );
    }
}

void pref_vo_temporal_distributor(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_vo,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_v,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_o,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_stream_distributor<float, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_vo, output_stream_v, output_stream_o, 0, KV_HIDDEN_DIM, seq_len
        );
        pref_stream_distributor<float, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_vo, output_stream_v, output_stream_o, 1, HIDDEN_DIM, seq_len
        );
    }
}

//MHA








// Residual Layer 0
// void pref_Residual_loader_res0(
//     tapa::istream<bool>& res0_cache_ready_stream,
//     tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> res0_cache_mmap,
//     tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& res0_stream,
//     int seq_len
// ){
//     decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
//         pref_Residual_loader<float, TOKEN_PARALLEL, HIDDEN_DIM>(
//             res0_cache_ready_stream, res0_cache_mmap, res0_stream, 0, seq_len
//         );
//     }
// }

void pref_Residual_Layer_0(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_o,
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_iembed,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_res0,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_Residual_Layer<float, TOKEN_PARALLEL, HIDDEN_DIM>(input_stream_o, input_stream_iembed, output_stream_res0, seq_len);
        cout << "Block_id: " << block_id << " Residual_Layer_0 completed." << endl;
    }
}

// cache residual layer 0 output for residual layer 1
void pref_res0_distributor(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_res0,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_ln,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_res1,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_stream_distributor<float, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_res0, output_stream_ln, output_stream_res1, 2, HIDDEN_DIM, seq_len
        );
    }
}

// void pref_Residual_storer_res1(
//     tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& res1_cache_stream,
//     tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> res1_cache_mmap,
//     tapa::ostream<bool>& res1_cache_finish_stream,
//     int seq_len
// ){
//     decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
//         pref_Residual_storer<float, TOKEN_PARALLEL, HIDDEN_DIM>(
//             res1_cache_stream, res1_cache_mmap, res1_cache_finish_stream, 0, seq_len
//         );
//     }
// }

// Layer Norm 1
void pref_Layer_Norm_1_gamma_beta_loader(
    tapa::mmap<float> gamma_beta_mmap,
    tapa::ostream<float>& gamma_beta_stream
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_Layer_Norm_gamma_beta_loader<float, HIDDEN_DIM>(gamma_beta_mmap, gamma_beta_stream, block_id);
    }
}

void pref_Layer_Norm_1(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::istream<float>& gamma_beta_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_Layer_Norm<float, TOKEN_PARALLEL, HIDDEN_DIM>(input_stream, gamma_beta_stream, output_stream, seq_len, HIDDEN_DIM);
        cout << "Block_id: " << block_id << " Layer_Norm_1 completed." << endl;
    }
}

// FFN Gate and FFN Up Layer
void pref_Gate_distributor(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_ffn_gate,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_ffn_up,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_stream_distributor<float, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream, output_stream_ffn_gate, output_stream_ffn_up, 2, HIDDEN_DIM, seq_len
        );
    }
}

void pref_quant_layer_ffn_gate_fp32_int4(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_ffn_gate,
    tapa::ostream<hls::vector<float, 2>>& input_s_b_stream_ffn_gate, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& output_stream_ffn_gate,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_quant_layer_fp32_qint<4, true, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_ffn_gate, input_s_b_stream_ffn_gate, output_stream_ffn_gate, seq_len, HIDDEN_DIM
        );
    }
}

void pref_weight_loader_w_ffn_gate(
    tapa::mmap<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM/2>> w_ffn_gate_mmap,
    tapa::ostream<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM>>& w_ffn_gate_stream, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_weight_loader_int4_pack_2_discard<PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM, PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM, TOKEN_PARALLEL, HIDDEN_DIM, INTER_DIM/PRE_FFN_W_BLOCK_NUM>(
            w_ffn_gate_mmap, w_ffn_gate_stream, block_id, seq_len, HIDDEN_DIM, INTER_DIM/PRE_FFN_W_BLOCK_NUM
        );
    }
}

void pref_weight_s_loader_w_ffn_gate(
    tapa::mmap<hls::vector<float, 2>> w_ffn_gate_s_sum_mmap,
    tapa::ostream<hls::vector<float, 2>>& w_ffn_gate_s_sum_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_weight_s_loader_fp32<TOKEN_PARALLEL, true, INTER_DIM>(
            w_ffn_gate_s_sum_mmap, w_ffn_gate_s_sum_stream, block_id, seq_len, INTER_DIM
        );
    }
}

void pref_Linear_Layer_i4xi4_ffn_gate(
    tapa::istream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& input_stream_ffn_gate,
    tapa::istreams<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM>, PRE_FFN_W_BLOCK_NUM>& w_ffn_gate_streams,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream_ffn_gate,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_Linear_Layer_i4xi4_blocked<TOKEN_PARALLEL, PRE_FFN_W_PARALLEL, PRE_FFN_W_BLOCK_NUM, HIDDEN_DIM, log2_HIDDEN_DIM, INTER_DIM, MAX_PRE_SEQ_LEN, true>(
            input_stream_ffn_gate, w_ffn_gate_streams, output_stream_ffn_gate, seq_len, HIDDEN_DIM, INTER_DIM
        );
        cout << "Block_id: " << block_id << " Linear_Layer_i4xi4_ffn_gate completed." << endl;
    }
}

void pref_ffn_gate_discard(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_io_discard<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL, (INTER_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL * PRE_FFN_W_PARALLEL, INTER_DIM>(
            input_stream, output_stream, seq_len, ((INTER_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * PRE_FFN_W_PARALLEL, INTER_DIM
        );
    }
}

void pref_ffn_gate_output_register(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_io_register<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL, INTER_DIM>(input_stream, output_stream, seq_len);
    }
}

void pref_dequant_layer_ffn_gate_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream_ffn_gate,
    tapa::istream<hls::vector<float, 2>>& input_s_b_stream_ffn_gate, //input's scale factor and zero point
    tapa::istream<hls::vector<float, 2>>& w_ffn_gate_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_ffn_gate,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_dequant_layer_qint_fp32<log2_HIDDEN_DIM + 8, true, TOKEN_PARALLEL, 2, INTER_DIM>(
            input_stream_ffn_gate, input_s_b_stream_ffn_gate, w_ffn_gate_s_sum_stream, output_stream_ffn_gate, seq_len, INTER_DIM
        );
    }
}

void pref_quant_layer_ffn_up_fp32_int4(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_ffn_up,
    tapa::ostream<hls::vector<float, 2>>& input_s_b_stream_ffn_up, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& output_stream_ffn_up,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_quant_layer_fp32_qint<4, true, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream_ffn_up, input_s_b_stream_ffn_up, output_stream_ffn_up, seq_len, HIDDEN_DIM
        );
    }
}

void pref_weight_loader_w_ffn_up(
    tapa::mmap<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM/2>> w_ffn_up_mmap,
    tapa::ostream<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM>>& w_ffn_up_stream, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_weight_loader_int4_pack_2_discard<PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM, PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM, TOKEN_PARALLEL, HIDDEN_DIM, INTER_DIM/PRE_FFN_W_BLOCK_NUM>(
            w_ffn_up_mmap, w_ffn_up_stream, block_id, seq_len, HIDDEN_DIM, INTER_DIM/PRE_FFN_W_BLOCK_NUM
        );
    }
}

void pref_weight_s_loader_w_ffn_up(
    tapa::mmap<hls::vector<float, 2>> w_ffn_up_s_sum_mmap,
    tapa::ostream<hls::vector<float, 2>>& w_ffn_up_s_sum_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_weight_s_loader_fp32<TOKEN_PARALLEL, true, INTER_DIM>(
            w_ffn_up_s_sum_mmap, w_ffn_up_s_sum_stream, block_id, seq_len, INTER_DIM
        );
    }
}

void pref_Linear_Layer_i4xi4_ffn_up(
    tapa::istream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& input_stream_ffn_up,
    tapa::istreams<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM>, PRE_FFN_W_BLOCK_NUM>& w_ffn_up_streams,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream_ffn_up,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_Linear_Layer_i4xi4_blocked<TOKEN_PARALLEL, PRE_FFN_W_PARALLEL, PRE_FFN_W_BLOCK_NUM, HIDDEN_DIM, log2_HIDDEN_DIM, INTER_DIM, MAX_PRE_SEQ_LEN, true>(
            input_stream_ffn_up, w_ffn_up_streams, output_stream_ffn_up, seq_len, HIDDEN_DIM, INTER_DIM
        );
        cout << "Block_id: " << block_id << " Linear_Layer_i4xi4_ffn_up completed." << endl;
    }
}

void pref_ffn_up_discard(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_io_discard<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL, (INTER_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL * PRE_FFN_W_PARALLEL, INTER_DIM>(
            input_stream, output_stream, seq_len, ((INTER_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * PRE_FFN_W_PARALLEL, INTER_DIM
        );
    }
}


void pref_dequant_layer_ffn_up_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream_ffn_up,
    tapa::istream<hls::vector<float, 2>>& input_s_b_stream_ffn_up, //input's scale factor and zero point
    tapa::istream<hls::vector<float, 2>>& w_ffn_up_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_ffn_up,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_dequant_layer_qint_fp32<log2_HIDDEN_DIM + 8, true, TOKEN_PARALLEL, 2, INTER_DIM>(
            input_stream_ffn_up, input_s_b_stream_ffn_up, w_ffn_up_s_sum_stream, output_stream_ffn_up, seq_len, INTER_DIM
        );
    }
}

void pref_Swish_Layer_ffn(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_ffn_up,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_sw_ffn_up,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_Swish<float, TOKEN_PARALLEL, INTER_DIM>(input_stream_ffn_up, output_stream_sw_ffn_up, seq_len);
        cout << "Block_id: " << block_id << " Swish_Layer_ffn completed." << endl;
    }
}

void pref_Gate_Layer_ffn(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_ffn_gate,
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_sw_ffn_up,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_gated_sw_ffn_up,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_Gate_Layer_fp32xfp32<TOKEN_PARALLEL, INTER_DIM>(input_stream_ffn_gate, input_stream_sw_ffn_up, output_stream_gated_sw_ffn_up, seq_len);
        cout << "Block_id: " << block_id << " Gate_Layer_ffn completed." << endl;
    }
}

// Fast Hadmard Transform (R4)
void pref_FHT_R4(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_gated_sw_ffn_up,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_r4_gated_sw_ffn_up,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_FHT<float, TOKEN_PARALLEL, INTER_DIM, log2_INTER_DIM>(
            input_stream_gated_sw_ffn_up, output_stream_r4_gated_sw_ffn_up, seq_len, sqrt_INTER_DIM
        );
        cout << "Block_id: " << block_id << " R4 rotation completed." << endl;
    }
}

// FFN Down Layer
void pref_quant_layer_ffn_down_fp32_int4(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_ffn_down,
    tapa::ostream<hls::vector<float, 2>>& input_s_b_stream_ffn_down, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& output_stream_ffn_down,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_quant_layer_fp32_qint<4, true, TOKEN_PARALLEL, INTER_DIM>(
            input_stream_ffn_down, input_s_b_stream_ffn_down, output_stream_ffn_down, seq_len, INTER_DIM
        );
    }
}

void pref_weight_loader_w_ffn_down(
    tapa::mmap<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM/2>> w_ffn_down_mmap,
    tapa::ostream<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM>>& w_ffn_down_stream, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_weight_loader_int4_pack_2_discard<PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM, PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM, TOKEN_PARALLEL, INTER_DIM, HIDDEN_DIM/PRE_FFN_W_BLOCK_NUM>(
            w_ffn_down_mmap, w_ffn_down_stream, block_id, seq_len, INTER_DIM, HIDDEN_DIM/PRE_FFN_W_BLOCK_NUM
        );
    }
}

void pref_weight_s_loader_w_ffn_down(
    tapa::mmap<hls::vector<float, 2>> w_ffn_down_s_sum_mmap,
    tapa::ostream<hls::vector<float, 2>>& w_ffn_down_s_sum_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_weight_s_loader_fp32<TOKEN_PARALLEL, true, HIDDEN_DIM>(
            w_ffn_down_s_sum_mmap, w_ffn_down_s_sum_stream, block_id, seq_len, HIDDEN_DIM
        );
    }
}

void pref_Linear_Layer_i4xi4_ffn_down(
    tapa::istream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& input_stream_ffn_down,
    tapa::istreams<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM>, PRE_FFN_W_BLOCK_NUM>& w_ffn_down_streams,
    tapa::ostream<hls::vector<ap_int<log2_INTER_DIM + 8>, TOKEN_PARALLEL>>& output_stream_ffn_down,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_Linear_Layer_i4xi4_blocked<TOKEN_PARALLEL, PRE_FFN_W_PARALLEL, PRE_FFN_W_BLOCK_NUM, INTER_DIM, log2_INTER_DIM, HIDDEN_DIM, MAX_PRE_SEQ_LEN, true>(
            input_stream_ffn_down, w_ffn_down_streams, output_stream_ffn_down, seq_len, INTER_DIM, HIDDEN_DIM
        );
        cout << "Block_id: " << block_id << " Linear_Layer_i4xi4_ffn_down completed." << endl;
    }
}

void pref_ffn_down_discard(
    tapa::istream<hls::vector<ap_int<log2_INTER_DIM + 8>, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<log2_INTER_DIM + 8>, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_io_discard<ap_int<log2_INTER_DIM + 8>, TOKEN_PARALLEL, (HIDDEN_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL * PRE_FFN_W_PARALLEL, HIDDEN_DIM>(
            input_stream, output_stream, seq_len, ((HIDDEN_DIM + PRE_FFN_W_PARALLEL - 1)/PRE_FFN_W_PARALLEL) * PRE_FFN_W_PARALLEL, HIDDEN_DIM
        );
    }
}

void pref_dequant_layer_ffn_down_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_INTER_DIM + 8>, TOKEN_PARALLEL>>& input_stream_ffn_down,
    tapa::istream<hls::vector<float, 2>>& input_s_b_stream_ffn_down, //input's scale factor and zero point
    tapa::istream<hls::vector<float, 2>>& w_ffn_down_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_ffn_down,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_dequant_layer_qint_fp32<log2_INTER_DIM + 8, true, TOKEN_PARALLEL, 2, HIDDEN_DIM>(
            input_stream_ffn_down, input_s_b_stream_ffn_down, w_ffn_down_s_sum_stream, output_stream_ffn_down, seq_len, HIDDEN_DIM
        );
    }
}


// Residual Layer 1
// void pref_Residual_loader_res1(
//     tapa::istream<bool>& res1_cache_ready_stream,
//     tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> res1_cache_mmap,
//     tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& res1_stream,
//     int seq_len
// ){
//     decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
//         pref_Residual_loader<float, TOKEN_PARALLEL, HIDDEN_DIM>(
//             res1_cache_ready_stream, res1_cache_mmap, res1_stream, 0, seq_len
//         );
//     }
// }


void pref_Residual_Layer_1(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_ffn_down,
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_res0,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_res1,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_Residual_Layer<float, TOKEN_PARALLEL, HIDDEN_DIM>(input_stream_ffn_down, input_stream_res0, output_stream_res1, seq_len);
        cout << "Block_id: " << block_id << " Residual_Layer_1 completed." << endl;
    }
}


void pref_block_output_drainer_sync(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_res1,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> io_mmap,
    tapa::ostream<bool>& block_output_finish_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_output_drainer<float, TOKEN_PARALLEL>(output_stream_res1, io_mmap, block_id + 1, seq_len, HIDDEN_DIM); 
        block_output_finish_stream.write(true); // signal that all outputs are ready
        cout << "Block_id: " << block_id << " Block output drainer completed." << endl;
    }
}


void SpinQuant_Prefilling(
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> io_mmap,
    tapa::mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>> wk_wq_mmap,
    tapa::mmap<hls::vector<float, 2>> wk_wq_s_sum_mmap,
    tapa::mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>> wv_wo_mmap,
    tapa::mmap<hls::vector<float, 2>> wv_wo_s_sum_mmap,
    tapa::mmap<hls::vector<ap_int<8>, PRE_K_PARALLEL>> k_cache,
    tapa::mmap<hls::vector<ap_int<8>, PRE_V_PARALLEL>> v_cache,
    tapa::mmaps<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM/2>, PRE_FFN_W_BLOCK_NUM> w_ffn_gate_mmaps,
    tapa::mmap<hls::vector<float, 2>> w_ffn_gate_s_sum_mmap,
    tapa::mmaps<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM/2>, PRE_FFN_W_BLOCK_NUM> w_ffn_up_mmaps,
    tapa::mmap<hls::vector<float, 2>> w_ffn_up_s_sum_mmap,
    tapa::mmaps<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/PRE_FFN_W_BLOCK_NUM/2>, PRE_FFN_W_BLOCK_NUM> w_ffn_down_mmaps,
    tapa::mmap<hls::vector<float, 2>> w_ffn_down_s_sum_mmap,
    tapa::mmap<float> gamma_beta_mmap_0,
    tapa::mmap<float> gamma_beta_mmap_1,
    // tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> res0_cache_mmap,
    // tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> res1_cache_mmap,
    int seq_len
){
    tapa::stream<bool> block_input_ready_stream("block_input_ready_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> iembed_stream("iembed_stream");

    // cache input embedding for residual layer 0
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> iembed_stream_ln("iembed_stream_ln");
    // tapa::stream<hls::vector<float, TOKEN_PARALLEL>> iembed_stream_res0_cache("iembed_stream_res0_cache");
    // tapa::stream<bool, MAX_PRE_SEQ_LEN/TOKEN_PARALLEL> res0_cache_finish_stream("res0_cache_finish_stream");

    // Layer Norm 0
    tapa::stream<float> gamma_beta_stream_0("gamma_beta_stream_0");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> ln_iembed_stream("ln_iembed_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> ln_iembed_stream_kq("ln_iembed_stream_kq");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> ln_iembed_stream_v("ln_iembed_stream_v");

    //Linear Layer for QKVO
    tapa::stream<hls::vector<float, 2>, 4 * TOKEN_PARALLEL> ln_iembed_s_b_stream_kq("ln_iembed_s_b_stream_kq");
    tapa::stream<hls::vector<ap_int<4>, TOKEN_PARALLEL>, HIDDEN_DIM> quant_ln_iembed_stream_kq(" quant_ln_iembed_stream_kq");
    tapa::stream<hls::vector<ap_int<4>, PRE_QKVO_W_PARALLEL>> wk_wq_stream("wk_wq_stream");
    tapa::stream<hls::vector<float, 2>> wk_wq_s_sum_stream("wk_wq_s_sum_stream");
    tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> quant_kq_stream_redundant("quant_kq_stream_redundant");
    tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>, HIDDEN_DIM> quant_kq_stream("quant_kq_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> kq_stream("kq_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> RoPE_kq_stream("RoPE_kq_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> k_stream("k_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> q_stream("q_stream");

    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> merged_stream_vo("merged_stream_vo");
    tapa::stream<hls::vector<float, 2>, 4 * TOKEN_PARALLEL> merged_s_b_stream_vo("quant_merged_stream_vo");
    tapa::stream<hls::vector<ap_int<4>, TOKEN_PARALLEL>, HIDDEN_DIM> quant_merged_stream_vo("quant_merged_stream_vo");
    tapa::stream<hls::vector<ap_int<4>, PRE_QKVO_W_PARALLEL>> wv_wo_stream("wv_wo_stream");
    tapa::stream<hls::vector<float, 2>> wv_wo_s_sum_stream("wv_wo_s_sum_stream");
    tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> quant_vo_stream_redundant("quant_vo_stream_redundant");
    tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>, HIDDEN_DIM> quant_vo_stream("quant_vo_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> vo_stream("vo_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> v_stream("v_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> o_stream("o_stream");

    //MHA
    tapa::stream<hls::vector<ap_int<8>, TOKEN_PARALLEL>> quant_q_stream("quant_q_stream");

    tapa::stream<hls::vector<ap_int<8>, TOKEN_PARALLEL>> quant_k_stream("quant_k_stream");
    tapa::stream<hls::vector<ap_int<8>, PRE_K_PARALLEL>> cache_quant_k_stream("cache_quant_k_stream");
    tapa::stream<hls::vector<ap_int<8>, PRE_K_PARALLEL>> load_quant_k_stream("load_quant_k_stream");

    tapa::stream<hls::vector<ap_int<log2_HEAD_DIM + 16>, TOKEN_PARALLEL>, MAX_PRE_SEQ_LEN> quant_a_stream("quant_a_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> a_stream("a_stream");

    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> masked_a_stream("masked_a_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> sfm_a_stream("sfm_a_stream");
    tapa::stream<hls::vector<ap_int<8>, TOKEN_PARALLEL>, MAX_PRE_SEQ_LEN> quant_sfm_a_stream("quant_sfm_a_stream");

    tapa::stream<hls::vector<ap_int<8>, TOKEN_PARALLEL>> quant_v_stream("quant_v_stream");
    tapa::stream<hls::vector<ap_int<8>, PRE_V_PARALLEL>> cache_quant_v_stream("cache_quant_v_stream");
    tapa::stream<hls::vector<ap_int<8>, PRE_V_PARALLEL>> load_quant_v_stream("load_quant_v_stream");

    tapa::stream<hls::vector<ap_int<log2_MAX_PRE_SEQ_LEN + 16>, TOKEN_PARALLEL>> quant_input_stream_o("quant_input_stream_o");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> input_stream_o("input_stream_o");


    // Residual Layer 0 and Layer Norm 1
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>, 8 * HIDDEN_DIM> iembed_stream_res0("iembed_stream_res0");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> res0_stream("res0_stream");

    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> res0_stream_ln("res0_stream_ln");
    // tapa::stream<hls::vector<float, TOKEN_PARALLEL>> res0_stream_res1_cache("res0_stream_res1_cache");
    // tapa::stream<bool, MAX_PRE_SEQ_LEN/TOKEN_PARALLEL> res1_cache_finish_stream;

    tapa::stream<float> gamma_beta_stream_1("gamma_beta_stream_1");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> ln_res0_stream("ln_res0_stream");

    // FFN Gate and Up Layer + swish
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> ln_res0_stream_ffn_gate("ln_res0_stream_ffn_gate");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> ln_res0_stream_ffn_up("ln_res0_stream_ffn_up");

    tapa::stream<hls::vector<float, 2>, 2 * TOKEN_PARALLEL> ln_res0_s_b_stream_ffn_gate("ln_res0_s_b_stream_ffn_gate");
    tapa::stream<hls::vector<ap_int<4>, TOKEN_PARALLEL>> quant_ln_res0_stream_ffn_gate("quant_ln_res0_stream_ffn_gate");

    tapa::streams<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM>, PRE_FFN_W_BLOCK_NUM> w_ffn_gate_streams("w_ffn_gate_streams");
    tapa::stream<hls::vector<float, 2>> w_ffn_gate_s_sum_stream("w_ffn_gate_s_sum_stream");

    tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> quant_ffn_gate_stream_redundant("quant_ffn_gate_stream_redundant");
    tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> quant_ffn_gate_stream("quant_ffn_gate_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> ffn_gate_stream("ffn_gate_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> sw_ffn_gate_stream("sw_ffn_gate_stream");

    tapa::stream<hls::vector<float, 2>, 2 * TOKEN_PARALLEL> ln_res0_s_b_stream_ffn_up("ln_res0_s_b_stream_ffn_up");
    tapa::stream<hls::vector<ap_int<4>, TOKEN_PARALLEL>> quant_ln_res0_stream_ffn_up("quant_ln_res0_stream_ffn_up");

    tapa::streams<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM>, PRE_FFN_W_BLOCK_NUM> w_ffn_up_streams("w_ffn_up_streams");
    tapa::stream<hls::vector<float, 2>> w_ffn_up_s_sum_stream("w_ffn_up_s_sum_stream");

    tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> quant_ffn_up_stream_redundant("quant_ffn_up_stream_redundant");
    tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> quant_ffn_up_stream("quant_ffn_up_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> ffn_up_stream("ffn_up_stream");
    

    // Gate Layer for FFN UP
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>, INTER_DIM> gated_sw_ffn_up_stream("gated_sw_ffn_up_stream");

    // R4 Rotation
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> r4_gated_sw_ffn_up_stream("r4_gated_sw_ffn_up_stream");

    // FFN Down Layer
    tapa::stream<hls::vector<float, 2>, 4 * TOKEN_PARALLEL> r4_gated_sw_ffn_up_s_b_stream("gate_ffn_up_s_b_stream");
    tapa::stream<hls::vector<ap_int<4>, TOKEN_PARALLEL>, INTER_DIM> quant_r4_gated_sw_ffn_up_stream("quant_gated_sw_ffn_up_stream");

    tapa::streams<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL/PRE_FFN_W_BLOCK_NUM>, PRE_FFN_W_BLOCK_NUM> w_ffn_down_streams("w_ffn_down_streams");
    tapa::stream<hls::vector<float, 2>> w_ffn_down_s_sum_stream("w_ffn_down_s_sum_stream");

    tapa::stream<hls::vector<ap_int<log2_INTER_DIM + 8>, TOKEN_PARALLEL>> quant_ffn_down_stream_redundant("quant_ffn_down_stream_redundant");
    tapa::stream<hls::vector<ap_int<log2_INTER_DIM + 8>, TOKEN_PARALLEL>, HIDDEN_DIM> quant_ffn_down_stream("quant_ffn_down_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> ffn_down_stream("ffn_down_stream");


    // Residual Layer 1
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>, 8 * HIDDEN_DIM> res0_stream_res1("res0_stream_res1");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> res1_stream("res1_stream");


    tapa::task()
    .invoke(pref_block_input_loader_sync, block_input_ready_stream, io_mmap, iembed_stream, seq_len)


    // cache input embedding for residual layer 0
    .invoke(pref_iembed_distributor, iembed_stream, iembed_stream_ln, iembed_stream_res0, seq_len)
    // .invoke(pref_Residual_storer_res0, iembed_stream_res0_cache, res0_cache_mmap, res0_cache_finish_stream, seq_len)

    // Layer Norm 0
    .invoke(pref_Layer_Norm_0_gamma_beta_loader, gamma_beta_mmap_0, gamma_beta_stream_0)
    .invoke(pref_Layer_Norm_0, iembed_stream_ln, gamma_beta_stream_0, ln_iembed_stream, seq_len)
    .invoke(pref_LN_iembed_temporal_distributor, ln_iembed_stream, ln_iembed_stream_kq, ln_iembed_stream_v, seq_len)

    // Linear Layer for QKVO
    .invoke(pref_quant_layer_kq_fp32_int4, ln_iembed_stream_kq, ln_iembed_s_b_stream_kq, quant_ln_iembed_stream_kq, seq_len)
    .invoke(pref_weight_loader_wk_wq, wk_wq_mmap, wk_wq_stream, seq_len)
    .invoke(pref_weight_s_loader_wk_wq, wk_wq_s_sum_mmap, wk_wq_s_sum_stream, seq_len)
    // .invoke(pref_Linear_Layer_i4xi4_kq, quant_ln_iembed_stream_kq, wk_wq_stream, quant_kq_stream, seq_len)
    .invoke(pref_Linear_Layer_i4xi4_kq, quant_ln_iembed_stream_kq, wk_wq_stream, quant_kq_stream_redundant, seq_len)
    .invoke(pref_kq_discard, quant_kq_stream_redundant, quant_kq_stream, seq_len)
    .invoke(pref_dequant_layer_kq_int_fp32, quant_kq_stream, ln_iembed_s_b_stream_kq, wk_wq_s_sum_stream, kq_stream, seq_len)          
    .invoke(pref_RoPE_layer_kq, kq_stream, RoPE_kq_stream, seq_len)
    .invoke(pref_qk_temporal_distributor, RoPE_kq_stream, k_stream, q_stream, seq_len)

    .invoke(pref_vo_temporal_merger, ln_iembed_stream_v, input_stream_o, merged_stream_vo, seq_len)
    .invoke(pref_quant_layer_vo_fp32_int4, merged_stream_vo, merged_s_b_stream_vo, quant_merged_stream_vo, seq_len)
    .invoke(pref_weight_loader_wv_wo, wv_wo_mmap, wv_wo_stream, seq_len)
    .invoke(pref_weight_s_loader_wv_wo, wv_wo_s_sum_mmap, wv_wo_s_sum_stream, seq_len)
    // .invoke(pref_Linear_Layer_i4xi4_vo, quant_merged_stream_vo, wv_wo_stream, quant_vo_stream, seq_len)
    .invoke(pref_Linear_Layer_i4xi4_vo, quant_merged_stream_vo, wv_wo_stream, quant_vo_stream_redundant, seq_len)
    .invoke(pref_vo_discard, quant_vo_stream_redundant, quant_vo_stream, seq_len)
    .invoke(pref_dequant_layer_vo_int_fp32, quant_vo_stream, merged_s_b_stream_vo, wv_wo_s_sum_stream, vo_stream, seq_len)
    .invoke(pref_vo_temporal_distributor, vo_stream, v_stream, o_stream, seq_len)


    // MHA
    .invoke(pref_quant_layer_k_fp32_int8, k_stream, quant_k_stream, seq_len)
    .invoke(pref_K_buffer, quant_k_stream, cache_quant_k_stream, seq_len)
    .invoke(pref_K_cache_manager, cache_quant_k_stream, k_cache, load_quant_k_stream, seq_len)

    .invoke(pref_quant_layer_v_fp32_int8, v_stream, quant_v_stream, seq_len)
    .invoke(pref_V_buffer_transpose, quant_v_stream, cache_quant_v_stream, seq_len)
    .invoke(pref_V_cache_manager, cache_quant_v_stream, v_cache, load_quant_v_stream, seq_len)

    .invoke(pref_quant_layer_q_fp32_int8, q_stream, quant_q_stream, seq_len)
    .invoke(pref_MHA_i8xi8_qxk, quant_q_stream, load_quant_k_stream, quant_a_stream, seq_len)
    .invoke(pref_dequant_layer_a_int_fp32, quant_a_stream, a_stream, seq_len)

    .invoke(pref_causal_mask, a_stream, masked_a_stream, seq_len)
    .invoke(pref_Softmax_MHA, masked_a_stream, sfm_a_stream, seq_len)
    .invoke(pref_quant_layer_sfm_a_fp32_int8, sfm_a_stream, quant_sfm_a_stream, seq_len)

    .invoke(pref_MHA_i8xi8_axv, quant_sfm_a_stream, load_quant_v_stream, quant_input_stream_o, seq_len)
    .invoke(pref_dequant_layer_o_int_fp32, quant_input_stream_o, input_stream_o, seq_len)

    // Residual Layer 0
    // .invoke(pref_Residual_loader_res0, res0_cache_finish_stream, res0_cache_mmap, iembed_stream_res0, seq_len)
    .invoke(pref_Residual_Layer_0, o_stream, iembed_stream_res0, res0_stream, seq_len)
    // .invoke(pref_block_output_drainer_sync, res1_stream, io_mmap, block_input_ready_stream, seq_len)

    .invoke(pref_res0_distributor, res0_stream, res0_stream_ln, res0_stream_res1, seq_len)
    // .invoke(pref_Residual_storer_res1, res0_stream_res1_cache, res1_cache_mmap, res1_cache_finish_stream, seq_len)

    // Layer Norm 1
    .invoke(pref_Layer_Norm_1_gamma_beta_loader, gamma_beta_mmap_1, gamma_beta_stream_1)
    .invoke(pref_Layer_Norm_1, res0_stream_ln, gamma_beta_stream_1, ln_res0_stream, seq_len)

    // FFN Gate and Up Layer + swish
    .invoke(pref_Gate_distributor, ln_res0_stream, ln_res0_stream_ffn_gate, ln_res0_stream_ffn_up, seq_len)

    .invoke(pref_quant_layer_ffn_gate_fp32_int4, ln_res0_stream_ffn_gate, ln_res0_s_b_stream_ffn_gate, quant_ln_res0_stream_ffn_gate, seq_len)
    .invoke<tapa::detach, PRE_FFN_W_BLOCK_NUM>(pref_weight_loader_w_ffn_gate, w_ffn_gate_mmaps, w_ffn_gate_streams, seq_len)
    .invoke(pref_weight_s_loader_w_ffn_gate, w_ffn_gate_s_sum_mmap, w_ffn_gate_s_sum_stream, seq_len)
    // .invoke(Linear_Layer_i4xi4_ffn_gate, quant_ln_res0_stream_ffn_gate, w_ffn_gate_streams, quant_ffn_gate_stream, seq_len)
    .invoke(pref_Linear_Layer_i4xi4_ffn_gate, quant_ln_res0_stream_ffn_gate, w_ffn_gate_streams, quant_ffn_gate_stream_redundant, seq_len)
    .invoke(pref_ffn_gate_discard, quant_ffn_gate_stream_redundant, quant_ffn_gate_stream, seq_len)
    .invoke(pref_dequant_layer_ffn_gate_int_fp32, quant_ffn_gate_stream, ln_res0_s_b_stream_ffn_gate, w_ffn_gate_s_sum_stream, ffn_gate_stream, seq_len)
    .invoke(pref_Swish_Layer_ffn, ffn_gate_stream, sw_ffn_gate_stream, seq_len)

    .invoke(pref_quant_layer_ffn_up_fp32_int4, ln_res0_stream_ffn_up, ln_res0_s_b_stream_ffn_up, quant_ln_res0_stream_ffn_up, seq_len)
    .invoke<tapa::detach, PRE_FFN_W_BLOCK_NUM>(pref_weight_loader_w_ffn_up, w_ffn_up_mmaps, w_ffn_up_streams, seq_len)
    .invoke(pref_weight_s_loader_w_ffn_up, w_ffn_up_s_sum_mmap, w_ffn_up_s_sum_stream, seq_len)
    // .invoke(Linear_Layer_i4xi4_ffn_up, quant_ln_res0_stream_ffn_up, w_ffn_up_streams, quant_ffn_up_stream, seq_len)
    .invoke(pref_Linear_Layer_i4xi4_ffn_up, quant_ln_res0_stream_ffn_up, w_ffn_up_streams, quant_ffn_up_stream_redundant, seq_len)
    .invoke(pref_ffn_up_discard, quant_ffn_up_stream_redundant, quant_ffn_up_stream, seq_len)
    .invoke(pref_dequant_layer_ffn_up_int_fp32, quant_ffn_up_stream, ln_res0_s_b_stream_ffn_up, w_ffn_up_s_sum_stream, ffn_up_stream, seq_len)
    

    // Gate Layer for FFN UP
    .invoke(pref_Gate_Layer_ffn, sw_ffn_gate_stream, ffn_up_stream, gated_sw_ffn_up_stream, seq_len)

    // R4 Rotation
    .invoke(pref_FHT_R4, gated_sw_ffn_up_stream, r4_gated_sw_ffn_up_stream, seq_len)

    // FFN Down Layer
    .invoke(pref_quant_layer_ffn_down_fp32_int4, r4_gated_sw_ffn_up_stream, r4_gated_sw_ffn_up_s_b_stream, quant_r4_gated_sw_ffn_up_stream, seq_len)
    .invoke<tapa::detach, PRE_FFN_W_BLOCK_NUM>(pref_weight_loader_w_ffn_down, w_ffn_down_mmaps, w_ffn_down_streams, seq_len)
    .invoke(pref_weight_s_loader_w_ffn_down, w_ffn_down_s_sum_mmap, w_ffn_down_s_sum_stream, seq_len)
    // .invoke(pref_Linear_Layer_i4xi4_ffn_down, quant_r4_gated_sw_ffn_up_stream, w_ffn_down_streams, quant_ffn_down_stream, seq_len)
    .invoke(pref_Linear_Layer_i4xi4_ffn_down, quant_r4_gated_sw_ffn_up_stream, w_ffn_down_streams, quant_ffn_down_stream_redundant, seq_len)
    .invoke(pref_ffn_down_discard, quant_ffn_down_stream_redundant, quant_ffn_down_stream, seq_len)
    .invoke(pref_dequant_layer_ffn_down_int_fp32, quant_ffn_down_stream, r4_gated_sw_ffn_up_s_b_stream, w_ffn_down_s_sum_stream, ffn_down_stream, seq_len)


    // Residual Layer 1
    // .invoke(pref_Residual_loader_res1, res1_cache_finish_stream, res1_cache_mmap, res0_stream_res1, seq_len)
    .invoke(pref_Residual_Layer_1, ffn_down_stream, res0_stream_res1, res1_stream, seq_len)
    .invoke(pref_block_output_drainer_sync, res1_stream, io_mmap, block_input_ready_stream, seq_len)
    
    ;
}

#endif




