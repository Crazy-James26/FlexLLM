#ifndef _HMT_SQ_PREF_H_
#define _HMT_SQ_PREF_H_

#include "HMT_SpinQuant_Unit.h"
#include "SpinQuant_Prefilling.h"

#define SpinQuant_Pref_module_num 58


void hmt_pref_control(
    tapa::istream<int>& seg_len_stream,
    tapa::ostreams<int, SpinQuant_Pref_module_num>& seg_len_stream_copies
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        for(int i = 0; i < SpinQuant_Pref_module_num; i++){
        #pragma HLS UNROLL
            while(! seg_len_stream_copies[i].try_write(seg_len)) {};
        }
    }
    
    // end signal
    for(int i = 0; i < SpinQuant_Pref_module_num; i++){
    #pragma HLS UNROLL
        while(! seg_len_stream_copies[i].try_write(0)) {};
    }
}


void hmt_pref_block_input_loader_sync(
    tapa::istream<bool>& hmt_stage_01_ready_stream,
    tapa::istream<bool>& block_input_ready_stream,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> pref_io_mmap,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& input_stream, 
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        bool stage_01_ready = hmt_stage_01_ready_stream.read();
        pref_block_input_loader_sync(
            block_input_ready_stream, pref_io_mmap, input_stream, seg_len
        );
    }
}

void hmt_pref_iembed_distributor(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_ln,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_res0,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_iembed_distributor(input_stream, output_stream_ln, output_stream_res0, seg_len);
    }
}

// Layer Norm 0
void hmt_pref_Layer_Norm_0_gamma_beta_loader(
    tapa::mmap<float> gamma_beta_mmap,
    tapa::ostream<float>& gamma_beta_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_Layer_Norm_0_gamma_beta_loader(gamma_beta_mmap, gamma_beta_stream);
    }
}

void hmt_pref_Layer_Norm_0(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::istream<float>& gamma_beta_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_Layer_Norm_0(input_stream, gamma_beta_stream, output_stream, seg_len);
    }
}


// Linear Layer QKVO
// reuse Linear Layer for k and q
void hmt_pref_LN_iembed_temporal_distributor(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_kq,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_v,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_LN_iembed_temporal_distributor(input_stream, output_stream_kq, output_stream_v, seg_len);
    }
}


void hmt_pref_quant_layer_kq_fp32_int4(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_kq,
    tapa::ostream<hls::vector<float, 2>>& input_s_b_stream_kq, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& output_stream_kq,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_quant_layer_kq_fp32_int4(input_stream_kq, input_s_b_stream_kq, output_stream_kq, seg_len);
    }
}


void hmt_pref_weight_loader_wk_wq(
    tapa::mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>> wk_wq_mmap,
    tapa::ostream<hls::vector<ap_int<4>, PRE_QKVO_W_PARALLEL>>& wk_wq_stream, 
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_weight_loader_wk_wq(wk_wq_mmap, wk_wq_stream, seg_len);
    }
}

void hmt_pref_weight_s_loader_wk_wq(
    tapa::mmap<hls::vector<float, 2>> wk_wq_s_sum_mmap,
    tapa::ostream<hls::vector<float, 2>>& wk_wq_s_sum_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_weight_s_loader_wk_wq(wk_wq_s_sum_mmap, wk_wq_s_sum_stream, seg_len);
    }
}


void hmt_pref_Linear_Layer_i4xi4_kq(
    tapa::istream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& input_stream_kq,
    tapa::istream<hls::vector<ap_int<4>, PRE_QKVO_W_PARALLEL>>& wk_wq_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream_kq,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_Linear_Layer_i4xi4_kq(input_stream_kq, wk_wq_stream, output_stream_kq, seg_len);
    }
}

void hmt_pref_kq_discard(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_kq_discard(input_stream, output_stream, seg_len);
    }
}


void hmt_pref_dequant_layer_kq_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream_kq,
    tapa::istream<hls::vector<float, 2>>& input_s_b_stream_kq, //input's scale factor and zero point
    tapa::istream<hls::vector<float, 2>>& wk_wq_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_kq,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_dequant_layer_kq_int_fp32(input_stream_kq, input_s_b_stream_kq, wk_wq_s_sum_stream, output_stream_kq, seg_len);
    }
}

void hmt_pref_RoPE_layer_kq(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_kq,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_kq,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_RoPE_layer_kq(input_stream_kq, output_stream_kq, seg_len);
    }
}

void hmt_pref_qk_temporal_distributor(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_kq,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_k,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_q,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_qk_temporal_distributor(input_stream_kq, output_stream_k, output_stream_q, seg_len);
    }
}


// reuse Linear Layer for v and o
void hmt_pref_vo_temporal_merger(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_v,
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_o,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_vo,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_vo_temporal_merger(input_stream_v, input_stream_o, output_stream_vo, seg_len);
    }
}


void hmt_pref_quant_layer_vo_fp32_int4(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_vo,
    tapa::ostream<hls::vector<float, 2>>& input_s_b_stream_vo, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& output_stream_vo,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_quant_layer_vo_fp32_int4(input_stream_vo, input_s_b_stream_vo, output_stream_vo, seg_len);
    }
}


void hmt_pref_weight_loader_wv_wo(
    tapa::mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>> wv_wo_mmap,
    tapa::ostream<hls::vector<ap_int<4>, PRE_QKVO_W_PARALLEL>>& wv_wo_stream, 
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_weight_loader_wv_wo(wv_wo_mmap, wv_wo_stream, seg_len);
    }
}

void hmt_pref_weight_s_loader_wv_wo(
    tapa::mmap<hls::vector<float, 2>> wv_wo_s_sum_mmap,
    tapa::ostream<hls::vector<float, 2>>& wv_wo_s_sum_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_weight_s_loader_wv_wo(wv_wo_s_sum_mmap, wv_wo_s_sum_stream, seg_len);
    }
}


void hmt_pref_Linear_Layer_i4xi4_vo(
    tapa::istream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& input_stream_vo,
    tapa::istream<hls::vector<ap_int<4>, PRE_QKVO_W_PARALLEL>>& wv_wo_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream_vo,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_Linear_Layer_i4xi4_vo(input_stream_vo, wv_wo_stream, output_stream_vo, seg_len);
    }
}

void hmt_pref_vo_discard(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_vo_discard(input_stream, output_stream, seg_len);
    }
}

void hmt_pref_dequant_layer_vo_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream_vo,
    tapa::istream<hls::vector<float, 2>>& input_s_b_stream_vo, //input's scale factor and zero point
    tapa::istream<hls::vector<float, 2>>& wv_wo_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_vo,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_dequant_layer_vo_int_fp32(input_stream_vo, input_s_b_stream_vo, wv_wo_s_sum_stream, output_stream_vo, seg_len);
    }
}

void hmt_pref_vo_temporal_distributor(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_vo,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_v,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_o,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_vo_temporal_distributor(input_stream_vo, output_stream_v, output_stream_o, seg_len);
    }
}

//MHA
void hmt_pref_quant_layer_k_fp32_int8(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_quant_layer_k_fp32_int8(input_stream, output_stream, seg_len);
    }
}

void hmt_pref_K_buffer(
    tapa::istream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& input_k_stream,
    tapa::ostream<hls::vector<ap_int<8>, PRE_K_PARALLEL>>& output_k_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_K_buffer(input_k_stream, output_k_stream, seg_len);
    }
}


void hmt_pref_K_cache_manager(
    tapa::istream<hls::vector<ap_int<8>, PRE_K_PARALLEL>>& input_k_stream,
    tapa::mmap<hls::vector<ap_int<8>, PRE_K_PARALLEL>> k_cache,
    tapa::ostream<hls::vector<ap_int<8>, PRE_K_PARALLEL>>& output_k_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_K_cache_manager(input_k_stream, k_cache, output_k_stream, seg_len);
    }
}

void hmt_pref_quant_layer_v_fp32_int8(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_quant_layer_v_fp32_int8(input_stream, output_stream, seg_len);
    }
}

void hmt_pref_V_buffer_transpose(
    tapa::istream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& input_v_stream,
    tapa::ostream<hls::vector<ap_int<8>, PRE_V_PARALLEL>>& output_v_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_V_buffer_transpose(input_v_stream, output_v_stream, seg_len);
    }
}

void hmt_pref_V_cache_manager(
    tapa::istream<hls::vector<ap_int<8>, PRE_V_PARALLEL>>& input_v_stream,
    tapa::mmap<hls::vector<ap_int<8>, PRE_V_PARALLEL>> v_cache,
    tapa::ostream<hls::vector<ap_int<8>, PRE_V_PARALLEL>>& output_v_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_V_cache_manager(input_v_stream, v_cache, output_v_stream, seg_len);
    }
}


void hmt_pref_quant_layer_q_fp32_int8(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    // tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& input_s_stream, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_quant_layer_q_fp32_int8(input_stream, output_stream, seg_len);
    }
}


void hmt_pref_MHA_i8xi8_qxk(
    tapa::istream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& input_seq,
    tapa::istream<hls::vector<ap_int<8>, PRE_K_PARALLEL>>& weight_loader,
    tapa::ostream<hls::vector<ap_int<log2_HEAD_DIM + 16>, TOKEN_PARALLEL>>& output_seq,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_MHA_i8xi8_qxk(input_seq, weight_loader, output_seq, seg_len);
    }
}

void hmt_pref_dequant_layer_a_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_HEAD_DIM + 16>, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_dequant_layer_a_int_fp32(input_stream, output_stream, seg_len);
    }
}

void hmt_pref_causal_mask(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_causal_mask(input_stream, output_stream, seg_len);
    }
}

void hmt_pref_Softmax_MHA(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_Softmax_MHA(input_stream, output_stream, seg_len);
    }
}


void hmt_pref_quant_layer_sfm_a_fp32_int8(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_quant_layer_sfm_a_fp32_int8(input_stream, output_stream, seg_len);
    }
}

void hmt_pref_MHA_i8xi8_axv(
    tapa::istream<hls::vector<ap_int<8>, TOKEN_PARALLEL>>& input_seq,
    tapa::istream<hls::vector<ap_int<8>, PRE_V_PARALLEL>>& weight_loader,
    tapa::ostream<hls::vector<ap_int<log2_MAX_PRE_SEQ_LEN + 16>, TOKEN_PARALLEL>>& output_seq,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_MHA_i8xi8_axv(input_seq, weight_loader, output_seq, seg_len);
    }
}


void hmt_pref_dequant_layer_o_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_MAX_PRE_SEQ_LEN + 16>, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_dequant_layer_o_int_fp32(input_stream, output_stream, seg_len);
    }
}



// Residual Layer 0

void hmt_pref_Residual_Layer_0(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_o,
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_iembed,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_res0,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_Residual_Layer_0(input_stream_o, input_stream_iembed, output_stream_res0, seg_len);
    }
}

// cache residual layer 0 output for residual layer 1
void hmt_pref_res0_distributor(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_res0,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_ln,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_res1,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_res0_distributor(input_stream_res0, output_stream_ln, output_stream_res1, seg_len);
    }
}

// Layer Norm 1
void hmt_pref_Layer_Norm_1_gamma_beta_loader(
    tapa::mmap<float> gamma_beta_mmap,
    tapa::ostream<float>& gamma_beta_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_Layer_Norm_1_gamma_beta_loader(gamma_beta_mmap, gamma_beta_stream);
    }
}

void hmt_pref_Layer_Norm_1(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::istream<float>& gamma_beta_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_Layer_Norm_1(input_stream, gamma_beta_stream, output_stream, seg_len);
    }
}

// FFN Gate and FFN Up Layer
void hmt_pref_Gate_distributor(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_ffn_gate,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_ffn_up,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_Gate_distributor(input_stream, output_stream_ffn_gate, output_stream_ffn_up, seg_len);
    }
}

void hmt_pref_quant_layer_ffn_gate_fp32_int4(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_ffn_gate,
    tapa::ostream<hls::vector<float, 2>>& input_s_b_stream_ffn_gate, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& output_stream_ffn_gate,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_quant_layer_ffn_gate_fp32_int4(input_stream_ffn_gate, input_s_b_stream_ffn_gate, output_stream_ffn_gate, seg_len);
    }
}

void hmt_pref_weight_loader_w_ffn_gate(
    tapa::mmap<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>> w_ffn_gate_mmap,
    tapa::ostream<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL>>& w_ffn_gate_stream, 
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_weight_loader_w_ffn_gate(w_ffn_gate_mmap, w_ffn_gate_stream, seg_len);
    }
}

void hmt_pref_weight_s_loader_w_ffn_gate(
    tapa::mmap<hls::vector<float, 2>> w_ffn_gate_s_sum_mmap,
    tapa::ostream<hls::vector<float, 2>>& w_ffn_gate_s_sum_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_weight_s_loader_w_ffn_gate(w_ffn_gate_s_sum_mmap, w_ffn_gate_s_sum_stream, seg_len);
    }
}

void hmt_pref_Linear_Layer_i4xi4_ffn_gate(
    tapa::istream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& input_stream_ffn_gate,
    tapa::istream<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL>>& w_ffn_gate_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream_ffn_gate,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_Linear_Layer_i4xi4_ffn_gate(input_stream_ffn_gate, w_ffn_gate_stream, output_stream_ffn_gate, seg_len);
    }
}

void hmt_pref_ffn_gate_discard(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_ffn_gate_discard(input_stream, output_stream, seg_len);
    }
}

void hmt_pref_ffn_gate_output_register(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_ffn_gate_output_register(input_stream, output_stream, seg_len);
    }
}

void hmt_pref_dequant_layer_ffn_gate_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream_ffn_gate,
    tapa::istream<hls::vector<float, 2>>& input_s_b_stream_ffn_gate, //input's scale factor and zero point
    tapa::istream<hls::vector<float, 2>>& w_ffn_gate_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_ffn_gate,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_dequant_layer_ffn_gate_int_fp32(input_stream_ffn_gate, input_s_b_stream_ffn_gate, w_ffn_gate_s_sum_stream, output_stream_ffn_gate, seg_len);
    }
}

void hmt_pref_quant_layer_ffn_up_fp32_int4(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_ffn_up,
    tapa::ostream<hls::vector<float, 2>>& input_s_b_stream_ffn_up, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& output_stream_ffn_up,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_quant_layer_ffn_up_fp32_int4(input_stream_ffn_up, input_s_b_stream_ffn_up, output_stream_ffn_up, seg_len);
    }
}

void hmt_pref_weight_loader_w_ffn_up(
    tapa::mmap<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>> w_ffn_up_mmap,
    tapa::ostream<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL>>& w_ffn_up_stream, 
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_weight_loader_w_ffn_up(w_ffn_up_mmap, w_ffn_up_stream, seg_len);
    }
}

void hmt_pref_weight_s_loader_w_ffn_up(
    tapa::mmap<hls::vector<float, 2>> w_ffn_up_s_sum_mmap,
    tapa::ostream<hls::vector<float, 2>>& w_ffn_up_s_sum_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_weight_s_loader_w_ffn_up(w_ffn_up_s_sum_mmap, w_ffn_up_s_sum_stream, seg_len);
    }
}

void hmt_pref_Linear_Layer_i4xi4_ffn_up(
    tapa::istream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& input_stream_ffn_up,
    tapa::istream<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL>>& w_ffn_up_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream_ffn_up,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_Linear_Layer_i4xi4_ffn_up(input_stream_ffn_up, w_ffn_up_stream, output_stream_ffn_up, seg_len);
    }
}

void hmt_pref_ffn_up_discard(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_ffn_up_discard(input_stream, output_stream, seg_len);
    }
}


void hmt_pref_dequant_layer_ffn_up_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream_ffn_up,
    tapa::istream<hls::vector<float, 2>>& input_s_b_stream_ffn_up, //input's scale factor and zero point
    tapa::istream<hls::vector<float, 2>>& w_ffn_up_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_ffn_up,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_dequant_layer_ffn_up_int_fp32(input_stream_ffn_up, input_s_b_stream_ffn_up, w_ffn_up_s_sum_stream, output_stream_ffn_up, seg_len);
    }
}

void hmt_pref_Swish_Layer_ffn(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_ffn_up,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_sw_ffn_up,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_Swish_Layer_ffn(input_stream_ffn_up, output_stream_sw_ffn_up, seg_len);
    }
}

void hmt_pref_Gate_Layer_ffn(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_ffn_gate,
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_sw_ffn_up,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_gated_sw_ffn_up,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_Gate_Layer_ffn(input_stream_ffn_gate, input_stream_sw_ffn_up, output_stream_gated_sw_ffn_up, seg_len);
    }
}

// Fast Hadmard Transform (R4)
void hmt_pref_FHT_R4(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_gated_sw_ffn_up,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_r4_gated_sw_ffn_up,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_FHT_R4(input_stream_gated_sw_ffn_up, output_stream_r4_gated_sw_ffn_up, seg_len);
    }
}

// FFN Down Layer
void hmt_pref_quant_layer_ffn_down_fp32_int4(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_ffn_down,
    tapa::ostream<hls::vector<float, 2>>& input_s_b_stream_ffn_down, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& output_stream_ffn_down,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_quant_layer_ffn_down_fp32_int4(input_stream_ffn_down, input_s_b_stream_ffn_down, output_stream_ffn_down, seg_len);
    }
}

void hmt_pref_weight_loader_w_ffn_down(
    tapa::mmap<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>> w_ffn_down_mmap,
    tapa::ostream<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL>>& w_ffn_down_stream, 
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_weight_loader_w_ffn_down(w_ffn_down_mmap, w_ffn_down_stream, seg_len);
    }
}

void hmt_pref_weight_s_loader_w_ffn_down(
    tapa::mmap<hls::vector<float, 2>> w_ffn_down_s_sum_mmap,
    tapa::ostream<hls::vector<float, 2>>& w_ffn_down_s_sum_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_weight_s_loader_w_ffn_down(w_ffn_down_s_sum_mmap, w_ffn_down_s_sum_stream, seg_len);
    }
}

void hmt_pref_Linear_Layer_i4xi4_ffn_down(
    tapa::istream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& input_stream_ffn_down,
    tapa::istream<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL>>& w_ffn_down_stream,
    tapa::ostream<hls::vector<ap_int<log2_INTER_DIM + 8>, TOKEN_PARALLEL>>& output_stream_ffn_down,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_Linear_Layer_i4xi4_ffn_down(input_stream_ffn_down, w_ffn_down_stream, output_stream_ffn_down, seg_len);
    }
}

void hmt_pref_ffn_down_discard(
    tapa::istream<hls::vector<ap_int<log2_INTER_DIM + 8>, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<ap_int<log2_INTER_DIM + 8>, TOKEN_PARALLEL>>& output_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_ffn_down_discard(input_stream, output_stream, seg_len);
    }
}

void hmt_pref_dequant_layer_ffn_down_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_INTER_DIM + 8>, TOKEN_PARALLEL>>& input_stream_ffn_down,
    tapa::istream<hls::vector<float, 2>>& input_s_b_stream_ffn_down, //input's scale factor and zero point
    tapa::istream<hls::vector<float, 2>>& w_ffn_down_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_ffn_down,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_dequant_layer_ffn_down_int_fp32(input_stream_ffn_down, input_s_b_stream_ffn_down, w_ffn_down_s_sum_stream, output_stream_ffn_down, seg_len);
    }
}


void hmt_pref_Residual_Layer_1(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_ffn_down,
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream_res0,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_res1,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_Residual_Layer_1(input_stream_ffn_down, input_stream_res0, output_stream_res1, seg_len);
    }
}


void hmt_pref_block_output_drainer_sync(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& output_stream_res1,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> io_mmap,
    tapa::ostream<bool>& block_output_finish_stream,
    tapa::ostream<bool>& hmt_stage_01_finish_stream,
    tapa::istream<int>& seg_len_stream
){
    seg_loop: for(int seg_len = seg_len_stream.read(); seg_len != 0; seg_len = seg_len_stream.read()){
        pref_block_output_drainer_sync(output_stream_res1, io_mmap, block_output_finish_stream, seg_len);
        // hmt_stage_01_finish_stream.write(true);
        while(! hmt_stage_01_finish_stream.try_write(true)) {};
    }
}


void HMT_SpinQuant_Prefilling(
    tapa::mmap<hls::vector<float, HMT_T_BLOCK_PARALLEL>> hmt_io_mmap,
    tapa::mmaps<hls::vector<float, HMT_W_QK_ATTN_PARALLEL>, HMT_T_BLOCK_PARALLEL> hmt_wq_wk_mmaps,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> pref_io_mmap,
    tapa::mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>> wk_wq_mmap,
    tapa::mmap<hls::vector<float, 2>> wk_wq_s_sum_mmap,
    tapa::mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>> wv_wo_mmap,
    tapa::mmap<hls::vector<float, 2>> wv_wo_s_sum_mmap,
    tapa::mmap<hls::vector<ap_int<8>, PRE_K_PARALLEL>> k_cache,
    tapa::mmap<hls::vector<ap_int<8>, PRE_V_PARALLEL>> v_cache,
    tapa::mmap<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>> w_ffn_gate_mmap,
    tapa::mmap<hls::vector<float, 2>> w_ffn_gate_s_sum_mmap,
    tapa::mmap<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>> w_ffn_up_mmap,
    tapa::mmap<hls::vector<float, 2>> w_ffn_up_s_sum_mmap,
    tapa::mmap<hls::vector<ap_int<8>, PRE_FFN_W_PARALLEL_READ/2>> w_ffn_down_mmap,
    tapa::mmap<hls::vector<float, 2>> w_ffn_down_s_sum_mmap,
    tapa::mmap<float> gamma_beta_mmap_0,
    tapa::mmap<float> gamma_beta_mmap_1,
    int seq_len,
    int seg_num
){
    tapa::stream<bool> hmt_stage_01_finish_stream("hmt_stage_01_finish_stream");
    tapa::stream<bool> hmt_stage_01_ready_stream("hmt_stage_01_ready_stream");
    tapa::stream<int> seg_len_stream("seg_len_stream");

    tapa::stream<hls::vector<float, HMT_T_BLOCK_PARALLEL>> hmt_Sn_stream("hmt_Sn_stream");
    tapa::stream<hls::vector<float, HMT_T_BLOCK_PARALLEL>> hmt_Qn_stream("hmt_Qn_stream");
    
    tapa::stream<hls::vector<float, HMT_T_BLOCK_PARALLEL>> hmt_Mn_stream("hmt_Mn_stream");
    tapa::stream<hls::vector<float, HMT_T_BLOCK_PARALLEL>> hmt_Mn_cache_stream("hmt_Mn_cache_stream");
    tapa::stream<hls::vector<float, HMT_T_BLOCK_PARALLEL>> hmt_Kn_stream("hmt_Kn_stream");

    tapa::stream<hls::vector<float, HMT_T_BLOCK_PARALLEL>> hmt_An_stream("hmt_An_stream");
    tapa::stream<hls::vector<float, HMT_T_BLOCK_PARALLEL>> hmt_sft_An_stream("hmt_sft_An_stream");


    tapa::stream<hls::vector<float, HMT_T_BLOCK_PARALLEL>> hmt_Pn_stream("hmt_Pn_stream");


    tapa::streams<hls::vector<float, HMT_W_QK_ATTN_PARALLEL>, HMT_T_BLOCK_PARALLEL> load_Kn_streams("load_Kn_streams");
    tapa::streams<hls::vector<float, HMT_W_QK_ATTN_PARALLEL>, HMT_T_BLOCK_PARALLEL> load_Mn_streams("load_Mn_streams");
    tapa::streams<hls::vector<float, HMT_W_QK_ATTN_PARALLEL>, HMT_T_BLOCK_PARALLEL> w_qk_attn_streams("w_qk_attn_streams");

    tapa::stream<hls::vector<float, HMT_T_BLOCK_PARALLEL>, HIDDEN_DIM/HMT_T_BLOCK_PARALLEL> hmt_ll_input_stream("hmt_ll_input_stream");
    tapa::stream<hls::vector<float, HMT_T_BLOCK_PARALLEL>, HIDDEN_DIM/HMT_T_BLOCK_PARALLEL> hmt_ll_output_stream("hmt_ll_output_stream");


    tapa::streams<int, SpinQuant_Pref_module_num> seg_len_streams("seg_len_streams");

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
    // tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> quant_kq_stream_redundant("quant_kq_stream_redundant");
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
    // tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> quant_vo_stream_redundant("quant_vo_stream_redundant");
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
    tapa::stream<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL>> w_ffn_gate_stream("w_ffn_gate_stream");
    tapa::stream<hls::vector<float, 2>> w_ffn_gate_s_sum_stream("w_ffn_gate_s_sum_stream");
    // tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> quant_ffn_gate_stream_redundant("quant_ffn_gate_stream_redundant");
    tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> quant_ffn_gate_stream("quant_ffn_gate_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> ffn_gate_stream("ffn_gate_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> sw_ffn_gate_stream("sw_ffn_gate_stream");

    tapa::stream<hls::vector<float, 2>, 2 * TOKEN_PARALLEL> ln_res0_s_b_stream_ffn_up("ln_res0_s_b_stream_ffn_up");
    tapa::stream<hls::vector<ap_int<4>, TOKEN_PARALLEL>> quant_ln_res0_stream_ffn_up("quant_ln_res0_stream_ffn_up");
    tapa::stream<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL>> w_ffn_up_stream("w_ffn_up_stream");
    tapa::stream<hls::vector<float, 2>> w_ffn_up_s_sum_stream("w_ffn_up_s_sum_stream");
    // tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> quant_ffn_up_stream_redundant("quant_ffn_up_stream_redundant");
    tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> quant_ffn_up_stream("quant_ffn_up_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> ffn_up_stream("ffn_up_stream");
    

    // Gate Layer for FFN UP
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>, INTER_DIM> gated_sw_ffn_up_stream("gated_sw_ffn_up_stream");

    // R4 Rotation
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> r4_gated_sw_ffn_up_stream("r4_gated_sw_ffn_up_stream");

    // FFN Down Layer
    tapa::stream<hls::vector<float, 2>, 4 * TOKEN_PARALLEL> r4_gated_sw_ffn_up_s_b_stream("gate_ffn_up_s_b_stream");
    tapa::stream<hls::vector<ap_int<4>, TOKEN_PARALLEL>, INTER_DIM> quant_r4_gated_sw_ffn_up_stream("quant_gated_sw_ffn_up_stream");
    tapa::stream<hls::vector<ap_int<4>, PRE_FFN_W_PARALLEL>> w_ffn_down_stream("w_ffn_down_stream");
    tapa::stream<hls::vector<float, 2>> w_ffn_down_s_sum_stream("w_ffn_down_s_sum_stream");
    // tapa::stream<hls::vector<ap_int<log2_INTER_DIM + 8>, TOKEN_PARALLEL>> quant_ffn_down_stream_redundant("quant_ffn_down_stream_redundant");
    tapa::stream<hls::vector<ap_int<log2_INTER_DIM + 8>, TOKEN_PARALLEL>, HIDDEN_DIM> quant_ffn_down_stream("quant_ffn_down_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> ffn_down_stream("ffn_down_stream");


    // Residual Layer 1
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>, 8 * HIDDEN_DIM> res0_stream_res1("res0_stream_res1");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> res1_stream("res1_stream");


    tapa::task()
    // hmt plug in
    .invoke(hmt_segment_loader_sync, hmt_io_mmap, pref_io_mmap, 
            hmt_stage_01_finish_stream, hmt_stage_01_ready_stream,
            hmt_Sn_stream, hmt_Pn_stream, hmt_Mn_stream, seg_len_stream, seq_len, seg_num
    )
    .invoke<tapa::detach, HMT_T_BLOCK_PARALLEL>(hmt_weight_loader_qk_attn, hmt_wq_wk_mmaps, load_Kn_streams, load_Mn_streams, w_qk_attn_streams, seg_num)
    .invoke(hmt_Linear_Layer_fp32xfp32_qk_attn_input_merger, hmt_Sn_stream, hmt_Qn_stream, hmt_sft_An_stream, hmt_Mn_stream, hmt_Mn_cache_stream, hmt_ll_input_stream, seg_num)
    .invoke(hmt_Linear_Layer_fp32xfp32_qk_attn, hmt_ll_input_stream, w_qk_attn_streams, hmt_ll_output_stream, seg_num)
    .invoke(hmt_Linear_Layer_fp32xfp32_qk_attn_output_merger, hmt_ll_output_stream, hmt_Qn_stream, hmt_An_stream, hmt_Pn_stream, hmt_Kn_stream, seg_num)
    .invoke(hmt_memory_cache_manager, hmt_Mn_cache_stream, load_Mn_streams, seg_num)
    .invoke(hmt_k_mem_cache_manager, hmt_Kn_stream, load_Kn_streams, seg_num)
    .invoke(hmt_attn_softmax, hmt_An_stream, hmt_sft_An_stream, seg_num)

    // SpinQuant Prefilling
    .invoke(hmt_pref_control, seg_len_stream, seg_len_streams)
    .invoke(hmt_pref_block_input_loader_sync, hmt_stage_01_ready_stream, block_input_ready_stream, pref_io_mmap, iembed_stream, seg_len_streams[0])


    // cache input embedding for residual layer 0
    .invoke(hmt_pref_iembed_distributor, iembed_stream, iembed_stream_ln, iembed_stream_res0, seg_len_streams[1])

    // Layer Norm 0
    .invoke(hmt_pref_Layer_Norm_0_gamma_beta_loader, gamma_beta_mmap_0, gamma_beta_stream_0, seg_len_streams[2])
    .invoke(hmt_pref_Layer_Norm_0, iembed_stream_ln, gamma_beta_stream_0, ln_iembed_stream, seg_len_streams[3])
    .invoke(hmt_pref_LN_iembed_temporal_distributor, ln_iembed_stream, ln_iembed_stream_kq, ln_iembed_stream_v, seg_len_streams[4])

    // Linear Layer for QKVO
    .invoke(hmt_pref_quant_layer_kq_fp32_int4, ln_iembed_stream_kq, ln_iembed_s_b_stream_kq, quant_ln_iembed_stream_kq, seg_len_streams[5])
    .invoke(hmt_pref_weight_loader_wk_wq, wk_wq_mmap, wk_wq_stream, seg_len_streams[6])
    .invoke(hmt_pref_weight_s_loader_wk_wq, wk_wq_s_sum_mmap, wk_wq_s_sum_stream, seg_len_streams[7])
    .invoke(hmt_pref_Linear_Layer_i4xi4_kq, quant_ln_iembed_stream_kq, wk_wq_stream, quant_kq_stream, seg_len_streams[8])
    .invoke(hmt_pref_dequant_layer_kq_int_fp32, quant_kq_stream, ln_iembed_s_b_stream_kq, wk_wq_s_sum_stream, kq_stream, seg_len_streams[9])
    .invoke(hmt_pref_RoPE_layer_kq, kq_stream, RoPE_kq_stream, seg_len_streams[10])
    .invoke(hmt_pref_qk_temporal_distributor, RoPE_kq_stream, k_stream, q_stream, seg_len_streams[11])

    .invoke(hmt_pref_vo_temporal_merger, ln_iembed_stream_v, input_stream_o, merged_stream_vo, seg_len_streams[12])
    .invoke(hmt_pref_quant_layer_vo_fp32_int4, merged_stream_vo, merged_s_b_stream_vo, quant_merged_stream_vo, seg_len_streams[13])
    .invoke(hmt_pref_weight_loader_wv_wo, wv_wo_mmap, wv_wo_stream, seg_len_streams[14])
    .invoke(hmt_pref_weight_s_loader_wv_wo, wv_wo_s_sum_mmap, wv_wo_s_sum_stream, seg_len_streams[15])
    .invoke(hmt_pref_Linear_Layer_i4xi4_vo, quant_merged_stream_vo, wv_wo_stream, quant_vo_stream, seg_len_streams[16])
    .invoke(hmt_pref_dequant_layer_vo_int_fp32, quant_vo_stream, merged_s_b_stream_vo, wv_wo_s_sum_stream, vo_stream, seg_len_streams[17])
    .invoke(hmt_pref_vo_temporal_distributor, vo_stream, v_stream, o_stream, seg_len_streams[18])


    // MHA
    .invoke(hmt_pref_quant_layer_k_fp32_int8, k_stream, quant_k_stream, seg_len_streams[19])
    .invoke(hmt_pref_K_buffer, quant_k_stream, cache_quant_k_stream, seg_len_streams[20])
    .invoke(hmt_pref_K_cache_manager, cache_quant_k_stream, k_cache, load_quant_k_stream, seg_len_streams[21])

    .invoke(hmt_pref_quant_layer_v_fp32_int8, v_stream, quant_v_stream, seg_len_streams[22])
    .invoke(hmt_pref_V_buffer_transpose, quant_v_stream, cache_quant_v_stream, seg_len_streams[23])
    .invoke(hmt_pref_V_cache_manager, cache_quant_v_stream, v_cache, load_quant_v_stream, seg_len_streams[24])

    .invoke(hmt_pref_quant_layer_q_fp32_int8, q_stream, quant_q_stream, seg_len_streams[25])
    .invoke(hmt_pref_MHA_i8xi8_qxk, quant_q_stream, load_quant_k_stream, quant_a_stream, seg_len_streams[26])
    .invoke(hmt_pref_dequant_layer_a_int_fp32, quant_a_stream, a_stream, seg_len_streams[27])

    .invoke(hmt_pref_causal_mask, a_stream, masked_a_stream, seg_len_streams[28])
    .invoke(hmt_pref_Softmax_MHA, masked_a_stream, sfm_a_stream, seg_len_streams[29])
    .invoke(hmt_pref_quant_layer_sfm_a_fp32_int8, sfm_a_stream, quant_sfm_a_stream, seg_len_streams[30])

    .invoke(hmt_pref_MHA_i8xi8_axv, quant_sfm_a_stream, load_quant_v_stream, quant_input_stream_o, seg_len_streams[31])
    .invoke(hmt_pref_dequant_layer_o_int_fp32, quant_input_stream_o, input_stream_o, seg_len_streams[32])

    // Residual Layer 0
    .invoke(hmt_pref_Residual_Layer_0, o_stream, iembed_stream_res0, res0_stream, seg_len_streams[33])
    .invoke(hmt_pref_res0_distributor, res0_stream, res0_stream_ln, res0_stream_res1, seg_len_streams[34])

    // Layer Norm 1
    .invoke(hmt_pref_Layer_Norm_1_gamma_beta_loader, gamma_beta_mmap_1, gamma_beta_stream_1, seg_len_streams[35])
    .invoke(hmt_pref_Layer_Norm_1, res0_stream_ln, gamma_beta_stream_1, ln_res0_stream, seg_len_streams[36])

    // FFN Gate and Up Layer + swish
    .invoke(hmt_pref_Gate_distributor, ln_res0_stream, ln_res0_stream_ffn_gate, ln_res0_stream_ffn_up, seg_len_streams[37])

    .invoke(hmt_pref_quant_layer_ffn_gate_fp32_int4, ln_res0_stream_ffn_gate, ln_res0_s_b_stream_ffn_gate, quant_ln_res0_stream_ffn_gate, seg_len_streams[38])
    .invoke(hmt_pref_weight_loader_w_ffn_gate, w_ffn_gate_mmap, w_ffn_gate_stream, seg_len_streams[39])
    .invoke(hmt_pref_weight_s_loader_w_ffn_gate, w_ffn_gate_s_sum_mmap, w_ffn_gate_s_sum_stream, seg_len_streams[40])
    .invoke(hmt_pref_Linear_Layer_i4xi4_ffn_gate, quant_ln_res0_stream_ffn_gate, w_ffn_gate_stream, quant_ffn_gate_stream, seg_len_streams[41])
    .invoke(hmt_pref_dequant_layer_ffn_gate_int_fp32, quant_ffn_gate_stream, ln_res0_s_b_stream_ffn_gate, w_ffn_gate_s_sum_stream, ffn_gate_stream, seg_len_streams[42])
    .invoke(hmt_pref_Swish_Layer_ffn, ffn_gate_stream, sw_ffn_gate_stream, seg_len_streams[43])

    .invoke(hmt_pref_quant_layer_ffn_up_fp32_int4, ln_res0_stream_ffn_up, ln_res0_s_b_stream_ffn_up, quant_ln_res0_stream_ffn_up, seg_len_streams[44])
    .invoke(hmt_pref_weight_loader_w_ffn_up, w_ffn_up_mmap, w_ffn_up_stream, seg_len_streams[45])
    .invoke(hmt_pref_weight_s_loader_w_ffn_up, w_ffn_up_s_sum_mmap, w_ffn_up_s_sum_stream, seg_len_streams[46])
    .invoke(hmt_pref_Linear_Layer_i4xi4_ffn_up, quant_ln_res0_stream_ffn_up, w_ffn_up_stream, quant_ffn_up_stream, seg_len_streams[47])
    .invoke(hmt_pref_dequant_layer_ffn_up_int_fp32, quant_ffn_up_stream, ln_res0_s_b_stream_ffn_up, w_ffn_up_s_sum_stream, ffn_up_stream, seg_len_streams[48])

    // Gate Layer for FFN UP
    .invoke(hmt_pref_Gate_Layer_ffn, sw_ffn_gate_stream, ffn_up_stream, gated_sw_ffn_up_stream, seg_len_streams[49])

    // R4 Rotation
    .invoke(hmt_pref_FHT_R4, gated_sw_ffn_up_stream, r4_gated_sw_ffn_up_stream, seg_len_streams[50])

    // FFN Down Layer
    .invoke(hmt_pref_quant_layer_ffn_down_fp32_int4, r4_gated_sw_ffn_up_stream, r4_gated_sw_ffn_up_s_b_stream, quant_r4_gated_sw_ffn_up_stream, seg_len_streams[51])
    .invoke(hmt_pref_weight_loader_w_ffn_down, w_ffn_down_mmap, w_ffn_down_stream, seg_len_streams[52])
    .invoke(hmt_pref_weight_s_loader_w_ffn_down, w_ffn_down_s_sum_mmap, w_ffn_down_s_sum_stream, seg_len_streams[53])
    .invoke(hmt_pref_Linear_Layer_i4xi4_ffn_down, quant_r4_gated_sw_ffn_up_stream, w_ffn_down_stream, quant_ffn_down_stream, seg_len_streams[54])
    .invoke(hmt_pref_dequant_layer_ffn_down_int_fp32, quant_ffn_down_stream, r4_gated_sw_ffn_up_s_b_stream, w_ffn_down_s_sum_stream, ffn_down_stream, seg_len_streams[55])


    // Residual Layer 1
    .invoke(hmt_pref_Residual_Layer_1, ffn_down_stream, res0_stream_res1, res1_stream, seg_len_streams[56])
    .invoke(hmt_pref_block_output_drainer_sync, res1_stream, pref_io_mmap, block_input_ready_stream, hmt_stage_01_finish_stream, seg_len_streams[57])
    ;
}



#endif

