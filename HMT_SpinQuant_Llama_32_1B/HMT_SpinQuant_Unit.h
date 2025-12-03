#ifndef _HMT_SQ_U_H_
#define _HMT_SQ_U_H_

#include "HMT.h"


void hmt_segment_loader_sync(
    tapa::mmap<hls::vector<float, HMT_T_BLOCK_PARALLEL>> hmt_io_mmap,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> pref_io_mmap,
    tapa::istream<bool>& hmt_stage_01_finish_stream,
    tapa::ostream<bool>& hmt_stage_01_ready_stream,
    tapa::ostream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& hmt_Sn_stream, 
    tapa::istream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& hmt_Pn_stream, 
    tapa::ostream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& hmt_Mn_stream,
    tapa::ostream<int>& seg_len_stream,
    int seq_len,
    int seg_num
){
    hmt_segment_loader_sync_template<float, HMT_T_BLOCK_PARALLEL, TOKEN_PARALLEL>(
        hmt_io_mmap, pref_io_mmap,
        hmt_stage_01_finish_stream, hmt_stage_01_ready_stream,
        hmt_Sn_stream, hmt_Pn_stream, hmt_Mn_stream, 
        seg_len_stream, seq_len, seg_num
    );
}

void hmt_weight_loader_qk_attn(
    tapa::mmap<hls::vector<float, HMT_W_QK_ATTN_PARALLEL>> wq_wk_mmap,
    tapa::istream<hls::vector<float, HMT_W_QK_ATTN_PARALLEL>>& load_Kn_stream,
    tapa::istream<hls::vector<float, HMT_W_QK_ATTN_PARALLEL>>& load_Mn_stream,
    tapa::ostream<hls::vector<float, HMT_W_QK_ATTN_PARALLEL>>& w_qk_attn_stream,
    int seg_num
){
    hmt_weight_loader_qk_attn_template<float, HMT_T_BLOCK_PARALLEL, HMT_W_QK_ATTN_PARALLEL>(
        wq_wk_mmap, load_Kn_stream, load_Mn_stream, w_qk_attn_stream, seg_num
    );
}

void hmt_Linear_Layer_fp32xfp32_qk_attn_input_merger(
    tapa::istream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& hmt_Sn_stream, 
    tapa::istream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& hmt_Qn_stream, 
    tapa::istream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& hmt_sft_An_stream,
    tapa::istream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& hmt_Mn_stream,
    tapa::ostream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& hmt_Mn_cache_stream,
    tapa::ostream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& hmt_ll_input_stream,
    int seg_num
){
    hmt_Linear_Layer_fp32xfp32_qk_attn_input_merger_template<float, HMT_T_BLOCK_PARALLEL>(
        hmt_Sn_stream, hmt_Qn_stream, hmt_sft_An_stream, hmt_Mn_stream, hmt_Mn_cache_stream, hmt_ll_input_stream, seg_num
    );
}

void hmt_Linear_Layer_fp32xfp32_qk_attn(
    tapa::istream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& input_seq,
    tapa::istreams<hls::vector<float, HMT_W_QK_ATTN_PARALLEL>, HMT_T_BLOCK_PARALLEL>& weight_loaders,
    tapa::ostream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& output_seq,
    int seg_num
){
    hmt_Linear_Layer_fp32xfp32_qk_attn_template<HMT_T_BLOCK_PARALLEL, HMT_W_QK_ATTN_PARALLEL>(
        input_seq, weight_loaders, output_seq, seg_num
    );
}

void hmt_Linear_Layer_fp32xfp32_qk_attn_output_merger(
    tapa::istream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& hmt_ll_output_stream, 
    tapa::ostream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& hmt_Qn_stream,
    tapa::ostream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& hmt_An_stream,
    tapa::ostream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& hmt_Pn_stream,
    tapa::ostream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& hmt_Kn_stream,
    int seg_num
){
    hmt_Linear_Layer_fp32xfp32_qk_attn_output_merger_template<float, HMT_T_BLOCK_PARALLEL>(
        hmt_ll_output_stream, hmt_Qn_stream, hmt_An_stream, hmt_Pn_stream, hmt_Kn_stream, seg_num
    );
}

void hmt_memory_cache_manager(
    tapa::istream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& input_Mn_stream,
    tapa::ostreams<hls::vector<float, HMT_W_QK_ATTN_PARALLEL>, HMT_T_BLOCK_PARALLEL>& output_Mn_streams,
    int seg_num
){
    hmt_memory_cache_manager_template<float, HMT_T_BLOCK_PARALLEL, HMT_W_QK_ATTN_PARALLEL>(
        input_Mn_stream, output_Mn_streams, seg_num
    );
}

void hmt_k_mem_cache_manager(
    tapa::istream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& input_Kn_stream,
    tapa::ostreams<hls::vector<float, HMT_W_QK_ATTN_PARALLEL>, HMT_T_BLOCK_PARALLEL>& output_Kn_streams,
    int seg_num
){
    hmt_k_mem_cache_manager_template<float, HMT_T_BLOCK_PARALLEL, HMT_W_QK_ATTN_PARALLEL>(
        input_Kn_stream, output_Kn_streams, seg_num
    );
}

void hmt_attn_softmax(
    tapa::istream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, HMT_T_BLOCK_PARALLEL>>& output_stream,
    int seg_num
){
    hmt_attn_softmax_template<float, HMT_T_BLOCK_PARALLEL, MEM_NUM, HMT_SEG_LEN, true, true>(
        input_stream, output_stream, seg_num, MEM_NUM, sqrt_HIDDEN_DIM
    );
}

void hmt_dummy_prefilling(
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> pref_io_mmap,
    tapa::istream<bool>& hmt_stage_01_ready_stream,
    tapa::ostream<bool>& hmt_stage_01_finish_stream,
    tapa::istream<int>& seg_len_stream
){
    hmt_dummy_prefilling_template<float, TOKEN_PARALLEL>(
        pref_io_mmap,
        hmt_stage_01_ready_stream, hmt_stage_01_finish_stream, 
        seg_len_stream
    );
}


#endif