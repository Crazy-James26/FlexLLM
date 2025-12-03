#ifndef _HMT_SQ_PREF_H_
#define _HMT_SQ_PREF_H_

#include "HMT_SpinQuant_Unit.h"
#include "SpinQuant_Prefilling.h"

#define SpinQuant_Pref_module_num 58

void HMT_SpinQuant_Prefilling(
    tapa::mmap<hls::vector<float, HMT_T_BLOCK_PARALLEL>> hmt_io_mmap,
    tapa::mmaps<hls::vector<float, HMT_W_QK_ATTN_PARALLEL>, HMT_T_BLOCK_PARALLEL> hmt_wq_wk_mmaps,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> pref_io_mmap,
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

    tapa::task()
    .invoke(hmt_segment_loader_sync, hmt_io_mmap, pref_io_mmap, 
            hmt_stage_01_finish_stream, hmt_stage_01_ready_stream,
            hmt_Sn_stream, hmt_Pn_stream, hmt_Mn_stream, seg_len_stream, seq_len, seg_num
    )
    .invoke(hmt_dummy_prefilling, pref_io_mmap, 
            hmt_stage_01_ready_stream, hmt_stage_01_finish_stream, 
            seg_len_stream
    )
    .invoke<tapa::detach, HMT_T_BLOCK_PARALLEL>(hmt_weight_loader_qk_attn, hmt_wq_wk_mmaps, load_Kn_streams, load_Mn_streams, w_qk_attn_streams, seg_num)
    .invoke(hmt_Linear_Layer_fp32xfp32_qk_attn_input_merger, hmt_Sn_stream, hmt_Qn_stream, hmt_sft_An_stream, hmt_Mn_stream, hmt_Mn_cache_stream, hmt_ll_input_stream, seg_num)
    .invoke(hmt_Linear_Layer_fp32xfp32_qk_attn, hmt_ll_input_stream, w_qk_attn_streams, hmt_ll_output_stream, seg_num)
    .invoke(hmt_Linear_Layer_fp32xfp32_qk_attn_output_merger, hmt_ll_output_stream, hmt_Qn_stream, hmt_An_stream, hmt_Pn_stream, hmt_Kn_stream, seg_num)
    .invoke(hmt_memory_cache_manager, hmt_Mn_cache_stream, load_Mn_streams, seg_num)
    .invoke(hmt_k_mem_cache_manager, hmt_Kn_stream, load_Kn_streams, seg_num)
    .invoke(hmt_attn_softmax, hmt_An_stream, hmt_sft_An_stream, seg_num)
    ;
}


#endif

