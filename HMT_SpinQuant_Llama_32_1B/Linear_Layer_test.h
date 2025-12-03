#ifndef _LL_test_H_
#define _LL_test_H_
#include "config.h"
#include "data_io.h"
#include "quant.h"
#include "Linear_Layer.h"


// test seperate linear layer
void pref_input_loader_r1_ln_iembed_int4(
    tapa::mmap<hls::vector<ap_int<4>, TOKEN_PARALLEL>> input_mmap,
    tapa::ostream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& input_stream, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_input_loader<ap_int<4>, TOKEN_PARALLEL>(input_mmap, input_stream, block_id, seq_len, HIDDEN_DIM);
    }
}


void pref_weight_loader_wq(
    tapa::mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>> weight_mmap,
    tapa::ostream<hls::vector<ap_int<4>, PRE_QKVO_W_PARALLEL>>& weight_stream, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_weight_loader_int4_pack_2_discard<PRE_QKVO_W_PARALLEL_READ, PRE_QKVO_W_PARALLEL, TOKEN_PARALLEL>(
            weight_mmap, weight_stream, block_id, seq_len, HIDDEN_DIM, HIDDEN_DIM
        );
    }
}


void pref_Linear_Layer_i4xi4_q(
    tapa::istream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& input_stream,
    tapa::istream<hls::vector<ap_int<4>, PRE_QKVO_W_PARALLEL>>& weight_stream,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_Linear_Layer_i4xi4<TOKEN_PARALLEL, PRE_QKVO_W_PARALLEL, HIDDEN_DIM, log2_HIDDEN_DIM, HIDDEN_DIM, MAX_PRE_SEQ_LEN, true>(
            input_stream, weight_stream, output_stream, seq_len, HIDDEN_DIM, HIDDEN_DIM
        );
    }
}


void pref_output_drainer_q_int(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& output_stream,
    tapa::mmap<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> output_mmap, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_output_drainer<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>(output_stream, output_mmap, block_id, seq_len, HIDDEN_DIM);
    }
}


void Linear_Layer_q_Prefilling_tb(
    tapa::mmap<hls::vector<ap_int<4>, TOKEN_PARALLEL>> input_mmap,
    tapa::mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>> weight_mmap,
    tapa::mmap<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> output_mmap,
    
    int seq_len
){
    tapa::stream<hls::vector<ap_int<4>, TOKEN_PARALLEL>> input_stream("input_stream");
    tapa::stream<hls::vector<ap_int<4>, PRE_QKVO_W_PARALLEL>> weight_stream("weight_stream");
    tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> output_stream("output_stream");

    
    tapa::task()
    .invoke(pref_input_loader_r1_ln_iembed_int4, input_mmap, input_stream, seq_len)
    .invoke(pref_weight_loader_wq, weight_mmap, weight_stream, seq_len)
    .invoke(pref_Linear_Layer_i4xi4_q, input_stream, weight_stream, output_stream, seq_len)
    .invoke(pref_output_drainer_q_int, output_stream, output_mmap, seq_len);
}




// quant_layer + linear_layer + dequant_layer
void pref_input_loader_r1_ln_iembed_fp32(
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> input_mmap,
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& input_stream, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_input_loader<float, TOKEN_PARALLEL>(input_mmap, input_stream, block_id, seq_len, HIDDEN_DIM);
    }
}

void pref_quant_layer_r1_ln_iembed_fp32_int4(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, 2>>& input_s_b_stream, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<4>, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_quant_layer_fp32_qint<4, true, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream, input_s_b_stream, output_stream, seq_len, HIDDEN_DIM
        );
    }
}


void pref_weight_s_loader_wq(
    tapa::mmap<hls::vector<float, 2>> weight_s_sum_mmap,
    tapa::ostream<hls::vector<float, 2>>& weight_s_sum_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_weight_s_loader_fp32<TOKEN_PARALLEL, true>(
            weight_s_sum_mmap, weight_s_sum_stream, block_id, seq_len, HIDDEN_DIM
        );
    }
}


void pref_dequant_layer_q_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>>& input_stream,
    tapa::istream<hls::vector<float, 2>>& input_s_b_stream, //input's scale factor and zero point
    tapa::istream<hls::vector<float, 2>>& weight_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_dequant_layer_qint_fp32<log2_HIDDEN_DIM + 8, true, TOKEN_PARALLEL, HIDDEN_DIM>(
            input_stream, input_s_b_stream, weight_s_sum_stream, output_stream, seq_len, HIDDEN_DIM
        );
    }
}


void pref_output_drainer_q_fp32(
    tapa::istream<hls::vector<float, TOKEN_PARALLEL>>& output_stream,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> output_mmap, 
    int seq_len
){
    decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
        pref_output_drainer<float, TOKEN_PARALLEL>(output_stream, output_mmap, block_id, seq_len, HIDDEN_DIM);
    }
}


void QuantWrapper_Linear_Layer_q_Prefilling_tb(
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> input_mmap,
    tapa::mmap<hls::vector<ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2>> quant_weight_mmap,
    tapa::mmap<hls::vector<float, 2>> weight_s_sum_mmap,
    tapa::mmap<hls::vector<float, TOKEN_PARALLEL>> output_mmap,
    int seq_len
){
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> input_stream("input_stream");
    tapa::stream<hls::vector<ap_int<4>, TOKEN_PARALLEL>> quant_input_stream("quant_input_stream");
    tapa::stream<hls::vector<float, 2>> input_s_b_stream("input_s_b_stream");

    tapa::stream<hls::vector<ap_int<4>, PRE_QKVO_W_PARALLEL>> quant_weight_stream("quant_weight_stream");
    tapa::stream<hls::vector<float, 2>> weight_s_sum_stream("weight_s_sum_stream");

    tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, TOKEN_PARALLEL>> quant_output_stream("quant_output_stream");
    tapa::stream<hls::vector<float, TOKEN_PARALLEL>> output_stream("output_stream");

    tapa::task()
    
    .invoke(pref_input_loader_r1_ln_iembed_fp32, input_mmap, input_stream, seq_len)

    .invoke(pref_quant_layer_r1_ln_iembed_fp32_int4, input_stream, input_s_b_stream, quant_input_stream, seq_len)

    .invoke(pref_weight_loader_wq, quant_weight_mmap, quant_weight_stream, seq_len)

    .invoke(pref_weight_s_loader_wq, weight_s_sum_mmap, weight_s_sum_stream, seq_len)

    .invoke(pref_Linear_Layer_i4xi4_q, quant_input_stream, quant_weight_stream, quant_output_stream, seq_len)

    .invoke(pref_dequant_layer_q_int_fp32, quant_output_stream, input_s_b_stream, weight_s_sum_stream, output_stream, seq_len)

    .invoke(pref_output_drainer_q_fp32, output_stream, output_mmap, seq_len);
}



// test seperate linear layer
void dec_input_loader_r1_ln_iembed_int4(
    tapa::mmap<hls::vector<ap_int<4>, T_BLOCK_PARALLEL>> input_mmap,
    tapa::ostream<hls::vector<ap_int<4>, T_BLOCK_PARALLEL>>& input_stream, 
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_input_loader<ap_int<4>, T_BLOCK_PARALLEL, HIDDEN_DIM>(input_mmap, input_stream, block_id, dec_seq_id, HIDDEN_DIM);
        }
    }
}


void dec_weight_loader_wq(
    tapa::mmap<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>> weight_mmap,
    tapa::ostream<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>>& weight_stream, 
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_weight_loader_int4_pack_2<T_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, HIDDEN_DIM>(
                weight_mmap, weight_stream, block_id, HIDDEN_DIM, HIDDEN_DIM
            );
        }
    }
}


void dec_Linear_Layer_i4xi4_q(
    tapa::istream<hls::vector<ap_int<4>, T_BLOCK_PARALLEL>>& input_stream,
    tapa::istreams<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_BLOCK_PARALLEL>& weight_streams,
    tapa::ostream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, T_BLOCK_PARALLEL>>& output_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_Linear_Layer_i4xi4<T_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM, log2_HIDDEN_DIM, HIDDEN_DIM, true>(
                input_stream, weight_streams, output_stream, HIDDEN_DIM, HIDDEN_DIM
            );
        }
    }
}


void dec_output_drainer_q_int(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, T_BLOCK_PARALLEL>>& output_stream,
    tapa::mmap<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, T_BLOCK_PARALLEL>> output_mmap, 
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_output_drainer<ap_int<log2_HIDDEN_DIM + 8>, T_BLOCK_PARALLEL, HIDDEN_DIM>(output_stream, output_mmap, block_id, dec_seq_id, HIDDEN_DIM);
        }
    }
}


void Linear_Layer_q_Decoding_tb(
    tapa::mmap<hls::vector<ap_int<4>, T_BLOCK_PARALLEL>> input_mmap,
    tapa::mmaps<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, T_BLOCK_PARALLEL> weight_mmaps,
    tapa::mmap<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, T_BLOCK_PARALLEL>> output_mmap,
    int dec_seq_len
){
    tapa::stream<hls::vector<ap_int<4>, T_BLOCK_PARALLEL>> input_stream("input_stream");
    tapa::streams<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_BLOCK_PARALLEL> weight_streams("weight_streams");
    tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, T_BLOCK_PARALLEL>> output_stream("output_stream");

    tapa::task()
    .invoke(dec_input_loader_r1_ln_iembed_int4, input_mmap, input_stream, dec_seq_len)
    .invoke<tapa::detach, T_BLOCK_PARALLEL>(dec_weight_loader_wq, weight_mmaps, weight_streams, dec_seq_len)
    .invoke(dec_Linear_Layer_i4xi4_q, input_stream, weight_streams, output_stream, dec_seq_len)
    .invoke(dec_output_drainer_q_int, output_stream, output_mmap, dec_seq_len);
}




// quant_layer + linear_layer + dequant_layer
void dec_input_loader_r1_ln_iembed_fp32(
    tapa::mmap<hls::vector<float, T_BLOCK_PARALLEL>> input_mmap,
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& input_stream, 
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_input_loader<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(input_mmap, input_stream, block_id, dec_seq_id, HIDDEN_DIM);
        }
    }
}

void dec_quant_layer_r1_ln_iembed_fp32_int4(
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, 2>>& input_s_b_stream, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<4>, T_BLOCK_PARALLEL>>& output_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_quant_layer_fp32_qint<4, true, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                input_stream, input_s_b_stream, output_stream, HIDDEN_DIM
            );
        }
    }
}

void dec_weight_s_loader_wq(
    tapa::mmap<hls::vector<float, 2>> weight_s_sum_mmap,
    tapa::ostream<hls::vector<float, 2>>& weight_s_sum_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_weight_s_loader_fp32_bandwidth<T_BLOCK_PARALLEL, true, HIDDEN_DIM>(
                weight_s_sum_mmap, weight_s_sum_stream, block_id, HIDDEN_DIM
            );
        }
    }
}


void dec_dequant_layer_q_int_fp32(
    tapa::istream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, T_BLOCK_PARALLEL>>& input_stream,
    tapa::istream<hls::vector<float, 2>>& input_s_b_stream, //input's scale factor and zero point
    tapa::istream<hls::vector<float, 2>>& weight_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_dequant_layer_qint_fp32_bandwidth<log2_HIDDEN_DIM + 8, true, T_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, HIDDEN_DIM>(
                input_stream, input_s_b_stream, weight_s_sum_stream, output_stream, HIDDEN_DIM
            );
        }
    }
}


void dec_output_drainer_q_fp32(
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream,
    tapa::mmap<hls::vector<float, T_BLOCK_PARALLEL>> output_mmap, 
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_output_drainer<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(output_stream, output_mmap, block_id, dec_seq_id, HIDDEN_DIM);
        }
    }
}


void QuantWrapper_Linear_Layer_q_Decoding_tb(
    tapa::mmap<hls::vector<float, T_BLOCK_PARALLEL>> input_mmap,
    tapa::mmaps<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL/2>, T_BLOCK_PARALLEL> quant_weight_mmaps,
    tapa::mmap<hls::vector<float, 2>> weight_s_sum_mmap,
    tapa::mmap<hls::vector<float, T_BLOCK_PARALLEL>> output_mmap,
    int dec_seq_len
){
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> input_stream("input_stream");
    tapa::stream<hls::vector<ap_int<4>, T_BLOCK_PARALLEL>> quant_input_stream("quant_input_stream");
    tapa::stream<hls::vector<float, 2>> input_s_b_stream("input_s_b_stream");

    tapa::streams<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_BLOCK_PARALLEL> quant_weight_streams("quant_weight_streams");
    tapa::stream<hls::vector<float, 2>> weight_s_sum_stream("weight_s_sum_stream");

    tapa::stream<hls::vector<ap_int<log2_HIDDEN_DIM + 8>, T_BLOCK_PARALLEL>> quant_output_stream("quant_output_stream");
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> output_stream("output_stream");

    tapa::task()
    
    .invoke(dec_input_loader_r1_ln_iembed_fp32, input_mmap, input_stream, dec_seq_len)

    .invoke(dec_quant_layer_r1_ln_iembed_fp32_int4, input_stream, input_s_b_stream, quant_input_stream, dec_seq_len)

    .invoke<tapa::detach, T_BLOCK_PARALLEL>(dec_weight_loader_wq, quant_weight_mmaps, quant_weight_streams, dec_seq_len)

    .invoke(dec_weight_s_loader_wq, weight_s_sum_mmap, weight_s_sum_stream, dec_seq_len)

    .invoke(dec_Linear_Layer_i4xi4_q, quant_input_stream, quant_weight_streams, quant_output_stream, dec_seq_len)

    .invoke(dec_dequant_layer_q_int_fp32, quant_output_stream, input_s_b_stream, weight_s_sum_stream, output_stream, dec_seq_len)

    .invoke(dec_output_drainer_q_fp32, output_stream, output_mmap, dec_seq_len);
}



#endif