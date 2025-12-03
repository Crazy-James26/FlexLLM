#ifndef _SQ_DEC_H_
#define _SQ_DEC_H_
#include "config_u280_mem_opt_new.h"
#include "PE.h"
#include "data_io.h"
#include "quant.h"
#include "Linear_Layer.h"
#include "Linear_Layer_flatten.h"
#include "RoPE.h"
#include "parameters/RoPE_sin_cos.h"
#include "MHA_i8xi8.h"
#include "Residual_Layer.h"
#include "Swish.h"
#include "LayerNorm.h"
#include "Softmax.h"
#include "FHT.h"
#include "Logits.h"

constexpr int KV_HIDDEN_DIM_PAD = (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) > (2 * KV_HIDDEN_DIM) ?
                                    (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) : (2 * KV_HIDDEN_DIM);

// constexpr int VOCAB_SIZE_PAD = (VOCAB_SIZE + (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) - 1)  / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) 
//                                     * (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL);

constexpr int VOCAB_SIZE_PAD = 131072;

// const int w_k_addr_bias = 0;
// const int w_v_addr_bias = w_k_addr_bias + DECODER_LAYER_NUM * KV_HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM;
// const int w_q_addr_bias = w_v_addr_bias + DECODER_LAYER_NUM * KV_HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM;
const int w_kv_addr_bias = 0;
const int w_q_addr_bias = w_kv_addr_bias + DECODER_LAYER_NUM * KV_HIDDEN_DIM_PAD / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM;
const int w_o_addr_bias = w_q_addr_bias + DECODER_LAYER_NUM * HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM;
const int w_ffn_up_addr_bias = w_o_addr_bias + DECODER_LAYER_NUM * HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM;
const int w_ffn_gate_addr_bias = w_ffn_up_addr_bias + DECODER_LAYER_NUM * INTER_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM;
const int w_ffn_down_addr_bias = w_ffn_gate_addr_bias +  DECODER_LAYER_NUM * INTER_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM;
const int w_vocab_addr_bias = w_ffn_down_addr_bias + DECODER_LAYER_NUM * HIDDEN_DIM / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * INTER_DIM;
const int w_qkvo_FFN_size = w_vocab_addr_bias + VOCAB_SIZE_PAD / (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM;

// const int w_s_k_addr_bias = 0;
// const int w_s_v_addr_bias = w_s_k_addr_bias + DECODER_LAYER_NUM * KV_HIDDEN_DIM;
// const int w_s_q_addr_bias = w_s_v_addr_bias + DECODER_LAYER_NUM * KV_HIDDEN_DIM;
const int w_s_kv_addr_bias = 0;
const int w_s_q_addr_bias = w_s_kv_addr_bias + DECODER_LAYER_NUM * KV_HIDDEN_DIM_PAD;
const int w_s_o_addr_bias = w_s_q_addr_bias + DECODER_LAYER_NUM * HIDDEN_DIM;
const int w_s_ffn_up_addr_bias = w_s_o_addr_bias + DECODER_LAYER_NUM * HIDDEN_DIM;
const int w_s_ffn_gate_addr_bias = w_s_ffn_up_addr_bias + DECODER_LAYER_NUM * INTER_DIM;
const int w_s_ffn_down_addr_bias = w_s_ffn_gate_addr_bias + DECODER_LAYER_NUM * INTER_DIM;
const int w_s_vocab_addr_bias = w_s_ffn_down_addr_bias + DECODER_LAYER_NUM * HIDDEN_DIM;
const int w_s_qkvo_FFN_size = w_s_vocab_addr_bias + VOCAB_SIZE_PAD;

void dec_block_io_sync(
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& new_embedding_stream,
    tapa::mmap<hls::vector<float, T_BLOCK_PARALLEL>> io_mmap,
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& input_stream,
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream,
    int dec_seq_len
){  
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        // Read the hidden embedding for the subsequent decoder blocks
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_input_loader<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                io_mmap, input_stream, block_id, dec_seq_id, HIDDEN_DIM
            );
            printf("Dec_seq_id %d: Block_id %d: block_input_loader completed.\n", dec_seq_id, block_id);

            dec_output_drainer<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                output_stream, io_mmap, block_id + 1, dec_seq_id, HIDDEN_DIM
            );
            printf("Dec_seq_id %d: Block_id %d: block_output_drainer completed.\n", dec_seq_id, block_id);
        }
        // store the new embedding for the first decoder block
        dec_output_drainer<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
            new_embedding_stream, io_mmap, 0, dec_seq_id + 1, HIDDEN_DIM
        );
        printf("Dec_seq_id %d: Block_id %d: new embedding is ready.\n", dec_seq_id + 1, 0);
    }
}


void dec_block_input_loader_sync(
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& new_embedding_stream,
    tapa::mmap<hls::vector<float, T_BLOCK_PARALLEL>> io_mmap,
    tapa::istream<bool>& block_input_ready_stream,
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& input_stream,
    int dec_seq_len
){  
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        // Read the hidden embedding for the subsequent decoder blocks
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_input_loader<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                io_mmap, input_stream, block_id, dec_seq_id, HIDDEN_DIM
            );
            printf("Dec_seq_id %d: Block_id %d: block_input_loader completed.\n", dec_seq_id, block_id);
            bool block_input_ready = block_input_ready_stream.read();
        }
        // store the new embedding for the first decoder block
        dec_output_drainer<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
            new_embedding_stream, io_mmap, 0, dec_seq_id + 1, HIDDEN_DIM
        );
        printf("Dec_seq_id %d: Block_id %d: new embedding is ready.\n", dec_seq_id + 1, 0);
    }
}


void dec_residual_broadcastor_merger(
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& input_stream_res0, // iembed_stream
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& input_stream_res1, // res0_stream
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream_res0_res1,
    int dec_seq_len
){  
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_stream_merger_2<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                input_stream_res0, input_stream_res1, output_stream_res0_res1, 0, HIDDEN_DIM
            );
            dec_stream_merger_2<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                input_stream_res0, input_stream_res1, output_stream_res0_res1, 1, HIDDEN_DIM
            );
        }
    }
}


void dec_residual_broadcastor(
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& residual_stream, // length is HIDDEN_DIM
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_stream_distributor_2<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                input_stream, residual_stream, output_stream, 2, HIDDEN_DIM
            );
            dec_stream_distributor_2<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                input_stream, residual_stream, output_stream, 2, HIDDEN_DIM
            );
        }
    }
}


void dec_Layer_Norm_merger(
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& LN_input_stream_01,
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& LN_input_stream_2,
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& LN_input_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_stream_merger_2<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                LN_input_stream_01, LN_input_stream_2, LN_input_stream, 0, HIDDEN_DIM
            );

            dec_stream_merger_2<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                LN_input_stream_01, LN_input_stream_2, LN_input_stream, 0, HIDDEN_DIM
            );
        }

        dec_stream_merger_2<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
            LN_input_stream_01, LN_input_stream_2, LN_input_stream, 1, HIDDEN_DIM
        );
    }
}

void dec_Layer_Norm_gamma_beta_loader_012(
    tapa::mmap<hls::vector<float, T_BLOCK_PARALLEL>> gamma_beta_mmap,
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& gamma_beta_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_Layer_Norm_gamma_beta_loader<float, T_BLOCK_PARALLEL, HIDDEN_DIM, false>(
                gamma_beta_mmap, gamma_beta_stream, block_id
            ); // for the first layer_norm after input_loader

            dec_Layer_Norm_gamma_beta_loader<float, T_BLOCK_PARALLEL, HIDDEN_DIM, false>(
                gamma_beta_mmap, gamma_beta_stream, DECODER_LAYER_NUM + block_id
            ); // for the second layer_norm after MHA
        }

        dec_Layer_Norm_gamma_beta_loader<float, T_BLOCK_PARALLEL, HIDDEN_DIM, false>(
            gamma_beta_mmap, gamma_beta_stream, 2 * DECODER_LAYER_NUM
        ); // for the third layer_norm after FFN
    }
}

void dec_Layer_Norm_012(
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& input_stream,
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& gamma_beta_stream,
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_Layer_Norm<float, T_BLOCK_PARALLEL, HIDDEN_DIM, false>(
                input_stream, gamma_beta_stream, output_stream
            ); // for the first layer_norm after input_loader
            printf("Dec_seq_id %d: Block_id %d: Layer_Norm_0 completed.\n", dec_seq_id, block_id);

            dec_Layer_Norm<float, T_BLOCK_PARALLEL, HIDDEN_DIM, false>(
                input_stream, gamma_beta_stream, output_stream
            ); // for the second layer_norm after MHA
            printf("Dec_seq_id %d: Block_id %d: Layer_Norm_1 completed.\n", dec_seq_id, block_id);
        }

        dec_Layer_Norm<float, T_BLOCK_PARALLEL, HIDDEN_DIM, false>(
            input_stream, gamma_beta_stream, output_stream
        ); // for the third layer_norm after FFN
        printf("Dec_seq_id %d: Block_id %d: Layer_Norm_2 completed.\n", dec_seq_id, DECODER_LAYER_NUM - 1);
    }
}

void dec_Layer_Norm_distributor(
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream_ln0, // to linear layer qkv
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream_ln1, // to FFN layer
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream_ln2, // to vocab layer
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_stream_distributor_3<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                input_stream, output_stream_ln0, output_stream_ln1, output_stream_ln2, 0, HIDDEN_DIM
            );

            dec_stream_distributor_3<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                input_stream, output_stream_ln0, output_stream_ln1, output_stream_ln2, 1, HIDDEN_DIM
            );
        }
        dec_stream_distributor_3<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
            input_stream, output_stream_ln0, output_stream_ln1, output_stream_ln2, 2, HIDDEN_DIM
        );
    }
}


void dec_qkvo_FFN_input_merger(
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& input_qkv_stream,
    tapa::istream<hls::vector<float, DEC_HEAD_PARALLEL>>& input_o_stream, //from dequant_layer_o_int_fp32
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& input_ffn_gate_ffn_up_stream,
    tapa::istream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& input_ffn_down_stream, //from dequant_layer_o_int_fp32
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& input_vocab_stream, 
    tapa::ostream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& input_qkvo_FFN_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            // io_buffer<float, T_BLOCK_PARALLEL, T_QKVO_FFN_BLOCK_PARALLEL, HIDDEN_DIM, 3>(
            //     input_qkv_stream, input_qkvo_FFN_stream, HIDDEN_DIM
            // );

            dec_io_buffer<float, T_BLOCK_PARALLEL, T_QKVO_FFN_BLOCK_PARALLEL, HIDDEN_DIM, 2>(
                input_qkv_stream, input_qkvo_FFN_stream, HIDDEN_DIM
            ); // kv at the smae time and then q
            
            dec_io_buffer<float, DEC_HEAD_PARALLEL, T_QKVO_FFN_BLOCK_PARALLEL, HIDDEN_DIM>(
                input_o_stream, input_qkvo_FFN_stream, HIDDEN_DIM
            );

            dec_io_buffer<float, T_BLOCK_PARALLEL, T_QKVO_FFN_BLOCK_PARALLEL, HIDDEN_DIM, 2>(
                input_ffn_gate_ffn_up_stream, input_qkvo_FFN_stream, HIDDEN_DIM
            );

            dec_io_buffer<float, T_QKVO_FFN_BLOCK_PARALLEL, T_QKVO_FFN_BLOCK_PARALLEL, INTER_DIM>(
                input_ffn_down_stream, input_qkvo_FFN_stream, INTER_DIM
            );
        }

        dec_io_buffer<float, T_BLOCK_PARALLEL, T_QKVO_FFN_BLOCK_PARALLEL, HIDDEN_DIM>(
            input_vocab_stream, input_qkvo_FFN_stream, HIDDEN_DIM
        );
    }
}


void dec_quant_layer_qkvo_FFN(
    tapa::istream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, 2>>& input_s_b_stream, //input's scale factor and zero point
    tapa::ostream<hls::vector<ap_int<4>, T_QKVO_FFN_BLOCK_PARALLEL>>& output_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            // quant_layer_fp32_qint<4, true, T_QKVO_FFN_BLOCK_PARALLEL, INTER_DIM>(
            //     input_stream, input_s_b_stream, output_stream, HIDDEN_DIM
            // ); //for k

            // quant_layer_fp32_qint<4, true, T_QKVO_FFN_BLOCK_PARALLEL, INTER_DIM>(
            //     input_stream, input_s_b_stream, output_stream, HIDDEN_DIM
            // ); // for v

            dec_quant_layer_fp32_qint<4, true, T_QKVO_FFN_BLOCK_PARALLEL, INTER_DIM>(
                input_stream, input_s_b_stream, output_stream, HIDDEN_DIM
            ); // for k and v

            dec_quant_layer_fp32_qint<4, true, T_QKVO_FFN_BLOCK_PARALLEL, INTER_DIM>(
                input_stream, input_s_b_stream, output_stream, HIDDEN_DIM
            ); // for q

            dec_quant_layer_fp32_qint<4, true, T_QKVO_FFN_BLOCK_PARALLEL, INTER_DIM>(
                input_stream, input_s_b_stream, output_stream, HIDDEN_DIM
            ); // for o

            dec_quant_layer_fp32_qint<4, true, T_QKVO_FFN_BLOCK_PARALLEL, INTER_DIM>(
                input_stream, input_s_b_stream, output_stream, HIDDEN_DIM
            ); // for ffn_gate

            dec_quant_layer_fp32_qint<4, true, T_QKVO_FFN_BLOCK_PARALLEL, INTER_DIM>(
                input_stream, input_s_b_stream, output_stream, HIDDEN_DIM
            ); // for ffn_up

            dec_quant_layer_fp32_qint<4, true, T_QKVO_FFN_BLOCK_PARALLEL, INTER_DIM>(
                input_stream, input_s_b_stream, output_stream, INTER_DIM
            ); // for ffn_down
        }
        dec_quant_layer_fp32_qint<4, true, T_QKVO_FFN_BLOCK_PARALLEL, INTER_DIM>(
            input_stream, input_s_b_stream, output_stream, HIDDEN_DIM
        ); // for vocab
    }
}

void dec_weight_loader_qkvo_FFN(
    tapa::mmap<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL>> weight_mmap,
    tapa::ostream<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>>& weight_stream_0,
    tapa::ostream<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>>& weight_stream_1, 
    int dec_seq_len,
    int addr_bias = 0
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            // dec_weight_loader_int4_blockpack_2<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, INTER_DIM>(
            //     weight_mmap, weight_stream_0, weight_stream_1, block_id, HIDDEN_DIM, KV_HIDDEN_DIM, addr_bias + w_k_addr_bias
            // );//for k

            // dec_weight_loader_int4_blockpack_2<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, INTER_DIM>(
            //     weight_mmap, weight_stream_0, weight_stream_1, block_id, HIDDEN_DIM, KV_HIDDEN_DIM, addr_bias + w_v_addr_bias
            // );// for v

            dec_weight_loader_int4_blockpack_2<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, INTER_DIM>(
                weight_mmap, weight_stream_0, weight_stream_1, block_id, HIDDEN_DIM, KV_HIDDEN_DIM_PAD, addr_bias + w_kv_addr_bias
            ); // for k and v, block 0~3: K, block 4~7: V

            dec_weight_loader_int4_blockpack_2<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, INTER_DIM>(
                weight_mmap, weight_stream_0, weight_stream_1, block_id, HIDDEN_DIM, HIDDEN_DIM, addr_bias + w_q_addr_bias
            );// for q

            dec_weight_loader_int4_blockpack_2<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, INTER_DIM>(
                weight_mmap, weight_stream_0, weight_stream_1, block_id, HIDDEN_DIM, HIDDEN_DIM, addr_bias + w_o_addr_bias
            ); //for o

            dec_weight_loader_int4_blockpack_2<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, INTER_DIM>(
                weight_mmap, weight_stream_0, weight_stream_1, block_id, HIDDEN_DIM, INTER_DIM, addr_bias + w_ffn_gate_addr_bias
            ); // for ffn_gate

            dec_weight_loader_int4_blockpack_2<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, INTER_DIM>(
                weight_mmap, weight_stream_0, weight_stream_1, block_id, HIDDEN_DIM, INTER_DIM, addr_bias + w_ffn_up_addr_bias
            ); // for ffn_up

            dec_weight_loader_int4_blockpack_2<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, INTER_DIM>(
                weight_mmap, weight_stream_0, weight_stream_1, block_id, INTER_DIM, HIDDEN_DIM, addr_bias + w_ffn_down_addr_bias
            ); // for ffn_down
        }
        dec_weight_loader_int4_blockpack_2<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, INTER_DIM>(
            weight_mmap, weight_stream_0, weight_stream_1, 0, HIDDEN_DIM, VOCAB_SIZE_PAD, addr_bias + w_vocab_addr_bias
        ); // for vocab
    }
}

void dec_weight_s_loader_qkvo_FFN(
    tapa::mmap<hls::vector<float, 2>> weight_s_sum_mmap, 
    tapa::ostream<hls::vector<float, 2>>& weight_s_sum_stream,
    int dec_seq_len,
    int addr_bias = 0
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            // dec_weight_s_loader_fp32_bandwidth<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, true, INTER_DIM>(
            //     weight_s_sum_mmap, weight_s_sum_stream, block_id, KV_HIDDEN_DIM, addr_bias + w_s_k_addr_bias
            // ); // for k

            // dec_weight_s_loader_fp32_bandwidth<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, true, INTER_DIM>(
            //     weight_s_sum_mmap, weight_s_sum_stream, block_id, KV_HIDDEN_DIM, addr_bias + w_s_v_addr_bias
            // ); // for v

            dec_weight_s_loader_fp32_bandwidth<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, true, INTER_DIM>(
                weight_s_sum_mmap, weight_s_sum_stream, block_id, KV_HIDDEN_DIM_PAD, addr_bias + w_s_kv_addr_bias 
            ); // for k and v, block 0~3: K, block 4~7: V

            dec_weight_s_loader_fp32_bandwidth<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, true, INTER_DIM>(
                weight_s_sum_mmap, weight_s_sum_stream, block_id, HIDDEN_DIM, addr_bias + w_s_q_addr_bias
            ); // for q

            dec_weight_s_loader_fp32_bandwidth<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, true, INTER_DIM>(
                weight_s_sum_mmap, weight_s_sum_stream, block_id, HIDDEN_DIM, addr_bias + w_s_o_addr_bias
            ); // for o

            dec_weight_s_loader_fp32_bandwidth<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, true, INTER_DIM>(
                weight_s_sum_mmap, weight_s_sum_stream, block_id, INTER_DIM, addr_bias + w_s_ffn_gate_addr_bias
            ); // for ffn_gate

            dec_weight_s_loader_fp32_bandwidth<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, true, INTER_DIM>(
                weight_s_sum_mmap, weight_s_sum_stream, block_id, INTER_DIM, addr_bias + w_s_ffn_up_addr_bias
            ); // for ffn_up

            dec_weight_s_loader_fp32_bandwidth<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, true, INTER_DIM>(
                weight_s_sum_mmap, weight_s_sum_stream, block_id, HIDDEN_DIM, addr_bias + w_s_ffn_down_addr_bias
            ); // for ffn_down
        }
        dec_weight_s_loader_fp32_bandwidth<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, true, INTER_DIM>(
            weight_s_sum_mmap, weight_s_sum_stream, 0, VOCAB_SIZE_PAD, addr_bias + w_s_vocab_addr_bias
        ); // for vocab
    }
}

void dec_Linear_Layer_i4xi4_qkvo_FFN(
    tapa::istream<hls::vector<ap_int<4>, T_QKVO_FFN_BLOCK_PARALLEL>>& input_stream,
    tapa::istreams<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL>& weight_streams,
    tapa::ostream<hls::vector<ap_int<log2_INTER_DIM + 8>, T_QKVO_FFN_BLOCK_PARALLEL>>& output_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            // dec_Linear_Layer_i4xi4<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
            //     input_stream, weight_streams, output_stream, HIDDEN_DIM, KV_HIDDEN_DIM
            // ); // for k
            // printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for k.\n", dec_seq_id, block_id);

            // dec_Linear_Layer_i4xi4<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
            //     input_stream, weight_streams, output_stream, HIDDEN_DIM, KV_HIDDEN_DIM
            // ); // for v
            // printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for v.\n", dec_seq_id, block_id);

            dec_Linear_Layer_i4xi4<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
                input_stream, weight_streams, output_stream, HIDDEN_DIM, KV_HIDDEN_DIM_PAD
            ); // for k and v
            printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for k and v.\n", dec_seq_id, block_id);

            dec_Linear_Layer_i4xi4<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
                input_stream, weight_streams, output_stream, HIDDEN_DIM, HIDDEN_DIM
            ); // for q
            printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for q.\n", dec_seq_id, block_id);

            dec_Linear_Layer_i4xi4<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
                input_stream, weight_streams, output_stream, HIDDEN_DIM, HIDDEN_DIM
            ); // for o
            printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for o.\n", dec_seq_id, block_id);

            dec_Linear_Layer_i4xi4<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
                input_stream, weight_streams, output_stream, HIDDEN_DIM, INTER_DIM
            ); // for ffn_gate   
            printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for ffn_gate.\n", dec_seq_id, block_id);

            dec_Linear_Layer_i4xi4<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
                input_stream, weight_streams, output_stream, HIDDEN_DIM, INTER_DIM
            ); // for ffn_up
            printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for ffn_up.\n", dec_seq_id, block_id);

            dec_Linear_Layer_i4xi4<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
                input_stream, weight_streams, output_stream, INTER_DIM, HIDDEN_DIM
            ); // for ffn_down
            printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for ffn_down.\n", dec_seq_id, block_id);
        }

        dec_Linear_Layer_i4xi4<T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
            input_stream, weight_streams, output_stream, HIDDEN_DIM, VOCAB_SIZE_PAD
        ); // for vocab
        printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for vocab.\n", dec_seq_id, DECODER_LAYER_NUM - 1);
    }
}


void dec_Linear_Layer_i4xi4_qkvo_FFN_input_broadcastor(
    tapa::istream<hls::vector<ap_int<4>, T_QKVO_FFN_BLOCK_PARALLEL>>& input_stream,
    tapa::ostreams<hls::vector<ap_int<4>, T_QKVO_FFN_BLOCK_PARALLEL>, 4>& input_streams_quart,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_stream_distributor<ap_int<4>, T_QKVO_FFN_BLOCK_PARALLEL, 4, INTER_DIM>(
                input_stream, input_streams_quart, 4, HIDDEN_DIM
            ); // for k and v
            dec_stream_distributor<ap_int<4>, T_QKVO_FFN_BLOCK_PARALLEL, 4, INTER_DIM>(
                input_stream, input_streams_quart, 4, HIDDEN_DIM
            ); // for q
            dec_stream_distributor<ap_int<4>, T_QKVO_FFN_BLOCK_PARALLEL, 4, INTER_DIM>(
                input_stream, input_streams_quart, 4, HIDDEN_DIM
            ); // for o
            dec_stream_distributor<ap_int<4>, T_QKVO_FFN_BLOCK_PARALLEL, 4, INTER_DIM>(
                input_stream, input_streams_quart, 4, HIDDEN_DIM
            ); // for ffn_gate
            dec_stream_distributor<ap_int<4>, T_QKVO_FFN_BLOCK_PARALLEL, 4, INTER_DIM>(
                input_stream, input_streams_quart, 4, HIDDEN_DIM
            ); // for ffn_up
            dec_stream_distributor<ap_int<4>, T_QKVO_FFN_BLOCK_PARALLEL, 4, INTER_DIM>(
                input_stream, input_streams_quart, 4, INTER_DIM
            ); // for ffn_down
        }
        dec_stream_distributor<ap_int<4>, T_QKVO_FFN_BLOCK_PARALLEL, 4, INTER_DIM>(
            input_stream, input_streams_quart, 4, HIDDEN_DIM
        ); // for vocab
    }
}

void dec_Linear_Layer_i4xi4_qkvo_FFN_quart(
    tapa::istream<hls::vector<ap_int<4>, T_QKVO_FFN_BLOCK_PARALLEL>>& input_stream,
    tapa::istreams<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL/4>& weight_streams,
    tapa::ostream<hls::vector<ap_int<log2_INTER_DIM + 8>, T_QKVO_FFN_BLOCK_PARALLEL/4>>& output_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_Linear_Layer_i4xi4_unroll<T_QKVO_FFN_BLOCK_PARALLEL, 4, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
                input_stream, weight_streams, output_stream, HIDDEN_DIM, KV_HIDDEN_DIM_PAD
            ); // for k and v
            printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for k and v.\n", dec_seq_id, block_id);

            dec_Linear_Layer_i4xi4_unroll<T_QKVO_FFN_BLOCK_PARALLEL, 4, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
                input_stream, weight_streams, output_stream, HIDDEN_DIM, HIDDEN_DIM
            ); // for q
            printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for q.\n", dec_seq_id, block_id);

            dec_Linear_Layer_i4xi4_unroll<T_QKVO_FFN_BLOCK_PARALLEL, 4, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
                input_stream, weight_streams, output_stream, HIDDEN_DIM, HIDDEN_DIM
            ); // for o
            printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for o.\n", dec_seq_id, block_id);

            dec_Linear_Layer_i4xi4_unroll<T_QKVO_FFN_BLOCK_PARALLEL, 4, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
                input_stream, weight_streams, output_stream, HIDDEN_DIM, INTER_DIM
            ); // for ffn_gate   
            printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for ffn_gate.\n", dec_seq_id, block_id);

            dec_Linear_Layer_i4xi4_unroll<T_QKVO_FFN_BLOCK_PARALLEL, 4, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
                input_stream, weight_streams, output_stream, HIDDEN_DIM, INTER_DIM
            ); // for ffn_up
            printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for ffn_up.\n", dec_seq_id, block_id);

            dec_Linear_Layer_i4xi4_unroll<T_QKVO_FFN_BLOCK_PARALLEL, 4, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
                input_stream, weight_streams, output_stream, INTER_DIM, HIDDEN_DIM
            ); // for ffn_down
            printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for ffn_down.\n", dec_seq_id, block_id);
        }

        dec_Linear_Layer_i4xi4_unroll<T_QKVO_FFN_BLOCK_PARALLEL, 4, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM, log2_INTER_DIM, INTER_DIM, true>(
            input_stream, weight_streams, output_stream, HIDDEN_DIM, VOCAB_SIZE_PAD
        ); // for vocab
        printf("Dec_seq_id %d: Block_id %d: Linear_Layer_i4xi4_qkvo_FFN completed for vocab.\n", dec_seq_id, DECODER_LAYER_NUM - 1);
    }
}

void dec_Linear_Layer_i4xi4_qkvo_FFN_output_merger(
    tapa::istreams<hls::vector<ap_int<log2_INTER_DIM + 8>, T_QKVO_FFN_BLOCK_PARALLEL/4>, 4>& output_streams_quart,
    tapa::ostream<hls::vector<ap_int<log2_INTER_DIM + 8>, T_QKVO_FFN_BLOCK_PARALLEL>>& output_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_stream_block_parallel_merger<ap_int<log2_INTER_DIM + 8>, T_QKVO_FFN_BLOCK_PARALLEL, 4, INTER_DIM>(
                output_streams_quart, output_stream, KV_HIDDEN_DIM_PAD
            ); // for k and v
            dec_stream_block_parallel_merger<ap_int<log2_INTER_DIM + 8>, T_QKVO_FFN_BLOCK_PARALLEL, 4, INTER_DIM>(
                output_streams_quart, output_stream, HIDDEN_DIM
            ); // for q
            dec_stream_block_parallel_merger<ap_int<log2_INTER_DIM + 8>, T_QKVO_FFN_BLOCK_PARALLEL, 4, INTER_DIM>(
                output_streams_quart, output_stream, HIDDEN_DIM
            ); // for o
            dec_stream_block_parallel_merger<ap_int<log2_INTER_DIM + 8>, T_QKVO_FFN_BLOCK_PARALLEL, 4, INTER_DIM>(
                output_streams_quart, output_stream, INTER_DIM
            ); // for ffn_gate
            dec_stream_block_parallel_merger<ap_int<log2_INTER_DIM + 8>, T_QKVO_FFN_BLOCK_PARALLEL, 4, INTER_DIM>(
                output_streams_quart, output_stream, INTER_DIM
            ); // for ffn_up
            dec_stream_block_parallel_merger<ap_int<log2_INTER_DIM + 8>, T_QKVO_FFN_BLOCK_PARALLEL, 4, INTER_DIM>(
                output_streams_quart, output_stream, HIDDEN_DIM
            ); // for ffn_down
        }
        dec_stream_block_parallel_merger<ap_int<log2_INTER_DIM + 8>, T_QKVO_FFN_BLOCK_PARALLEL, 4, INTER_DIM>(
            output_streams_quart, output_stream, VOCAB_SIZE_PAD
        ); // for vocab
    }
}


void dec_dequant_layer_qkvo_FFN(
    tapa::istream<hls::vector<ap_int<log2_INTER_DIM + 8>, T_QKVO_FFN_BLOCK_PARALLEL>>& input_stream,
    tapa::istream<hls::vector<float, 2>>& input_s_b_stream, //input's scale factor and zero point
    tapa::istream<hls::vector<float, 2>>& weight_s_sum_stream, //weight's scale factor and row sum
    tapa::ostream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& output_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            // dec_dequant_layer_qint_fp32_bandwidth<log2_INTER_DIM + 8, true, T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM>(
            //     input_stream, input_s_b_stream, weight_s_sum_stream, output_stream, KV_HIDDEN_DIM
            // ); // for k

            // dec_dequant_layer_qint_fp32_bandwidth<log2_INTER_DIM + 8, true, T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM>(
            //     input_stream, input_s_b_stream, weight_s_sum_stream, output_stream, KV_HIDDEN_DIM
            // ); // for v

            dec_dequant_layer_qint_fp32_bandwidth<log2_INTER_DIM + 8, true, T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM>(
                input_stream, input_s_b_stream, weight_s_sum_stream, output_stream, KV_HIDDEN_DIM_PAD
            ); // for k and v

            dec_dequant_layer_qint_fp32_bandwidth<log2_INTER_DIM + 8, true, T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM>(
                input_stream, input_s_b_stream, weight_s_sum_stream, output_stream, HIDDEN_DIM
            ); // for q

            dec_dequant_layer_qint_fp32_bandwidth<log2_INTER_DIM + 8, true, T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM>(
                input_stream, input_s_b_stream, weight_s_sum_stream, output_stream, HIDDEN_DIM
            ); // for o

            dec_dequant_layer_qint_fp32_bandwidth<log2_INTER_DIM + 8, true, T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM>(
                input_stream, input_s_b_stream, weight_s_sum_stream, output_stream, INTER_DIM
            ); // for ffn_gate

            dec_dequant_layer_qint_fp32_bandwidth<log2_INTER_DIM + 8, true, T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM>(
                input_stream, input_s_b_stream, weight_s_sum_stream, output_stream, INTER_DIM
            ); // for ffn_up

            dec_dequant_layer_qint_fp32_bandwidth<log2_INTER_DIM + 8, true, T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM>(
                input_stream, input_s_b_stream, weight_s_sum_stream, output_stream, HIDDEN_DIM
            ); // for ffn_down
        }

        dec_dequant_layer_qint_fp32_bandwidth<log2_INTER_DIM + 8, true, T_QKVO_FFN_BLOCK_PARALLEL, DEC_QKVO_FFN_W_PARALLEL, INTER_DIM>(
            input_stream, input_s_b_stream, weight_s_sum_stream, output_stream, VOCAB_SIZE_PAD
        ); // for vocab
    }
}


void dec_qkvo_FFN_output_distributor(
    tapa::istream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& output_qkvo_ffn_stream,
    tapa::ostream<hls::vector<float, DEC_HEAD_PARALLEL>>& output_qkv_stream,
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& output_o_stream,
    tapa::ostream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& output_ffn_up_stream,
    tapa::ostream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& output_ffn_gate_stream, // length is INTER_DIM
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& output_ffn_down_stream,
    tapa::ostream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& output_vocab_logits_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_io_buffer<float, T_QKVO_FFN_BLOCK_PARALLEL, DEC_HEAD_PARALLEL, KV_HIDDEN_DIM>(
                output_qkvo_ffn_stream, output_qkv_stream, KV_HIDDEN_DIM
            ); // for k

            dec_io_buffer<float, T_QKVO_FFN_BLOCK_PARALLEL, DEC_HEAD_PARALLEL, KV_HIDDEN_DIM>(
                output_qkvo_ffn_stream, output_qkv_stream, KV_HIDDEN_DIM
            ); // for v

            // dec_io_buffer_split<float, T_QKVO_FFN_BLOCK_PARALLEL, 2, DEC_HEAD_PARALLEL, 2 * KV_HIDDEN_DIM>(
            //     output_qkvo_ffn_stream, output_qkv_stream, 2 * KV_HIDDEN_DIM
            // ); // for k and v

            if (KV_HIDDEN_DIM_PAD > 2 * KV_HIDDEN_DIM){
                for(int k = 0; k < (KV_HIDDEN_DIM_PAD - 2 * KV_HIDDEN_DIM)/T_QKVO_FFN_BLOCK_PARALLEL; k++)
                    auto discard_pack = output_qkvo_ffn_stream.read();
            }


            dec_io_buffer<float, T_QKVO_FFN_BLOCK_PARALLEL, DEC_HEAD_PARALLEL, HIDDEN_DIM>(
                output_qkvo_ffn_stream, output_qkv_stream, HIDDEN_DIM
            ); // for q

            dec_io_buffer<float, T_QKVO_FFN_BLOCK_PARALLEL, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                output_qkvo_ffn_stream, output_o_stream, HIDDEN_DIM
            ); // for o
            
            dec_io_buffer<float, T_QKVO_FFN_BLOCK_PARALLEL, T_QKVO_FFN_BLOCK_PARALLEL, INTER_DIM>(
                output_qkvo_ffn_stream, output_ffn_gate_stream, INTER_DIM
            ); // for ffn_gate
            
            dec_io_buffer<float, T_QKVO_FFN_BLOCK_PARALLEL, T_QKVO_FFN_BLOCK_PARALLEL, INTER_DIM>(
                output_qkvo_ffn_stream, output_ffn_up_stream, INTER_DIM
            ); // for ffn_up

            dec_io_buffer<float, T_QKVO_FFN_BLOCK_PARALLEL, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                output_qkvo_ffn_stream, output_ffn_down_stream, HIDDEN_DIM
            ); // for ffn_down
        }

        dec_io_buffer<float, T_QKVO_FFN_BLOCK_PARALLEL, T_QKVO_FFN_BLOCK_PARALLEL, VOCAB_SIZE_PAD>(
            output_qkvo_ffn_stream, output_vocab_logits_stream, VOCAB_SIZE_PAD
        ); // for vocab
    }
}


//MHA
void dec_RoPE_layer_qk(
    tapa::istream<hls::vector<float, DEC_HEAD_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, DEC_HEAD_PARALLEL>>& output_stream,
    int pre_seq_len,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_RoPE_layer<float, DEC_HEAD_PARALLEL, HEAD_DIM, Q_HEAD_NUM>(
                input_stream, output_stream, PE_sin, PE_cos, KV_HEAD_NUM, pre_seq_len + dec_seq_id
            ); // for k
            printf("Dec_seq_id %d: Block_id %d: RoPE_layer completed for k.\n", dec_seq_id, block_id);

            dec_io_buffer<float, DEC_HEAD_PARALLEL, DEC_HEAD_PARALLEL, KV_HIDDEN_DIM>(
                input_stream, output_stream, KV_HIDDEN_DIM
            ); // for v

            dec_RoPE_layer<float, DEC_HEAD_PARALLEL, HEAD_DIM, Q_HEAD_NUM>(
                input_stream, output_stream, PE_sin, PE_cos, Q_HEAD_NUM, pre_seq_len + dec_seq_id
            ); // for q
            printf("Dec_seq_id %d: Block_id %d: RoPE_layer completed for q.\n", dec_seq_id, block_id);
        }
    }
}


void dec_quant_layer_qkv_fp32_int8(
    tapa::istream<hls::vector<float, DEC_HEAD_PARALLEL>>& qkv_stream,
    tapa::ostream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>>& quant_qkv_stream,
    int dec_seq_len
){
    // static float Q_s[DECODER_LAYER_NUM][Q_HEAD_NUM];
    // static float K_s[DECODER_LAYER_NUM][KV_HEAD_NUM];
    // static float V_s[DECODER_LAYER_NUM][KV_HEAD_NUM];
    // for(int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
    //     for(int h = 0; h < Q_HEAD_NUM; h++){
    //         Q_s[block_id][h] = float(h+1)/HIDDEN_DIM;
    //     }
    //     for(int h = 0; h < KV_HEAD_NUM; h++){
    //         K_s[block_id][h] = float(h+1)/HIDDEN_DIM;
    //         V_s[block_id][h] = float(h+1)/HIDDEN_DIM;
    //     }
    // }

    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_MHA_static_sym_per_tensor_quant_layer_fp32_qint<8, DEC_HEAD_PARALLEL, HEAD_DIM, KV_HEAD_NUM>(
                qkv_stream, quant_qkv_stream, K_s, block_id, HEAD_DIM
            ); // for k

            dec_MHA_static_sym_per_tensor_quant_layer_fp32_qint<8, DEC_HEAD_PARALLEL, HEAD_DIM, KV_HEAD_NUM>(
                qkv_stream, quant_qkv_stream, V_s, block_id, HEAD_DIM
            ); // for v
            
            dec_MHA_static_sym_per_tensor_quant_layer_fp32_qint<8, DEC_HEAD_PARALLEL, HEAD_DIM, Q_HEAD_NUM>(
                qkv_stream, quant_qkv_stream, Q_s, block_id, HEAD_DIM, sqrt_HEAD_DIM
            ); // for q
            printf("Dec_seq_id %d: Block_id %d: quant_layer_qkv_fp32_int8 completed.\n", dec_seq_id, block_id);
        }
    }
}

void dec_quant_qkv_distributor(
    tapa::istream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>>& quant_qkv_stream,
    tapa::ostream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>>& quant_k_stream,
    tapa::ostream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>>& quant_v_stream,
    tapa::ostream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>>& quant_q_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_io_buffer<ap_int<8>, DEC_HEAD_PARALLEL, DEC_HEAD_PARALLEL, HIDDEN_DIM>(
                quant_qkv_stream, quant_k_stream, KV_HIDDEN_DIM
            ); // for k

            dec_io_buffer<ap_int<8>, DEC_HEAD_PARALLEL, DEC_HEAD_PARALLEL, HIDDEN_DIM>(
                quant_qkv_stream, quant_v_stream, KV_HIDDEN_DIM
            ); // for v

            dec_io_buffer<ap_int<8>, DEC_HEAD_PARALLEL, DEC_HEAD_PARALLEL, HIDDEN_DIM>(
                quant_qkv_stream, quant_q_stream, HIDDEN_DIM
            ); // for q
        }
    }
}


void dec_K_cache_discard_manager(
    tapa::istream<hls::vector<ap_int<8>, DEC_K_PARALLEL>>& input_k_stream,
    tapa::mmap<hls::vector<ap_int<8>, 2 * DEC_K_PARALLEL>> k_cache,
    tapa::ostream<hls::vector<ap_int<8>, DEC_K_PARALLEL>>& output_k_stream,
    int pre_seq_len,
    int dec_seq_len,
    int addr_bias = 0
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_K_cache_manager_discard_template<ap_int<8>, DEC_HEAD_PARALLEL, 2 * DEC_K_PARALLEL,  DEC_K_PARALLEL, KV_HEAD_NUM, HEAD_DIM, ATTN_GROUP_NUM, MAX_SUM_SEQ_LEN>(
                input_k_stream, k_cache, output_k_stream, block_id, pre_seq_len + dec_seq_id, addr_bias
            );
        }
    }
}


void dec_V_cache_discard_manager(
    tapa::istream<hls::vector<ap_int<8>, DEC_V_PARALLEL>>& input_v_stream,
    tapa::mmap<hls::vector<ap_int<8>, 2 * DEC_V_PARALLEL>> v_cache,
    tapa::ostream<hls::vector<ap_int<8>, DEC_V_PARALLEL>>& output_v_stream,
    int pre_seq_len,
    int dec_seq_len,
    int addr_bias = 0
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_V_cache_manager_discard_template<ap_int<8>, DEC_HEAD_PARALLEL, 2 * DEC_V_PARALLEL, DEC_V_PARALLEL, KV_HEAD_NUM, HEAD_DIM, ATTN_GROUP_NUM, MAX_SUM_SEQ_LEN>(
                input_v_stream, v_cache, output_v_stream, block_id, pre_seq_len + dec_seq_id, addr_bias
            );
        }
    }
}



// after MHA and projection layer
void dec_residual_layer_input_merger(
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& input_stream_res0, //output_o_stream
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& input_stream_res1, //output_ffn_down_stream
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_stream_merger_2<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                input_stream_res0, input_stream_res1, output_stream, 0, HIDDEN_DIM
            );
            dec_stream_merger_2<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                input_stream_res0, input_stream_res1, output_stream, 1, HIDDEN_DIM
            );
        }
    }
}


void dec_residual_layer(
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& input_stream,
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& residual_stream,
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_Residual_Layer<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                input_stream, residual_stream, output_stream, HIDDEN_DIM
            );
            printf("Dec_seq_id %d: Block_id %d: Residual_Layer_0 completed.\n", dec_seq_id, block_id);
            
            dec_Residual_Layer<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                input_stream, residual_stream, output_stream, HIDDEN_DIM
            );
            printf("Dec_seq_id %d: Block_id %d: Residual_Layer_1 completed.\n", dec_seq_id, block_id);
        }
    }
}

void dec_residual_layer_output_distributor(
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream,
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream_res0, // to FFN layer
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream_res1, // to R1N
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_stream_distributor_2<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                output_stream, output_stream_res0, output_stream_res1, 0, HIDDEN_DIM
            );
            dec_stream_distributor_2<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                output_stream, output_stream_res0, output_stream_res1, 1, HIDDEN_DIM
            );
        }
    }
}


//FFN Swish and Gate_layer and R4 FHT
void dec_Swish_Layer_FFN(
    tapa::istream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& output_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_Swish<float, T_QKVO_FFN_BLOCK_PARALLEL, INTER_DIM>(
                input_stream, output_stream, INTER_DIM
            );
            printf("Dec_seq_id %d: Block_id %d: Swish_Layer_ffn completed.\n", dec_seq_id, block_id);
        }
    }
}

void dec_Gate_layer_FFN(
    tapa::istream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& input_seq,
    tapa::istream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& gate_seq,
    tapa::ostream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& output_seq,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_Gate_Layer_fp32xfp32<T_QKVO_FFN_BLOCK_PARALLEL, INTER_DIM>(
                input_seq, gate_seq, output_seq, INTER_DIM
            );
            printf("Dec_seq_id %d: Block_id %d: Gate_layer_FFN completed.\n", dec_seq_id, block_id);
        }
    }
}

void dec_FHT_R4(
    tapa::istream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& input_stream,
    tapa::ostream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& output_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_FHT<float, T_QKVO_FFN_BLOCK_PARALLEL, INTER_DIM, log2_INTER_DIM>(
                input_stream, output_stream, sqrt_INTER_DIM
            );
            printf("Dec_seq_id %d: Block_id %d: R4 rotation completed.\n", dec_seq_id, block_id);
        }
    }
}


void dec_block_output_broadcastor(
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream,
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream_store,
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream_downstream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM - 1; block_id++){
            dec_stream_distributor_2<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                output_stream, output_stream_store, output_stream_downstream, 0, HIDDEN_DIM
            );
        }
        // The last block's output need to be sent to downstream
        dec_stream_distributor_2<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
            output_stream, output_stream_store, output_stream_downstream, 2, HIDDEN_DIM
        );
    }
}


void dec_block_output_drainer_sync(
    tapa::istream<hls::vector<float, T_BLOCK_PARALLEL>>& output_stream,
    tapa::mmap<hls::vector<float, T_BLOCK_PARALLEL>> io_mmap, 
    tapa::ostream<bool>& block_output_ready_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        decoder_block_loop: for (int block_id = 0; block_id < DECODER_LAYER_NUM; block_id++){
            dec_output_drainer<float, T_BLOCK_PARALLEL, HIDDEN_DIM>(
                output_stream, io_mmap, block_id + 1, dec_seq_id, HIDDEN_DIM
            );
            printf("Dec_seq_id %d: Block_id %d: block_output_drainer completed.\n", dec_seq_id, block_id + 1);
            block_output_ready_stream.write(true);
        }
    }
}


void dec_Top_K_Sampling_Embedding_Layer(
    tapa::istream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>>& output_stream_vocab_logits,
    tapa::mmap<float> rand_seeds_mmap,
    tapa::mmap<hls::vector<float, T_BLOCK_PARALLEL>> vocab_lib,
    tapa::mmap<int> sampled_token_idx_mmap,
    tapa::ostream<hls::vector<float, T_BLOCK_PARALLEL>>& new_embedding_stream,
    int dec_seq_len
){
    decoder_seq_loop: for (int dec_seq_id = 0; dec_seq_id < dec_seq_len; dec_seq_id++){
        int sample_idx;
        dec_Sampling_Embedding_Layer<float, T_QKVO_FFN_BLOCK_PARALLEL, VOCAB_SIZE_PAD, LOGITS_MAX_K, T_BLOCK_PARALLEL, HIDDEN_DIM, T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL>(
            output_stream_vocab_logits, vocab_lib, new_embedding_stream, rand_seeds_mmap[dec_seq_id], sample_idx
        );
        sampled_token_idx_mmap[dec_seq_id] = sample_idx;
        printf("Dec_seq_id %d: Block_id %d: Top_K_Sampling_Embedding_Layer completed.\n", dec_seq_id, DECODER_LAYER_NUM); 
    }
}






void SpinQuant_Decoding(
    tapa::mmap<hls::vector<float, T_BLOCK_PARALLEL>> vocab_lib,
    tapa::mmap<hls::vector<float, T_BLOCK_PARALLEL>> io_mmap,
    tapa::mmaps<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL/4> w_qkvo_FFN_mmaps_quart_01_k_caches,
    tapa::mmaps<hls::vector<ap_int<8>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL/4> w_qkvo_FFN_mmaps_quart_23_v_caches,
    tapa::mmap<hls::vector<float, 2>> w_s_sum_qkvo_FFN_mmap,
    tapa::mmap<hls::vector<float, T_BLOCK_PARALLEL>> gamma_beta_mmap,
    tapa::mmap<float> rand_seeds_mmap,
    tapa::mmap<int> sampled_token_idx_mmap,
    int pre_seq_len,
    int dec_seq_len
){
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> new_embedding_stream("new_embedding_stream");
    // tapa::stream<bool> block_io_ready_stream("block_io_ready_stream");
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>, HIDDEN_DIM/T_BLOCK_PARALLEL> iembed_stream("iembed_stream");


    // Residual broadcastor 0/1 and Layer Norm 0/1/2
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> iembed_stream_res0_stream("iembed_stream_res0_stream");
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> iembed_stream_res0_stream_pass("iembed_stream_res0_stream_pass");
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>, HIDDEN_DIM/T_BLOCK_PARALLEL> iembed_stream_res0_stream_residual("iembed_stream_res0_stream_residual");

    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> LN_input_stream("LN_input_stream");
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> gamma_beta_stream("gamma_beta_stream");
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> LN_output_stream("LN_output_stream");
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> ln_iembed_stream("ln_iembed_stream");
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> ln_res0_stream("ln_res0_stream");
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> ln_res1_stream("ln_res1_stream");


    // QKVO and FFN layer
    tapa::stream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> input_qkvo_ffn_stream("input_qkvo_ffn_stream");
    tapa::stream<hls::vector<float, 2>, 4> input_s_b_qkvo_ffn_stream("input_s_b_qkvo_ffn_stream");
    tapa::stream<hls::vector<ap_int<4>, T_QKVO_FFN_BLOCK_PARALLEL>, INTER_DIM/T_QKVO_FFN_BLOCK_PARALLEL> quant_input_qkvo_ffn_stream("quant_input_qkvo_ffn_stream");
    // tapa::streams<ap_int<4>, T_QKVO_FFN_BLOCK_PARALLEL> quant_input_qkvo_ffn_loaders("quant_input_qkvo_ffn_loaders");
    tapa::streams<hls::vector<ap_int<4>, T_QKVO_FFN_BLOCK_PARALLEL>, 4> quant_input_qkvo_ffn_streams_quart("quant_input_qkvo_ffn_streams_quart");

    // tapa::streams<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL> w_qkvo_ffn_streams("w_qkvo_ffn_streams");
    tapa::streams<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL/4> w_qkvo_ffn_streams_quart_0("w_qkvo_ffn_streams_quart_0");
    tapa::streams<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL/4> w_qkvo_ffn_streams_quart_1("w_qkvo_ffn_streams_quart_1");
    tapa::streams<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL/4> w_qkvo_ffn_streams_quart_2("w_qkvo_ffn_streams_quart_2");
    tapa::streams<hls::vector<ap_int<4>, DEC_QKVO_FFN_W_PARALLEL>, T_QKVO_FFN_BLOCK_PARALLEL/4> w_qkvo_ffn_streams_quart_3("w_qkvo_ffn_streams_quart_3");
    tapa::stream<hls::vector<float, 2>> w_s_sum_qkvo_ffn_stream("w_s_sum_qkvo_ffn_stream");

    tapa::streams<hls::vector<ap_int<log2_INTER_DIM + 8>, T_QKVO_FFN_BLOCK_PARALLEL/4>, 4> quant_output_qkvo_ffn_streams_quart("quant_output_qkvo_ffn_streams_quart");
    // tapa::streams<ap_int<log2_INTER_DIM + 8>, T_QKVO_FFN_BLOCK_PARALLEL> quant_output_qkvo_ffn_drainers("quant_output_qkvo_ffn_drainers");
    tapa::stream<hls::vector<ap_int<log2_INTER_DIM + 8>, T_QKVO_FFN_BLOCK_PARALLEL>, INTER_DIM/T_QKVO_FFN_BLOCK_PARALLEL> quant_output_qkvo_ffn_stream("quant_output_qkvo_ffn_stream");
    tapa::stream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> output_qkvo_ffn_stream("output_qkvo_ffn_stream");

    tapa::stream<hls::vector<float, DEC_HEAD_PARALLEL>> output_qkv_stream("output_qkv_stream");
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> output_o_stream("output_o_stream");
    tapa::stream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> output_ffn_up_stream("output_ffn_up_stream");
    tapa::stream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> output_ffn_gate_stream("output_ffn_gate_stream");
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> output_ffn_down_stream("output_ffn_down_stream");
    tapa::stream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>, HIDDEN_DIM/T_QKVO_FFN_BLOCK_PARALLEL> output_vocab_logits_stream("output_vocab_logits_stream");

    //MHA
    tapa::stream<hls::vector<float, DEC_HEAD_PARALLEL>> RoPE_qkv_stream("RoPE_qkv_stream");
    tapa::stream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>, HIDDEN_DIM/DEC_HEAD_PARALLEL> quant_qkv_stream("quant_qkv_stream");
    tapa::stream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>> quant_q_stream("quant_q_stream");   
    tapa::stream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>> quant_k_stream("quant_k_stream");
    tapa::stream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>> quant_v_stream("quant_v_stream");

    // tapa::streams<ap_int<8>, DEC_HEAD_PARALLEL> quant_q_loaders("quant_q_loaders");
    tapa::streams<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL> cache_quant_k_streams("cache_quant_k_streams");
    tapa::streams<hls::vector<ap_int<8>, DEC_K_PARALLEL>, DEC_HEAD_PARALLEL> load_quant_k_streams("load_quant_k_streams");
    // tapa::streams<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL> quant_a_drainers("quant_a_drainers");

    tapa::stream<hls::vector<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL>, MAX_SUM_SEQ_LEN> quant_a_stream_redundant("quant_a_stream_redundant");
    tapa::stream<hls::vector<ap_int<log2_HEAD_DIM + 16>, DEC_HEAD_PARALLEL>> quant_a_stream("quant_a_stream");
    tapa::stream<hls::vector<float, DEC_HEAD_PARALLEL>> a_stream("a_stream");

    tapa::stream<hls::vector<float, DEC_HEAD_PARALLEL>> sfm_a_stream("sfm_a_stream");
    tapa::stream<hls::vector<ap_int<8>, DEC_HEAD_PARALLEL>, MAX_SUM_SEQ_LEN> quant_sfm_a_stream("quant_sfm_a_stream");

    // tapa::streams<ap_int<8>, DEC_HEAD_PARALLEL> quant_sfm_a_loaders("quant_sfm_a_loaders");
    tapa::streams<hls::vector<ap_int<8>, DEC_V_PARALLEL>, DEC_HEAD_PARALLEL> cache_quant_v_streams("cache_quant_v_streams");
    tapa::streams<hls::vector<ap_int<8>, DEC_V_PARALLEL>, DEC_HEAD_PARALLEL> load_quant_v_streams("load_quant_v_streams");
    // tapa::streams<ap_int<log2_MAX_SUM_SEQ_LEN + 16>, DEC_HEAD_PARALLEL> quant_o_drainers("quant_o_drainers");

    tapa::stream<hls::vector<ap_int<log2_MAX_SUM_SEQ_LEN + 16>, DEC_HEAD_PARALLEL>> quant_o_stream("quant_o_stream");
    tapa::stream<hls::vector<float, DEC_HEAD_PARALLEL>> input_o_stream("input_o_stream");

    // Residual layer 0/1
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> output_o_stream_output_ffn_down_stream("output_o_stream_output_ffn_down_stream");
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> res0_stream_res1_stream("res0_stream_res1_stream");
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>, HIDDEN_DIM/T_BLOCK_PARALLEL> res0_stream("res0_stream");
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>, HIDDEN_DIM/T_BLOCK_PARALLEL> res1_stream("res1_stream");

    //FFN Layer
    tapa::stream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>, INTER_DIM/T_QKVO_FFN_BLOCK_PARALLEL> sw_output_ffn_gate_stream("sw_output_ffn_gate_stream");
    tapa::stream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> gated_sw_ffn_up_stream("gated_sw_ffn_up_stream");
    tapa::stream<hls::vector<float, T_QKVO_FFN_BLOCK_PARALLEL>> input_ffn_down_stream("input_ffn_down_stream");

    //Output and downstream
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> res1_stream_store("res1_stream_store");
    tapa::stream<hls::vector<float, T_BLOCK_PARALLEL>> res1_stream_ln("res1_stream_R1N");

    
    
    tapa::task()
    //input loader
    .invoke(dec_block_io_sync, new_embedding_stream, io_mmap, iembed_stream, res1_stream_store, dec_seq_len)
    // .invoke(dec_block_input_loader_sync, new_embedding_stream, io_mmap, block_io_ready_stream, iembed_stream, dec_seq_len)
    

    // Residual broadcastor 0 and Layer Norm 0
    .invoke(dec_residual_broadcastor_merger, iembed_stream, res0_stream, iembed_stream_res0_stream, dec_seq_len)
    .invoke(dec_residual_broadcastor, iembed_stream_res0_stream, iembed_stream_res0_stream_residual, iembed_stream_res0_stream_pass, dec_seq_len)
    .invoke(dec_Layer_Norm_merger, iembed_stream_res0_stream_pass, res1_stream_ln, LN_input_stream, dec_seq_len)
    .invoke(dec_Layer_Norm_gamma_beta_loader_012, gamma_beta_mmap, gamma_beta_stream, dec_seq_len)
    .invoke(dec_Layer_Norm_012, LN_input_stream, gamma_beta_stream, LN_output_stream, dec_seq_len)
    .invoke(dec_Layer_Norm_distributor, LN_output_stream, ln_iembed_stream, ln_res0_stream, ln_res1_stream, dec_seq_len)

    // QKVO layer
    .invoke(dec_qkvo_FFN_input_merger, ln_iembed_stream, input_o_stream, ln_res0_stream, input_ffn_down_stream, ln_res1_stream, input_qkvo_ffn_stream, dec_seq_len)
    .invoke(dec_quant_layer_qkvo_FFN, input_qkvo_ffn_stream, input_s_b_qkvo_ffn_stream, quant_input_qkvo_ffn_stream, dec_seq_len)

    // .invoke<tapa::detach, T_QKVO_FFN_BLOCK_PARALLEL>(dec_weight_loader_qkvo_FFN, w_qkvo_FFN_mmaps, w_qkvo_ffn_streams, dec_seq_len)
    .invoke<tapa::detach, T_QKVO_FFN_BLOCK_PARALLEL/4>(dec_weight_loader_qkvo_FFN, w_qkvo_FFN_mmaps_quart_01_k_caches, w_qkvo_ffn_streams_quart_0,  w_qkvo_ffn_streams_quart_1, dec_seq_len, 0)
    .invoke<tapa::detach, T_QKVO_FFN_BLOCK_PARALLEL/4>(dec_weight_loader_qkvo_FFN, w_qkvo_FFN_mmaps_quart_23_v_caches, w_qkvo_ffn_streams_quart_2,  w_qkvo_ffn_streams_quart_3, dec_seq_len, 0)

    .invoke(dec_weight_s_loader_qkvo_FFN, w_s_sum_qkvo_FFN_mmap, w_s_sum_qkvo_ffn_stream, dec_seq_len, 0)

    // .invoke(dec_Linear_Layer_i4xi4_qkvo_FFN, quant_input_qkvo_ffn_stream, w_qkvo_ffn_streams, quant_output_qkvo_ffn_stream, dec_seq_len)
    .invoke(dec_Linear_Layer_i4xi4_qkvo_FFN_input_broadcastor, quant_input_qkvo_ffn_stream, quant_input_qkvo_ffn_streams_quart, dec_seq_len)
    .invoke(dec_Linear_Layer_i4xi4_qkvo_FFN_quart, quant_input_qkvo_ffn_streams_quart[0], w_qkvo_ffn_streams_quart_0, quant_output_qkvo_ffn_streams_quart[0], dec_seq_len)
    .invoke(dec_Linear_Layer_i4xi4_qkvo_FFN_quart, quant_input_qkvo_ffn_streams_quart[1], w_qkvo_ffn_streams_quart_1, quant_output_qkvo_ffn_streams_quart[1], dec_seq_len)
    .invoke(dec_Linear_Layer_i4xi4_qkvo_FFN_quart, quant_input_qkvo_ffn_streams_quart[2], w_qkvo_ffn_streams_quart_2, quant_output_qkvo_ffn_streams_quart[2], dec_seq_len)
    .invoke(dec_Linear_Layer_i4xi4_qkvo_FFN_quart, quant_input_qkvo_ffn_streams_quart[3], w_qkvo_ffn_streams_quart_3, quant_output_qkvo_ffn_streams_quart[3], dec_seq_len)
    .invoke(dec_Linear_Layer_i4xi4_qkvo_FFN_output_merger, quant_output_qkvo_ffn_streams_quart, quant_output_qkvo_ffn_stream, dec_seq_len)

    .invoke(dec_dequant_layer_qkvo_FFN, quant_output_qkvo_ffn_stream, input_s_b_qkvo_ffn_stream, w_s_sum_qkvo_ffn_stream, output_qkvo_ffn_stream, dec_seq_len)
    .invoke(dec_qkvo_FFN_output_distributor, output_qkvo_ffn_stream, output_qkv_stream, output_o_stream, output_ffn_up_stream, output_ffn_gate_stream, output_ffn_down_stream, output_vocab_logits_stream, dec_seq_len)

    //MHA
    .invoke(dec_RoPE_layer_qk, output_qkv_stream, RoPE_qkv_stream, pre_seq_len, dec_seq_len)
    .invoke(dec_quant_layer_qkv_fp32_int8, RoPE_qkv_stream, quant_qkv_stream, dec_seq_len)
    .invoke(dec_quant_qkv_distributor, quant_qkv_stream, quant_k_stream, quant_v_stream, quant_q_stream, dec_seq_len)

    .invoke(dec_K_cache_buffer, quant_k_stream, cache_quant_k_streams, pre_seq_len, dec_seq_len)
    .invoke<tapa::detach, DEC_HEAD_PARALLEL>(dec_K_cache_discard_manager, cache_quant_k_streams, w_qkvo_FFN_mmaps_quart_01_k_caches, load_quant_k_streams, pre_seq_len, dec_seq_len, w_qkvo_FFN_size)

    .invoke(dec_MHA_i8xi8_qxk, quant_q_stream, load_quant_k_streams, quant_a_stream_redundant, pre_seq_len, dec_seq_len)

    .invoke(dec_quant_a_discard, quant_a_stream_redundant, quant_a_stream, pre_seq_len, dec_seq_len)
    .invoke(dec_dequant_layer_a_int_fp32, quant_a_stream, a_stream, pre_seq_len, dec_seq_len)

    .invoke(dec_Softmax_MHA, a_stream, sfm_a_stream, pre_seq_len, dec_seq_len)
    .invoke(dec_quant_layer_sfm_a_fp32_int8, sfm_a_stream, quant_sfm_a_stream, pre_seq_len, dec_seq_len)

    .invoke(dec_V_cache_buffer, quant_v_stream, cache_quant_v_streams, dec_seq_len)
    .invoke<tapa::detach, DEC_HEAD_PARALLEL>(dec_V_cache_discard_manager, cache_quant_v_streams, w_qkvo_FFN_mmaps_quart_23_v_caches, load_quant_v_streams, pre_seq_len, dec_seq_len, w_qkvo_FFN_size)

    .invoke(dec_MHA_i8xi8_axv, quant_sfm_a_stream, load_quant_v_streams, quant_o_stream, pre_seq_len, dec_seq_len)

    .invoke(dec_dequant_layer_o_int_fp32, quant_o_stream, input_o_stream, dec_seq_len)


    // Residual layer 0
    .invoke(dec_residual_layer_input_merger, output_o_stream, output_ffn_down_stream, output_o_stream_output_ffn_down_stream, dec_seq_len)
    .invoke(dec_residual_layer, output_o_stream_output_ffn_down_stream, iembed_stream_res0_stream_residual, res0_stream_res1_stream, dec_seq_len)
    .invoke(dec_residual_layer_output_distributor, res0_stream_res1_stream, res0_stream, res1_stream, dec_seq_len)

    // FFN layer
    .invoke(dec_Swish_Layer_FFN, output_ffn_gate_stream, sw_output_ffn_gate_stream, dec_seq_len)
    .invoke(dec_Gate_layer_FFN, output_ffn_up_stream, sw_output_ffn_gate_stream, gated_sw_ffn_up_stream, dec_seq_len)
    .invoke(dec_FHT_R4, gated_sw_ffn_up_stream, input_ffn_down_stream, dec_seq_len)
    

    //output storage
    .invoke(dec_block_output_broadcastor, res1_stream, res1_stream_store, res1_stream_ln, dec_seq_len)

    //sampling and embedding layer
    .invoke(dec_Top_K_Sampling_Embedding_Layer, output_vocab_logits_stream, rand_seeds_mmap, vocab_lib, sampled_token_idx_mmap, new_embedding_stream, dec_seq_len)

    ;
    
}


#endif




