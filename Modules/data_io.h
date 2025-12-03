#ifndef _DATA_IO_H_
#define _DATA_IO_H_

#include "config.h"

template <typename T, int io_parallel, int max_hidden_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_input_loader(
    tapa::mmap<hls::vector<T, io_parallel>> input_mmap,
    tapa::ostream<hls::vector<T, io_parallel>>& input_stream, 
    int block_id,
    int seq_len = max_seq_len,
    int input_hidden_dim = max_hidden_dim,
    int addr_bias = 0
){
    int bias = addr_bias + block_id * max_seq_len/io_parallel*input_hidden_dim;
    read_input_loop: for(int i = 0; i < seq_len/io_parallel*input_hidden_dim; i++){
    #pragma HLS pipeline II=1
        hls::vector<T, io_parallel> input_pack = input_mmap[bias + i];
        input_stream.write(input_pack);
    }
}

template <typename T, int io_parallel, int max_hidden_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_output_drainer(
    tapa::istream<hls::vector<T, io_parallel>>& output_stream,
    tapa::mmap<hls::vector<T, io_parallel>> output_mmap, 
    int block_id,
    int seq_len = max_seq_len,
    int output_hidden_dim = max_hidden_dim,
    int addr_bias = 0
){
    int bias = addr_bias + block_id * max_seq_len/io_parallel*output_hidden_dim;
    write_output_loop: for(int i = 0; i < seq_len/io_parallel*output_hidden_dim; i++){
    #pragma HLS pipeline II=1
        hls::vector<T, io_parallel> output_pack = output_stream.read();
        output_mmap[bias + i] = output_pack;
    }
}



template <typename T, int io_parallel, int max_in_hidden_dim = HIDDEN_DIM, int max_out_hidden_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_io_discard(
    tapa::istream<hls::vector<T, io_parallel>>& input_stream,
    tapa::ostream<hls::vector<T, io_parallel>>& output_stream,
    int seq_len = max_seq_len,
    int in_hidden_dim = max_in_hidden_dim,
    int out_hidden_dim = max_out_hidden_dim
){
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        pass_loop: for (int k = 0; k < out_hidden_dim; k++) {
        #pragma HLS pipeline II=1
            output_stream.write(input_stream.read());
        }
        discard_loop: for (int k = 0; k < in_hidden_dim - out_hidden_dim; k++) {
        #pragma HLS pipeline II=1
            input_stream.read();
        }
    }
}

template <typename T, int io_parallel, int max_hidden_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_io_register(
    tapa::istream<hls::vector<T, io_parallel>>& input_stream,
    tapa::ostream<hls::vector<T, io_parallel>>& output_stream,
    int seq_len = max_seq_len,
    int io_hidden_dim = max_hidden_dim
){
    hls::vector<T, io_parallel> reg[max_hidden_dim];
    #pragma HLS bind_storage variable=reg type=ram_2p impl=uram
    
    o_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        store_loop: for (int k = 0; k < io_hidden_dim; k++) {
        #pragma HLS pipeline II=1
            reg[k] = input_stream.read();
        }
        load_loop: for (int k = 0; k < io_hidden_dim; k++) {
        #pragma HLS pipeline II=1
            output_stream.write(reg[k]);
        }
    }
}


template <typename T, int input_parallel, int output_parallel, int max_hidden_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_io_buffer(
    tapa::istream<hls::vector<T, input_parallel>>& input_stream,
    tapa::ostream<hls::vector<T, output_parallel>>& output_stream,
    int seq_len = max_seq_len,
    int io_hidden_dim = max_hidden_dim
){
    if(input_parallel >= output_parallel){
        hls::vector<T, output_parallel> output_pack;
        i_block_loop: for (int M = 0; M < seq_len/input_parallel; M++){
        #pragma HLS loop_tripcount min=1 max=max_seq_len/input_parallel
            hidden_dim_loop: for (int k = 0; k < io_hidden_dim; k++) { 
                hls::vector<T, input_parallel> input_pack = input_stream.read();
                i_buffer_loop: for(int m = 0; m < input_parallel/output_parallel; m++){
                #pragma HLS pipeline II=1
                    for(int i = 0; i < output_parallel; i++){
                    #pragma HLS unroll
                        output_pack[i] = input_pack[m * output_parallel + i];
                    }
                    output_stream.write(output_pack);
                }
            }
        }
    }
    else{
        hls::vector<T, output_parallel> output_pack[max_hidden_dim];
        o_block_loop: for (int M = 0; M < seq_len/output_parallel; M++){
        #pragma HLS loop_tripcount min=1 max=max_seq_len/output_parallel
            o_buffer_loop: for(int m = 0; m < output_parallel/input_parallel; m++){
                hidden_dim_loop_1: for (int k = 0; k < io_hidden_dim; k++) {
                #pragma HLS pipeline II=1
                    hls::vector<T, input_parallel> input_pack = input_stream.read();
                    for(int i = 0; i < input_parallel; i++){
                    #pragma HLS unroll
                        output_pack[k][m * input_parallel + i] = input_pack[i];
                    }
                    if(m == output_parallel/input_parallel - 1) output_stream.write(output_pack[k]);
                }
            }
        }
    }
    
}


template <typename T, int input_parallel, int output_parallel, int max_hidden_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_io_buffer_transpose(
    tapa::istream<hls::vector<T, input_parallel>>& input_stream,
    tapa::ostream<hls::vector<T, output_parallel>>& output_stream,
    int seq_len = max_seq_len,
    int io_hidden_dim = max_hidden_dim
){
    T buffer_data[output_parallel][input_parallel];
    #pragma HLS ARRAY_PARTITION variable=buffer_data cyclic factor=output_parallel/2 dim=1
    #pragma HLS ARRAY_PARTITION variable=buffer_data cyclic factor=input_parallel/2 dim=2

    i_block_loop: for (int M = 0; M < seq_len/input_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/input_parallel
        o_block_loop: for (int K = 0; K < io_hidden_dim/output_parallel; K++) {
            input_buffer_loop: for (int k = 0; k < output_parallel; k++) {
            #pragma HLS pipeline II=1
                hls::vector<T, input_parallel> input_pack = input_stream.read();
                for (int m = 0; m < input_parallel; m++){
                #pragma HLS unroll
                    buffer_data[k][m] = input_pack[m];
                }
            }
            
            output_buffer_loop: for (int m = 0; m < input_parallel; m++) {
            #pragma HLS pipeline II=1
                hls::vector<T, output_parallel> output_pack;   
                for (int k = 0; k < output_parallel; k++) {
                #pragma HLS unroll
                    output_pack[k] = buffer_data[k][m];
                }
                output_stream.write(output_pack);
            }
        }
    }
}


template <typename T, int input_parallel, int output_parallel, int max_hidden_dim = HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_io_buffer_transpose_mmap(
    tapa::istream<hls::vector<T, input_parallel>>& input_stream,
    tapa::mmap<hls::vector<T, output_parallel>> trans_mmap, //trans_mmap[io_hidden_dim/output_parallel * max_seq_len]
    tapa::ostream<hls::vector<T, output_parallel>>& output_stream,
    int seq_len = max_seq_len,
    int io_hidden_dim = max_hidden_dim
){
    T buffer_data[output_parallel][input_parallel];
    #pragma HLS ARRAY_PARTITION variable=buffer_data cyclic factor=output_parallel/2 dim=1
    #pragma HLS ARRAY_PARTITION variable=buffer_data cyclic factor=input_parallel/2 dim=2

    i_block_loop: for (int M = 0; M < seq_len/input_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/input_parallel
        hidden_dim_block_loop: for (int K = 0; K < io_hidden_dim/output_parallel; K++) {
            input_buffer_loop: for (int k = 0; k < output_parallel; k++) {
            #pragma HLS pipeline II=1
                hls::vector<T, input_parallel> input_pack = input_stream.read();
                for (int m = 0; m < input_parallel; m++){
                #pragma HLS unroll
                    buffer_data[k][m] = input_pack[m];
                }
            }
            int bias = K * max_seq_len + M * input_parallel;
            trans_loop: for (int m = 0; m < input_parallel; m++) {
            #pragma HLS pipeline II=1
                    hls::vector<T, output_parallel> trans_pack;
                    for (int k = 0; k < output_parallel; k++) {
                    #pragma HLS unroll
                        trans_pack[k] = buffer_data[k][m];
                    }
                trans_mmap[bias + m] = trans_pack;
            }
            
            // trans_loop: for (int m = 0, m_resp = 0; m_resp < input_parallel; ) {
            // #pragma HLS pipeline II=1
            //     if ((m < input_parallel) & !trans_mmap.write_addr.full() & !trans_mmap.write_data.full() ) {
            //         hls::vector<T, output_parallel> trans_pack;
            //         for (int k = 0; k < output_parallel; k++) {
            //         #pragma HLS unroll
            //             trans_pack[k] = buffer_data[k][m];
            //         }
            //         trans_mmap.write_addr.try_write(K * max_seq_len + M * input_parallel + m);
            //         trans_mmap.write_data.try_write(trans_pack);
            //         ++m;
            //     }
            //     uint8_t n_resp;
            //     if (trans_mmap.write_resp.try_read(n_resp)) {
            //         m_resp += int(n_resp) + 1;
            //     }
            //     // trans_mmap[K * max_seq_len + M * input_parallel + m] = trans_pack;
            // }
        }
    }

    o_block_loop: for (int K = 0; K < io_hidden_dim/output_parallel; K++) {
        for(int m = 0; m < seq_len; m++) {
        #pragma HLS loop_tripcount min=1 max=max_seq_len
            hls::vector<T, output_parallel> output_pack = trans_mmap[K * max_seq_len + m];
            output_stream.write(output_pack);
        }
        // read_trans_loop: for(int i_req = 0, i_resp = 0; i_resp < seq_len;){
        // #pragma HLS pipeline II=1
        //     read_async_mmap(
        //         trans_mmap,
        //         output_stream,
        //         K * max_seq_len,
        //         seq_len,
        //         i_req,
        //         i_resp
        //     );
        // }
    }
}


template <typename T, int io_parallel, int max_hidden_dim=HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_stream_distributor(
    tapa::istream<hls::vector<float, io_parallel>>& input_stream,
    tapa::ostream<hls::vector<float, io_parallel>>& output_stream_0,
    tapa::ostream<hls::vector<float, io_parallel>>& output_stream_1,
    int out_channel,
    int io_hidden_dim = max_hidden_dim,
    int seq_len = max_seq_len
){
    io_loop: for (int M = 0; M < seq_len/io_parallel*io_hidden_dim; M++){
    #pragma HLS loop_tripcount min=1*max_hidden_dim max=max_seq_len/io_parallel*max_hidden_dim
        hls::vector<float, io_parallel> input_pack = input_stream.read();
        if(out_channel == 0 || out_channel == 2)
            output_stream_0.write(input_pack);
        if(out_channel == 1 || out_channel == 2)
            output_stream_1.write(input_pack);
    }
}

template <typename T, int io_parallel, int max_hidden_dim=HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_stream_distributor_3(
    tapa::istream<hls::vector<float, io_parallel>>& input_stream,
    tapa::ostream<hls::vector<float, io_parallel>>& output_stream_0,
    tapa::ostream<hls::vector<float, io_parallel>>& output_stream_1,
    tapa::ostream<hls::vector<float, io_parallel>>& output_stream_2,
    int out_channel,
    int io_hidden_dim = max_hidden_dim,
    int seq_len = max_seq_len
){
    io_loop: for (int M = 0; M < seq_len/io_parallel*io_hidden_dim; M++){
    #pragma HLS loop_tripcount min=1*max_hidden_dim max=max_seq_len/io_parallel*max_hidden_dim
        hls::vector<float, io_parallel> input_pack = input_stream.read();
        if(out_channel == 0 || out_channel == 3)
            output_stream_0.write(input_pack);
        if(out_channel == 1 || out_channel == 3)
            output_stream_1.write(input_pack);
        if(out_channel == 2 || out_channel == 3)
            output_stream_2.write(input_pack);
    }
}


template <typename T, int io_parallel, int max_hidden_dim=HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN, int merge_block_dim = 1>
void pref_stream_merger(
    tapa::istream<hls::vector<float, io_parallel>>& input_stream_0,
    tapa::istream<hls::vector<float, io_parallel>>& input_stream_1,
    tapa::ostream<hls::vector<float, io_parallel>>& output_stream,
    int in_channel,
    int io_hidden_dim = max_hidden_dim,
    int seq_len = max_seq_len
){
    if(in_channel == 0 || in_channel == 1){
        seq_loop_01: for (int M = 0; M < seq_len/io_parallel*io_hidden_dim; M++){
        #pragma HLS loop_tripcount min=1*max_hidden_dim max=max_seq_len/io_parallel*max_hidden_dim
            hls::vector<float, io_parallel> input_pack;
            if(in_channel == 0)
                input_pack = input_stream_0.read();
            else if(in_channel == 1)
                input_pack = input_stream_1.read();
            output_stream.write(input_pack);
        }
    }
    else if(in_channel == 2){
        seq_loop_2: for (int M = 0; M < seq_len/io_parallel; M++){
        #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
            hidden_loop_2: for (int m = 0; m < io_hidden_dim/(2*merge_block_dim); m++){
                #pragma HLS loop_tripcount min=max_hidden_dim/(2*merge_block_dim) max=max_hidden_dim/(2*merge_block_dim)
                block_loop_0: for (int k = 0; k < merge_block_dim; k++){
                    output_stream.write(input_stream_0.read());
                }
                block_loop_1: for (int k = 0; k < merge_block_dim; k++){
                    output_stream.write(input_stream_1.read());
                }
            }
        }
    }
}


template <typename T, int io_parallel, int max_hidden_dim=HIDDEN_DIM, int max_seq_len = MAX_PRE_SEQ_LEN>
void pref_stream_merger_3(
    tapa::istream<hls::vector<float, io_parallel>>& input_stream_0,
    tapa::istream<hls::vector<float, io_parallel>>& input_stream_1,
    tapa::istream<hls::vector<float, io_parallel>>& input_stream_2,
    tapa::ostream<hls::vector<float, io_parallel>>& output_stream,
    int in_channel,
    int io_hidden_dim = max_hidden_dim,
    int seq_len = max_seq_len
){
    io_0_loop: for (int M = 0; M < seq_len/io_parallel*io_hidden_dim; M++){
    #pragma HLS loop_tripcount min=1*max_hidden_dim max=max_seq_len/io_parallel*max_hidden_dim
        hls::vector<float, io_parallel> input_pack;
        if(in_channel == 0)
            input_pack = input_stream_0.read();
        else if(in_channel == 1)
            input_pack = input_stream_1.read();
        else if(in_channel == 2)
            input_pack = input_stream_2.read();
        output_stream.write(input_pack);
    }
}


template <typename T, int block_parallel, int max_hidden_dim = HIDDEN_DIM, int max_dec_len = MAX_DEC_SEQ_LEN>
void dec_input_loader(
    tapa::mmap<hls::vector<T, block_parallel>> input_mmap,
    tapa::ostream<hls::vector<T, block_parallel>>& input_stream, 
    int block_id,
    int dec_seq_id,
    int input_hidden_dim = max_hidden_dim,
    int addr_bias = 0
){
    int bias = addr_bias + (block_id *  max_dec_len + dec_seq_id) * input_hidden_dim/block_parallel;
    read_input_loop: for(int i = 0; i < input_hidden_dim/block_parallel; i++) {
    #pragma HLS pipeline II=1
        hls::vector<T, block_parallel> in_pack = input_mmap[bias + i];
        input_stream.write(in_pack);
    }
}


template <typename T, int block_parallel, int max_hidden_dim = HIDDEN_DIM, int max_dec_len = MAX_DEC_SEQ_LEN>
void dec_output_drainer(
    tapa::istream<hls::vector<T, block_parallel>>& output_stream,
    tapa::mmap<hls::vector<T, block_parallel>> output_mmap, 
    int block_id,
    int dec_seq_id,
    int output_hidden_dim = max_hidden_dim,
    int addr_bias = 0
){
    int bias = addr_bias + (block_id *  max_dec_len + dec_seq_id) * output_hidden_dim/block_parallel;
    write_output_loop: for(int i = 0; i < output_hidden_dim/block_parallel; i++) {
    #pragma HLS pipeline II=1
        hls::vector<T, block_parallel> out_pack = output_stream.read();
        output_mmap[bias + i] = out_pack;
        // if(block_id == DECODER_LAYER_NUM){
            // if (i < 4) {
            //     for(int k = 0; k < T_BLOCK_PARALLEL; k++)
            //         cout << out_pack[k] << " ";
            // }
        // }
    }
}



template <typename T, int block_parallel_in, int block_parallel_out, int max_hidden_dim=HIDDEN_DIM, int repeated_times=1>
void dec_io_buffer(
    tapa::istream<hls::vector<T, block_parallel_in>>& input_stream,
    tapa::ostream<hls::vector<T, block_parallel_out>>& output_stream,
    int io_hidden_dim = max_hidden_dim
){  
    if(repeated_times == 1 & block_parallel_in == block_parallel_out) {
        for (int i = 0; i < io_hidden_dim/block_parallel_in; i++) {
        #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel_in
        #pragma HLS pipeline II=1
            hls::vector<T, block_parallel_in> in_pack = input_stream.read();
            hls::vector<T, block_parallel_out> out_pack;
            for(int j = 0; j < block_parallel_out; j++) {
                out_pack[j] = in_pack[j];
            }
            output_stream.write(out_pack);
        }
    }
    else{
        T data_reg[max_hidden_dim];
        #pragma HLS bind_storage variable=data_reg type=ram_2p impl=uram
        #pragma HLS ARRAY_PARTITION variable=data_reg type=block factor=block_parallel_in > block_parallel_out ? block_parallel_in : block_parallel_out

        in_buf_loop: for (int k = 0; k < io_hidden_dim/block_parallel_in; k++) {    // L19
        #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel_in
        #pragma HLS pipeline II=1
            hls::vector<T, block_parallel_in> data_pack = input_stream.read();
            for (int i = 0; i < block_parallel_in; i++) {
                data_reg[i * io_hidden_dim/block_parallel_in + k] = data_pack[i]; //block partition
            }
        }

        // if(block_parallel_in == T_QKVO_FFN_BLOCK_PARALLEL && block_parallel_out == DEC_HEAD_PARALLEL){
        //     cout << "data_reg: ";
        //     for(int i = 0; i < io_hidden_dim; i++) {
        //         cout << data_reg[i] << " "; 
        //     }
        //     cout << endl;
        // }

        repeated_loop: for(int i = 0; i < repeated_times; i++) {
            out_buf_loop: for (int k = 0; k < io_hidden_dim/block_parallel_out; k++) {    // L19
            #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel_out
            #pragma HLS pipeline II=1
                hls::vector<T, block_parallel_out> data_pack;
                for (int j = 0; j < block_parallel_out; j++) {
                    data_pack[j] = data_reg[j * io_hidden_dim/block_parallel_out + k]; //block partition
                }
                output_stream.write(data_pack);
            }
        }
    }
}


template <typename T, int block_parallel_in, int block_parallel_in_split_num, int block_parallel_out, int max_hidden_dim=HIDDEN_DIM>
void dec_io_buffer_split(
    tapa::istream<hls::vector<T, block_parallel_in>>& input_stream,
    tapa::ostream<hls::vector<T, block_parallel_out>>& output_stream,
    int io_hidden_dim = max_hidden_dim
){  
    T data_reg[block_parallel_in_split_num][max_hidden_dim/block_parallel_in_split_num];
    #pragma HLS bind_storage variable=data_reg type=ram_2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=data_reg type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=data_reg type=block factor=block_parallel_in/block_parallel_in_split_num > block_parallel_out ? block_parallel_in/block_parallel_in_split_num : block_parallel_out dim=2

    in_buf_loop: for (int k = 0; k < io_hidden_dim/block_parallel_in; k++) {    // L19
    #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel_in
    #pragma HLS pipeline II=1
        hls::vector<T, block_parallel_in> data_pack = input_stream.read();
        for (int I = 0; I < block_parallel_in_split_num; I++) {
            for (int i = 0; i < block_parallel_in/block_parallel_in_split_num; i++) {
                data_reg[I][i * io_hidden_dim/block_parallel_in + k] = data_pack[I * (block_parallel_in/block_parallel_in_split_num) + i]; //block partition
            }
        }
    }

    split_out_loop: for (int I = 0; I < block_parallel_in_split_num; I++) {
        out_buf_loop: for (int k = 0; k < io_hidden_dim/block_parallel_in_split_num/block_parallel_out; k++) {    // L19
        #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel_in_split_num/block_parallel_out
        #pragma HLS pipeline II=1
            hls::vector<T, block_parallel_out> data_pack;
            for (int j = 0; j < block_parallel_out; j++) {
                data_pack[j] = data_reg[I][j * io_hidden_dim/block_parallel_in_split_num/block_parallel_out + k]; //block partition
            }
            output_stream.write(data_pack);
        }
    }
}


template <typename T, int max_in_hidden_dim = HIDDEN_DIM, int max_out_hidden_dim = HIDDEN_DIM>
void dec_io_discard(
    tapa::istream<T>& input_stream,
    tapa::ostream<T>& output_stream,
    int in_hidden_dim = max_in_hidden_dim,
    int out_hidden_dim = max_out_hidden_dim
){
    pass_loop: for (int k = 0; k < out_hidden_dim; k++) {
    #pragma HLS pipeline II=1
        output_stream.write(input_stream.read());
    }
    discard_loop: for (int k = 0; k < in_hidden_dim - out_hidden_dim; k++) {
    #pragma HLS pipeline II=1
        input_stream.read();
    }
}


template <typename T, int block_parallel, int distribute_num, int max_hidden_dim=HIDDEN_DIM>
void dec_stream_distributor(
    tapa::istream<hls::vector<T, block_parallel>>& input_stream,
    tapa::ostreams<hls::vector<T, block_parallel>, distribute_num>& output_streams,
    int out_channel,
    int io_hidden_dim = max_hidden_dim
){
    io_loop: for (int M = 0; M < io_hidden_dim/block_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel
    #pragma HLS pipeline II=1
        hls::vector<T, block_parallel> input_pack = input_stream.read();
        for(int i = 0; i < distribute_num; i++) {
        #pragma HLS UNROLL
            if(out_channel == i || out_channel == distribute_num) 
                output_streams[i].write(input_pack);
        }
    }
}


template <typename T, int block_parallel, int max_hidden_dim=HIDDEN_DIM>
void dec_stream_distributor_2(
    tapa::istream<hls::vector<T, block_parallel>>& input_stream,
    tapa::ostream<hls::vector<T, block_parallel>>& output_stream_0,
    tapa::ostream<hls::vector<T, block_parallel>>& output_stream_1,
    int out_channel,
    int io_hidden_dim = max_hidden_dim
){
    io_loop: for (int M = 0; M < io_hidden_dim/block_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel
    #pragma HLS pipeline II=1
        hls::vector<T, block_parallel> input_pack = input_stream.read();
        if(out_channel == 0 || out_channel == 2)
            output_stream_0.write(input_pack);
        if(out_channel == 1 || out_channel == 2)
            output_stream_1.write(input_pack);
    }
}

template <typename T, int block_parallel, int max_hidden_dim=HIDDEN_DIM>
void dec_stream_distributor_3(
    tapa::istream<hls::vector<T, block_parallel>>& input_stream,
    tapa::ostream<hls::vector<T, block_parallel>>& output_stream_0,
    tapa::ostream<hls::vector<T, block_parallel>>& output_stream_1,
    tapa::ostream<hls::vector<T, block_parallel>>& output_stream_2,
    int out_channel,
    int io_hidden_dim = max_hidden_dim
){
    io_loop: for (int M = 0; M < io_hidden_dim/block_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel
    #pragma HLS pipeline II=1
        hls::vector<T, block_parallel> input_pack = input_stream.read();
        if(out_channel == 0 || out_channel == 3)
            output_stream_0.write(input_pack);
        if(out_channel == 1 || out_channel == 3)
            output_stream_1.write(input_pack);
        if(out_channel == 2 || out_channel == 3)
            output_stream_2.write(input_pack);
    }
}


template <typename T, int block_parallel, int max_hidden_dim=HIDDEN_DIM>
void dec_stream_merger_2(
    tapa::istream<hls::vector<T, block_parallel>>& input_stream_0,
    tapa::istream<hls::vector<T, block_parallel>>& input_stream_1,
    tapa::ostream<hls::vector<T, block_parallel>>& output_stream,
    int in_channel,
    int io_hidden_dim = max_hidden_dim
){
    io_0_loop: for (int M = 0; M < io_hidden_dim/block_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel
    #pragma HLS pipeline II=1
        hls::vector<T, block_parallel> input_pack;
        if(in_channel == 0)
            input_pack = input_stream_0.read();
        else if(in_channel == 1)
            input_pack = input_stream_1.read();
        output_stream.write(input_pack);
    }
}


template <typename T, int block_parallel, int merge_num, int max_hidden_dim=HIDDEN_DIM>
void dec_stream_block_parallel_merger(
    tapa::istreams<hls::vector<T, block_parallel/merge_num>, merge_num>& input_streams,
    tapa::ostream<hls::vector<T, block_parallel>>& output_stream,
    int io_hidden_dim = max_hidden_dim
){
    io_0_loop: for (int M = 0; M < io_hidden_dim/block_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_hidden_dim/block_parallel
    #pragma HLS pipeline II=1
        hls::vector<T, block_parallel> output_pack;
        for(int i = 0; i < merge_num; i++) {
        #pragma HLS UNROLL
            hls::vector<T, block_parallel/merge_num> temp_pack = input_streams[i].read();
            for(int j = 0; j < block_parallel/merge_num; j++) {
                output_pack[i * (block_parallel/merge_num) + j] = temp_pack[j];
            }
        }
        output_stream.write(output_pack);
    }
}


#endif
