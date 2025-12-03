#ifndef _FHT_H_
#define _FHT_H_

#include "config.h"


template <typename T, int io_parallel, int io_hidden_dim=HIDDEN_DIM, int log2_io_hidden_dim=log2_HIDDEN_DIM, int max_seq_len=MAX_PRE_SEQ_LEN>
void pref_FHT(
    tapa::istream<hls::vector<T, io_parallel>>& data_in,
    tapa::ostream<hls::vector<T, io_parallel>>& data_out,
    int seq_len = max_seq_len,
    const T scale_factor = 1.0
) {
    hls::vector<T, io_parallel> buffer[2][io_hidden_dim];
    #pragma HLS ARRAY_PARTITION variable=buffer complete dim=1
    // #pragma HLS bind_storage variable=buffer type=ram_2p impl=uram
    
    io_block_loop: for (int M = 0; M < seq_len/io_parallel; M++){
    #pragma HLS loop_tripcount min=1 max=max_seq_len/io_parallel
        // Copy input to stage buffer
        Read_Loop: for (int i = 0; i < io_hidden_dim; i++) {
        #pragma HLS PIPELINE II=1
            buffer[0][i] = data_in.read();
        }

        // Omega Network: 
        hls::vector<T, io_parallel> shuffle_buf[io_hidden_dim];
        // #pragma HLS bind_storage variable=shuffle_buf type=ram_2p impl=uram

        const int m = log2_io_hidden_dim;
        Stage_Loop: for (int s = 0; s < m; s++) {
            // 1) Perfect shuffle: 
            Shuffle_Loop: for (int i = 0; i < io_hidden_dim; i++) {
        #pragma HLS PIPELINE II=1
                // ((i << 1) & (N-1)) | (i >> (m-1))
                int idx = ((i << 1) & (io_hidden_dim - 1)) | (i >> (m - 1));
                shuffle_buf[i] = buffer[s % 2][idx];
            }
            // 2) Exchange: 
            Exchange_Loop: for (int i = 0; i < io_hidden_dim; i += 2) {
        #pragma HLS PIPELINE II=1
                hls::vector<T, io_parallel> u_in = shuffle_buf[i];
                hls::vector<T, io_parallel> v_in = shuffle_buf[i + 1];
                hls::vector<T, io_parallel> u_out, v_out;
                for (int k = 0; k < io_parallel; k++) {
        #pragma HLS UNROLL
                    u_out[k] = u_in[k] + v_in[k];
                    v_out[k] = u_in[k] - v_in[k];
                }
                buffer[(s + 1) % 2][i]     = u_out;
                buffer[(s + 1) % 2][i + 1] = v_out;
            }
        }
        
        

        // Write output
        Write_Loop: for (int i = 0; i < io_hidden_dim; i++) {
        #pragma HLS PIPELINE II=1
            hls::vector<T, io_parallel> out_pack;
            for(int k = 0; k < io_parallel; k++) {
                out_pack[k] = buffer[log2_io_hidden_dim % 2][i][k] / scale_factor;
            }
            data_out.write(out_pack);
        }
    }
}

template <typename T, int block_parallel, int io_hidden_dim=INTER_DIM, int log2_io_hidden_dim=log2_INTER_DIM>
void dec_FHT(
    tapa::istream<hls::vector<T, block_parallel>>& data_in,
    tapa::ostream<hls::vector<T, block_parallel>>& data_out,
    const T scale_factor = 1.0
) {
    hls::vector<T, block_parallel> buffer[2][io_hidden_dim/block_parallel];
    #pragma HLS ARRAY_PARTITION variable=buffer complete dim=1
    
    // Copy input to stage buffer
    Read_Loop: for (int i = 0; i < io_hidden_dim/block_parallel; i++) {
    #pragma HLS PIPELINE II=1
        buffer[0][i] = data_in.read();
    }

    // Omega Network: 
    hls::vector<T, block_parallel> shuffle_buf[io_hidden_dim/block_parallel];
    // #pragma HLS bind_storage variable=shuffle_buf type=ram_2p impl=uram

    const int m = log2_io_hidden_dim;
    Stage_Loop: for (int s = 0; s < m; s++) {
        // 1) Perfect shuffle: 
        Shuffle_Loop: for (int i = 0; i < io_hidden_dim/block_parallel; i++) {
        #pragma HLS PIPELINE II=1
            for(int k = 0; k < block_parallel; k++){
                // // ((i << 1) & (N-1)) | (i >> (m-1))
                // int idx = ((i << 1) & (io_hidden_dim - 1)) | (i >> (m - 1));
                // shuffle_buf[i] = buffer[s % 2][idx];

                int idx = (((k * io_hidden_dim/block_parallel + i) << 1) & (io_hidden_dim - 1)) | 
                          ((k * io_hidden_dim/block_parallel + i) >> (m - 1));
                shuffle_buf[i][k] = buffer[s % 2][idx % (io_hidden_dim/block_parallel)][idx / (io_hidden_dim/block_parallel)];
            }
        }
        // 2) Exchange: 
        Exchange_Loop: for (int i = 0; i < io_hidden_dim/block_parallel; i += 2) {
        #pragma HLS PIPELINE II=1
            hls::vector<T, block_parallel> u_in = shuffle_buf[i];
            hls::vector<T, block_parallel> v_in = shuffle_buf[i + 1];
            hls::vector<T, block_parallel> u_out, v_out;
            for (int k = 0; k < block_parallel; k++) {
            #pragma HLS UNROLL
                u_out[k] = u_in[k] + v_in[k];
                v_out[k] = u_in[k] - v_in[k];
            }
            buffer[(s + 1) % 2][i]     = u_out;
            buffer[(s + 1) % 2][i + 1] = v_out;
        }
    }
    

    // Write output
    Write_Loop: for (int i = 0; i < io_hidden_dim/block_parallel; i++) {
    #pragma HLS PIPELINE II=1
        hls::vector<T, block_parallel> out_pack;
        for(int k = 0; k < block_parallel; k++) {
            out_pack[k] = buffer[log2_io_hidden_dim % 2][i][k] / scale_factor;
        }
        data_out.write(out_pack);
    }
}

#endif