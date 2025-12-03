#ifndef _PE_H_
#define _PE_H_

#include "config.h"

// DSP48E2: A27 B18 C48
// DSP58:   A27 B24 C58

template <bool is_last_A = false, bool is_last_B = false>
void PE_fp32xfp32_2D(
    hls::stream<float>& A_in,
    hls::stream<float>& A_out,
    hls::stream<float>& B_in,
    hls::stream<float>& B_out,
    hls::stream<float>& C,
    const int k_size
){
    // // versal FPGA
    // float sum = 0;
    // PE_loop: for(int k = 0; k < k_size; k++){
    // #pragma HLS PIPELINE II=1
    //     float a_val = A_in.read();
    //     if(!is_last_A) A_out.write(a_val);
    //     float b_val = B_in.read();
    //     if(!is_last_B) B_out.write(b_val);
    //     sum += a_val * b_val;
    // }
    // C.write(sum);
    
    // ultrascale+ FPGA
    float p_sum[4];
    #pragma HLS ARRAY_PARTITION variable=p_sum type=complete
    init_sum_loop: for(int i = 0; i < 4; i++){
    #pragma HLS unroll
        p_sum[i] = 0;
    }
    // //u250 with vitis 2024.1
    // PE_loop: for(int k = 0; k < k_size; k++){
    // #pragma HLS PIPELINE II=1
    // #pragma HLS dependence variable=p_sum inter false
    //     float a_val = A_in.read();
    //     if(!is_last_A) A_out.write(a_val);
    //     float b_val = B_in.read();
    //     if(!is_last_B) B_out.write(b_val);
    //     p_sum[k % 4] += a_val * b_val;
    // }
    // float temp0 = p_sum[0] + p_sum[1];
    // float temp1 = p_sum[2] + p_sum[3];
    // float sum = temp0 + temp1;
    // C.write(sum);

    //u280 with vitis 2022.1
    PE_loop: for(int k = 0; k < k_size/4; k++){
    #pragma HLS PIPELINE II=4
        float a_val_0 = A_in.read();
        if(!is_last_A) A_out.write(a_val_0);
        float b_val_0 = B_in.read();
        if(!is_last_B) B_out.write(b_val_0);
        p_sum[0] += a_val_0 * b_val_0;

        float a_val_1 = A_in.read();
        if(!is_last_A) A_out.write(a_val_1);
        float b_val_1 = B_in.read();
        if(!is_last_B) B_out.write(b_val_1);
        p_sum[1] += a_val_1 * b_val_1;

        float a_val_2 = A_in.read();
        if(!is_last_A) A_out.write(a_val_2);
        float b_val_2 = B_in.read();
        if(!is_last_B) B_out.write(b_val_2);
        p_sum[2] += a_val_2 * b_val_2;

        float a_val_3 = A_in.read();
        if(!is_last_A) A_out.write(a_val_3);
        float b_val_3 = B_in.read();
        if(!is_last_B) B_out.write(b_val_3);
        p_sum[3] += a_val_3 * b_val_3;        
    }

    float temp0 = p_sum[0] + p_sum[1];
    float temp1 = p_sum[2] + p_sum[3];
    float sum = temp0 + temp1;
    C.write(sum);
}

void tree_adder_4(float p_sum[4], float& sum){
    float temp0 = p_sum[0] + p_sum[1];
    float temp1 = p_sum[2] + p_sum[3];
    sum = temp0 + temp1;
}

template <bool is_last_A = false, bool is_last_B = false>
void PE_fp32xfp32_pack_1x2_2D(
    hls::stream<float>& A_in,
    hls::stream<float>& A_out,
    hls::stream<ap_uint<64>>& B_in,
    hls::stream<ap_uint<64>>& B_out,
    hls::stream<float>& C,
    const int k_size
){
    // // versal FPGA
    // float sum_0 = 0;
    // float sum_1 = 0;
    // PE_loop: for(int k = 0; k < k_size; k++){
    // #pragma HLS PIPELINE II=1
    //     float a = A_in.read();
    //     if(!is_last_A) A_out.write(a);
    //     ap_uint<64> b = B_in.read();
    //     if(!is_last_B) B_out.write(b);
    //     ap_uint<32> b_0_temp = b(31, 0);
    //     ap_uint<32> b_1_temp = b(63, 32);
    //     float b_0 = *(float*)(& b_0_temp);
    //     float b_1 = *(float*)(& b_1_temp);
    //     sum_0 += a * b_0;
    //     sum_1 += a * b_1;
    // }
    // C.write(sum_0);
    // C.write(sum_1);

    // ultrascale+ FPGA
    float p_sum_0[4];
    #pragma HLS ARRAY_PARTITION variable=p_sum_0 type=complete
    float p_sum_1[4];
    #pragma HLS ARRAY_PARTITION variable=p_sum_1 type=complete
    init_sum_loop: for(int i = 0; i < 4; i++){
    #pragma HLS unroll
        p_sum_0[i] = 0;
        p_sum_1[i] = 0;
    }

    // //u250 with vitis 2024.1
    // PE_loop: for(int k = 0; k < k_size; k++){
    // #pragma HLS PIPELINE II=1
    // #pragma HLS dependence variable=p_sum_0 inter false
    // #pragma HLS dependence variable=p_sum_1 inter false
    //     float a = A_in.read();
    //     if(!is_last_A) A_out.write(a);

    //     ap_uint<64> b = B_in.read();
    //     if(!is_last_B) B_out.write(b);
    //     ap_uint<32> b_0_temp = b(31, 0);
    //     ap_uint<32> b_1_temp = b(63, 32);
    //     float b_0 = *(float*)(& b_0_temp);
    //     float b_1 = *(float*)(& b_1_temp);

    //     p_sum_0[k % 4] += a * b_0;
    //     p_sum_1[k % 4] += a * b_1;
    // }

    //u280 with vitis 2022.1
    PE_loop: for(int k = 0; k < k_size/4; k++){
    #pragma HLS PIPELINE II=4
        float      a0  = A_in.read();      if(!is_last_A) A_out.write(a0);
        ap_uint<64> b0 = B_in.read();      if(!is_last_B) B_out.write(b0);
        ap_uint<32> b0_lo = b0(31, 0);
        ap_uint<32> b0_hi = b0(63, 32);
        float      b0_0 = *(float*)(&b0_lo);
        float      b0_1 = *(float*)(&b0_hi);
        p_sum_0[0] += a0 * b0_0;
        p_sum_1[0] += a0 * b0_1;

        float      a1  = A_in.read();      if(!is_last_A) A_out.write(a1);
        ap_uint<64> b1 = B_in.read();      if(!is_last_B) B_out.write(b1);
        ap_uint<32> b1_lo = b1(31, 0);
        ap_uint<32> b1_hi = b1(63, 32);
        float      b1_0 = *(float*)(&b1_lo);
        float      b1_1 = *(float*)(&b1_hi);
        p_sum_0[1] += a1 * b1_0;
        p_sum_1[1] += a1 * b1_1;

        float      a2  = A_in.read();      if(!is_last_A) A_out.write(a2);
        ap_uint<64> b2 = B_in.read();      if(!is_last_B) B_out.write(b2);
        ap_uint<32> b2_lo = b2(31, 0);
        ap_uint<32> b2_hi = b2(63, 32);
        float      b2_0 = *(float*)(&b2_lo);
        float      b2_1 = *(float*)(&b2_hi);
        p_sum_0[2] += a2 * b2_0;
        p_sum_1[2] += a2 * b2_1;

        float      a3  = A_in.read();      if(!is_last_A) A_out.write(a3);
        ap_uint<64> b3 = B_in.read();      if(!is_last_B) B_out.write(b3);
        ap_uint<32> b3_lo = b3(31, 0);
        ap_uint<32> b3_hi = b3(63, 32);
        float      b3_0 = *(float*)(&b3_lo);
        float      b3_1 = *(float*)(&b3_hi);
        p_sum_0[3] += a3 * b3_0;
        p_sum_1[3] += a3 * b3_1;
    }

    float sum_0 = 0;
    float sum_1 = 0;
    #pragma HLS allocation function instances=tree_adder_4 limit=1
    tree_adder_4(p_sum_0, sum_0);
    tree_adder_4(p_sum_1, sum_1);
    C.write(sum_0);
    C.write(sum_1);
}

template <bool is_last_A = false, bool is_last_B = false>
void PE_fp32xfp32_pack_2x2_2D(
    hls::stream<ap_uint<64>>& A_in,
    hls::stream<ap_uint<64>>& A_out,
    hls::stream<ap_uint<64>>& B_in,
    hls::stream<ap_uint<64>>& B_out,
    hls::stream<ap_uint<64>>& C,
    const int k_size
){
    // // versal FPGA
    // float  sum_00 = 0;
    // float  sum_01 = 0;
    // float  sum_10 = 0;
    // float  sum_11 = 0;
    // PE_loop: for(int k = 0; k < k_size; k++){
    // #pragma HLS PIPELINE II=1
    //     ap_uint<64> a = A_in.read();
    //     if(!is_last_A) A_out.write(a);
    //     ap_uint<32> a_0_temp = a(31, 0);
    //     ap_uint<32> a_1_temp = a(63, 32);
    //     float a_0 = *(float*)(& a_0_temp);
    //     float a_1 = *(float*)(& a_1_temp);

    //     ap_uint<64> b = B_in.read();
    //     if(!is_last_B) B_out.write(b);
    //     ap_uint<32> b_0_temp = b(31, 0);
    //     ap_uint<32> b_1_temp = b(63, 32);
    //     float b_0 = *(float*)(& b_0_temp);
    //     float b_1 = *(float*)(& b_1_temp);

    //     sum_00 += a_0 * b_0;
    //     sum_01 += a_0 * b_1;
    //     sum_10 += a_1 * b_0;
    //     sum_11 += a_1 * b_1;
    // }
    // ap_uint<64> sum_pack_0;
    // sum_pack_0(31, 0) = *(ap_uint<32>*)(& sum_00);
    // sum_pack_0(63, 32) = *(ap_uint<32>*)(& sum_10);
    // C.write(sum_pack_0);
    // ap_uint<64> sum_pack_1;
    // sum_pack_1(31, 0) = *(ap_uint<32>*)(& sum_01);
    // sum_pack_1(63, 32) = *(ap_uint<32>*)(& sum_11);
    // C.write(sum_pack_1);


    // ultrascale+ FPGA
    float p_sum_00[4], p_sum_01[4];
    #pragma HLS ARRAY_PARTITION variable=p_sum_00 type=complete
    #pragma HLS ARRAY_PARTITION variable=p_sum_01 type=complete
    float p_sum_10[4], p_sum_11[4];
    #pragma HLS ARRAY_PARTITION variable=p_sum_10 type=complete
    #pragma HLS ARRAY_PARTITION variable=p_sum_11 type=complete

    init_sum_loop: for(int i = 0; i < 4; i++){
    #pragma HLS unroll
        p_sum_00[i] = 0;
        p_sum_01[i] = 0;
        p_sum_10[i] = 0;
        p_sum_11[i] = 0;
    }

    // //u250 with vitis 2024.1
    // PE_loop: for(int k = 0; k < k_size; k++){
    // #pragma HLS PIPELINE II=1
    // #pragma HLS dependence variable=p_sum_00 inter false
    // #pragma HLS dependence variable=p_sum_01 inter false
    // #pragma HLS dependence variable=p_sum_10 inter false
    // #pragma HLS dependence variable=p_sum_11 inter false
    //     ap_uint<64> a = A_in.read();
    //     if(!is_last_A) A_out.write(a);
    //     ap_uint<32> a_0_temp = a(31, 0);
    //     ap_uint<32> a_1_temp = a(63, 32);
    //     float a_0 = *(float*)(& a_0_temp);
    //     float a_1 = *(float*)(& a_1_temp);

    //     ap_uint<64> b = B_in.read();
    //     if(!is_last_B) B_out.write(b);
    //     ap_uint<32> b_0_temp = b(31, 0);
    //     ap_uint<32> b_1_temp = b(63, 32);
    //     float b_0 = *(float*)(& b_0_temp);
    //     float b_1 = *(float*)(& b_1_temp);

    //     p_sum_00[k % 4] += a_0 * b_0;
    //     // #pragma HLS bind_op variable=p_sum_00 op=fadd impl=fabric
    //     p_sum_01[k % 4] += a_0 * b_1;
    //     // #pragma HLS bind_op variable=p_sum_01 op=fadd impl=fabric
    //     p_sum_10[k % 4] += a_1 * b_0;
    //     // #pragma HLS bind_op variable=p_sum_10 op=fadd impl=fabric
    //     p_sum_11[k % 4] += a_1 * b_1;
    //     // #pragma HLS bind_op variable=p_sum_11 op=fadd impl=fabric
    // }

    //u280 with vitis 2022.1
    PE_loop: for(int k = 0; k < k_size/4; k++){
    #pragma HLS PIPELINE II=4
        inner_loop: for(int j = 0; j < 4; ++j) {
        #pragma HLS UNROLL
            ap_uint<64> a_p = A_in.read();  if(!is_last_A) A_out.write(a_p);
            ap_uint<64> b_p = B_in.read();  if(!is_last_B) B_out.write(b_p);
            
            ap_uint<32> a_lo = a_p(31,0), a_hi = a_p(63,32);
            ap_uint<32> b_lo = b_p(31,0), b_hi = b_p(63,32);

            float a0 = *(float*)(&a_lo);
            float a1 = *(float*)(&a_hi);
            float b0 = *(float*)(&b_lo);
            float b1 = *(float*)(&b_hi);

            p_sum_00[j] += a0 * b0;
            p_sum_01[j] += a0 * b1;
            p_sum_10[j] += a1 * b0;
            p_sum_11[j] += a1 * b1;
        }
    }

    float sum_00 = 0;
    float sum_01 = 0;
    float sum_10 = 0;
    float sum_11 = 0;
    
    #pragma HLS allocation function instances=tree_adder_4 limit=1

    tree_adder_4(p_sum_00, sum_00);
    tree_adder_4(p_sum_10, sum_10);
    ap_uint<64> sum_pack_0;
    sum_pack_0(31, 0) = *(ap_uint<32>*)(& sum_00);
    sum_pack_0(63, 32) = *(ap_uint<32>*)(& sum_10);
    C.write(sum_pack_0);

    tree_adder_4(p_sum_01, sum_01);
    tree_adder_4(p_sum_11, sum_11);
    ap_uint<64> sum_pack_1;
    sum_pack_1(31, 0) = *(ap_uint<32>*)(& sum_01);
    sum_pack_1(63, 32) = *(ap_uint<32>*)(& sum_11);
    C.write(sum_pack_1);
}


template <bool is_uint_A = false, int max_log2_k_size = 10, bool is_last_A = false, bool is_last_B = false>
void PE_i4xi4_pack_1x2_2D(
    hls::stream<ap_int<4>>& A_in, 
    hls::stream<ap_int<4>>& A_out,
    hls::stream<ap_uint<8>>& B_in, 
    hls::stream<ap_uint<8>>& B_out,
    hls::stream<ap_int<max_log2_k_size + 8>>& C_out, 
    int k_size
){
    ap_int<2*max_log2_k_size + 16> pack_c = 0;
    PE_LOOP: for (int k = 0; k < k_size; k++) {
    #pragma HLS PIPELINE II=1
        ap_int<4> a = A_in.read();
        ap_uint<8> b = B_in.read();
        ap_int<4> b0 = b(3, 0);
        ap_int<4> b1 = b(7, 4);
        ap_int<5> b1_temp = b[3] ? ap_int<5>(b1 + ap_int<4>(-1)) : ap_int<5>(b1);
        ap_int<max_log2_k_size + 4> b0_sign_ex = b[3] ? ap_int<max_log2_k_size + 4>(-1) : ap_int<max_log2_k_size + 4>(0);
        ap_int<max_log2_k_size + 13> pack_b = (b1_temp, b0_sign_ex, b0);
        
        if(is_uint_A)
            pack_c += ap_uint<4>(a) * pack_b;
        else
            pack_c += a * pack_b;
        if(!is_last_A) A_out.write(a);
        if(!is_last_B) B_out.write(b);
    }
    ap_int<max_log2_k_size + 8> c0 = pack_c.range(max_log2_k_size + 7, 0);
    ap_int<max_log2_k_size + 8> c1 = pack_c.range(2*max_log2_k_size + 15, max_log2_k_size + 8);
    c1 = c1 + c0[max_log2_k_size + 7];
    C_out.write(c0);
    C_out.write(c1);
}


template <bool is_uint_A = false, int max_log2_k_size = 10, bool is_last_A = false, bool is_last_B = false>
void PE_i4xi4_pack_2x2_2D( 
    hls::stream<ap_uint<8>>& A_in, 
    hls::stream<ap_uint<8>>& A_out,
    hls::stream<ap_uint<8>>& B_in, 
    hls::stream<ap_uint<8>>& B_out,
    hls::stream<ap_uint<2*max_log2_k_size + 16>>& C_out, 
    int k_size
){
    ap_int<2*max_log2_k_size + 16> pack_c_0 = 0;
    ap_int<2*max_log2_k_size + 16> pack_c_1 = 0;
    PE_LOOP: for (int k = 0; k < k_size; k++) {
    #pragma HLS PIPELINE II=1
        ap_uint<8> a = A_in.read();
        ap_int<4> a0 = a(3, 0);
        ap_int<4> a1 = a(7, 4);
        ap_uint<8> b = B_in.read();
        ap_int<4> b0 = b(3, 0);
        ap_int<4> b1 = b(7, 4);
        ap_int<5> b1_temp = b[3] ? ap_int<5>(b1 + ap_int<4>(-1)) : ap_int<5>(b1);
        ap_int<max_log2_k_size + 4> b0_sign_ex = b[3] ? ap_int<max_log2_k_size + 4>(-1) : ap_int<max_log2_k_size + 4>(0);
        ap_int<max_log2_k_size + 13> pack_b = (b1_temp, b0_sign_ex, b0);

        if(is_uint_A){
            pack_c_0 += ap_uint<4>(a0) * pack_b;
            pack_c_1 += ap_uint<4>(a1) * pack_b;
        }
        else{
            pack_c_0 += a0 * pack_b;
            pack_c_1 += a1 * pack_b;
        }
        if(!is_last_A) A_out.write(a);
        if(!is_last_B) B_out.write(b);
    }
    ap_int<max_log2_k_size + 8> c_00 = pack_c_0.range(max_log2_k_size + 7, 0);
    ap_int<max_log2_k_size + 8> c_01 = pack_c_0.range(2*max_log2_k_size + 15, max_log2_k_size + 8);
    ap_int<max_log2_k_size + 8> c_10 = pack_c_1.range(max_log2_k_size + 7, 0);
    ap_int<max_log2_k_size + 8> c_11 = pack_c_1.range(2*max_log2_k_size + 15, max_log2_k_size + 8);
    c_01 = c_01 + c_00[max_log2_k_size + 7];
    c_11 = c_11 + c_10[max_log2_k_size + 7];
    C_out.write((c_10, c_00));
    C_out.write((c_11, c_01));
}

template <bool is_uint_A = false, int max_log2_k_size = 10, bool is_last_A = false, bool is_last_B = false>
void PE_i4xi4_pack_2x2_1xDSP_2D(
    hls::stream<ap_uint<8>>& A_in, 
    hls::stream<ap_uint<8>>& A_out,
    hls::stream<ap_uint<8>>& B_in, 
    hls::stream<ap_uint<8>>& B_out,
    hls::stream<ap_uint<2*max_log2_k_size + 16>>& C_out, 
    int k_size
){
    ap_int<max_log2_k_size + 8> C_out_00 = 0;
    ap_int<max_log2_k_size + 8> C_out_01 = 0;
    ap_int<max_log2_k_size + 8> C_out_10 = 0;
    ap_int<max_log2_k_size + 8> C_out_11 = 0;
    PE_LOOP: for (int k = 0; k < k_size; k++) {
    #pragma HLS PIPELINE II=1
        ap_uint<8> a = A_in.read();
        ap_uint<8> b = B_in.read();

        if(!is_last_A) A_out.write(a);
        if(!is_last_B) B_out.write(b);

        ap_int<21> pack_a;
        if(is_uint_A){
            ap_uint<4> a0 = a(3, 0);
            ap_uint<4> a1 = a(7, 4);
            pack_a = (ap_uint<1>(0), a1, ap_uint<12>(0), a0); 
        }
        else{
            ap_int<4> a0 = a(3, 0);
            ap_int<4> a1 = a(7, 4);
            ap_int<5> a1_temp = a[3] ? ap_int<5>(a1 + ap_int<4>(-1)) : ap_int<5>(a1);
            ap_int<12> a0_sign_ex = a[3] ? ap_int<12>(-1) : ap_int<12>(0);
            pack_a = (a1_temp, a0_sign_ex, a0);
        }
        
        ap_int<4> b0 = b(3, 0);
        ap_int<4> b1 = b(7, 4);
        ap_int<5> b1_temp = b[3] ? ap_int<5>(b1 + ap_int<4>(-1)) : ap_int<5>(b1);
        ap_int<4> b0_sign_ex = b[3] ? ap_int<4>(-1) : ap_int<4>(0);
        ap_int<13> pack_b = (b1_temp, b0_sign_ex, b0);
        
        // ap_int<21> pack_a = ap_int<20>((a(7, 4), ap_uint<16>(0))) + ap_int<4>(a(3,0));
        // ap_int<13> pack_b = ap_int<12>((b(7, 4), ap_uint<8>(0))) + ap_int<4>(b(3,0));
        
        ap_int<32> pack_c = pack_a * pack_b;
        #pragma HLS bind_op variable=pack_c op=mul impl=dsp
        ap_int<8> c00 = pack_c.range(7, 0);
        ap_int<8> c01 = pack_c.range(15, 8);
        ap_int<8> c10 = pack_c.range(23, 16);
        ap_int<8> c11 = pack_c.range(31, 24);
        ap_uint<1> c00_b = c00[7];
        ap_uint<1> c01_b = c01[7];
        ap_uint<1> c10_b = c10[7];
        c01 = c01 + c00_b;
        c10 = c10 + c01_b;
        c11 = c11 + c10_b;
        C_out_00 += c00;
        C_out_01 += c01;
        C_out_10 += c10;
        C_out_11 += c11;
    }
    C_out.write((C_out_10, C_out_00));
    C_out.write((C_out_11, C_out_01));
}


template <bool is_uint_A = false, int max_log2_k_size = 10, bool is_last_A = false, bool is_last_B = false>
void PE_i8xi8_pack_1x2_1xDSP_2D(
    hls::stream<ap_int<8>>& A_in, 
    hls::stream<ap_int<8>>& A_out,
    hls::stream<ap_uint<16>>& B_in, 
    hls::stream<ap_uint<16>>& B_out,
    hls::stream<ap_int<max_log2_k_size + 16>>& C_out, 
    int k_size
){
    ap_int<max_log2_k_size + 16> C_out_0 = 0;
    ap_int<max_log2_k_size + 16> C_out_1 = 0;
    PE_LOOP: for (int k = 0; k < k_size; k++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=1 max=(1<<max_log2_k_size)
        ap_int<8> a = A_in.read();
        ap_uint<16> b = B_in.read();
        if(!is_last_A) A_out.write(a);
        if(!is_last_B) B_out.write(b);
        
        ap_int<8> b0 = b(7, 0);
        ap_int<8> b1 = b(15, 8);
        ap_int<9> b1_temp = b[7] ? ap_int<9>(b1 + ap_int<8>(-1)) : ap_int<9>(b1);
        ap_int<8> b0_sign_ex = b[7] ? ap_int<8>(-1) : ap_int<8>(0);
        ap_int<25> pack_b = (b1_temp, b0_sign_ex, b0);

        ap_int<32> pack_c;
        if(is_uint_A)
            pack_c = ap_uint<8>(a) * pack_b;
        else
            pack_c = a * pack_b;
        ap_int<16> c0 = pack_c.range(15, 0);
        ap_int<16> c1 = pack_c.range(31, 16);
        ap_uint<1> c0_b = c0[15];
        c1 += c0_b;
        C_out_0 += c0;
        C_out_1 += c1;
    }
    C_out.write(C_out_0);
    C_out.write(C_out_1);
}




template <bool is_uint_A = false, int max_log2_k_size = 10, bool is_last_A = false, bool is_last_B = false>
void PE_i8xi8_pack_2x2_2xDSP_2D( 
    hls::stream<ap_uint<16>>& A_in, 
    hls::stream<ap_uint<16>>& A_out,
    hls::stream<ap_uint<16>>& B_in, 
    hls::stream<ap_uint<16>>& B_out,
    hls::stream<ap_uint<2*max_log2_k_size + 32>>& C_out, 
    int k_size
){
    ap_int<max_log2_k_size + 16> C_out_00 = 0;
    ap_int<max_log2_k_size + 16> C_out_01 = 0;
    ap_int<max_log2_k_size + 16> C_out_10 = 0;
    ap_int<max_log2_k_size + 16> C_out_11 = 0;
    PE_LOOP: for (int k = 0; k < k_size; k++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=1 max=(1<<max_log2_k_size)
        ap_uint<16> a = A_in.read();
        ap_uint<16> b = B_in.read();

        if(!is_last_A) A_out.write(a);
        if(!is_last_B) B_out.write(b);

        ap_int<8> a0 = a(7, 0);
        ap_int<8> a1 = a(15, 8);

        ap_int<8> b0 = b(7, 0);
        ap_int<8> b1 = b(15, 8);
        ap_int<9> b1_temp = b[7] ? ap_int<9>(b1 + ap_int<8>(-1)) : ap_int<9>(b1);
        ap_int<8> b0_sign_ex = b[7] ? ap_int<8>(-1) : ap_int<8>(0);
        ap_int<25> pack_b = (b1_temp, b0_sign_ex, b0);

        ap_int<32> pack_c_0;
        ap_int<32> pack_c_1;
        if(is_uint_A){
            pack_c_0 = ap_uint<8>(a0) * pack_b;
            pack_c_1 = ap_uint<8>(a1) * pack_b;
        }
        else{
            pack_c_0 = a0 * pack_b;
            pack_c_1 = a1 * pack_b;
        }
        ap_int<16> c00 = pack_c_0.range(15, 0);
        ap_int<16> c01 = pack_c_0.range(31, 16);
        ap_int<16> c10 = pack_c_1.range(15, 0);
        ap_int<16> c11 = pack_c_1.range(31, 16);
        
        ap_uint<1> c00_b = c00[15];
        ap_uint<1> c10_b = c10[15];
        c01 += c00_b;
        c11 += c10_b;
        C_out_00 += c00;
        C_out_01 += c01;
        C_out_10 += c10;
        C_out_11 += c11;
    }
    C_out.write((C_out_10, C_out_00));
    C_out.write((C_out_11, C_out_01));
}



template <bool is_last_A = false>
void PE_fp32xfp32_1D(
    hls::stream<float>& A_in,
    hls::stream<float>& A_out,
    hls::stream<float>& B_in,
    hls::stream<float>& C,
    const int k_size
){
    // // versal FPGA
    // float sum = 0;
    // PE_loop: for(int k = 0; k < k_size; k++){
    // #pragma HLS PIPELINE II=1
    //     float a_val = A_in.read();
    //     if(!is_last_A) A_out.write(a_val);
    //     float b_val = B_in.read();
    //     sum += a_val * b_val;
    // }
    // C.write(sum);
    
    // ultrascale FPGA
    float p_sum[4];
    #pragma HLS ARRAY_PARTITION variable=p_sum type=complete
    init_sum_loop: for(int i = 0; i < 4; i++){
    #pragma HLS unroll
        p_sum[i] = 0;
    }

    // //u250 with vitis 2024.1
    // PE_loop: for(int k = 0; k < k_size; k++){
    // #pragma HLS PIPELINE II=1
    // #pragma HLS dependence variable=p_sum inter false
    //     float a_val = A_in.read();
    //     if(!is_last_A) A_out.write(a_val);
    //     float b_val = B_in.read();
    //     p_sum[k % 4] += a_val * b_val;
    // }

    //u280 with vitis 2022.1
    PE_loop: for(int k = 0; k < k_size/4; k++){
    #pragma HLS PIPELINE II=4
        float a_val_0 = A_in.read();
        if(!is_last_A) A_out.write(a_val_0);
        float b_val_0 = B_in.read();
        p_sum[0] += a_val_0 * b_val_0;

        float a_val_1 = A_in.read();
        if(!is_last_A) A_out.write(a_val_1);
        float b_val_1 = B_in.read();
        p_sum[1] += a_val_1 * b_val_1;

        float a_val_2 = A_in.read();
        if(!is_last_A) A_out.write(a_val_2);
        float b_val_2 = B_in.read();
        p_sum[2] += a_val_2 * b_val_2;

        float a_val_3 = A_in.read();
        if(!is_last_A) A_out.write(a_val_3);
        float b_val_3 = B_in.read();
        p_sum[3] += a_val_3 * b_val_3;        
    }

    float temp0 = p_sum[0] + p_sum[1];
    float temp1 = p_sum[2] + p_sum[3];
    float sum = temp0 + temp1;
    C.write(sum);
}

template <bool is_last_A = false>
void PE_fp32xfp32_pack_1x2_1D(
    hls::stream<float>& A_in,
    hls::stream<float>& A_out,
    hls::stream<ap_uint<64>>& B_in,
    hls::stream<float>& C,
    const int k_size
){
    // // versal FPGA
    // float sum_0 = 0;
    // float sum_1 = 0;
    // PE_loop: for(int k = 0; k < k_size; k++){
    // #pragma HLS PIPELINE II=1
    //     float a = A_in.read();
    //     if(!is_last_A) A_out.write(a);
    //     ap_uint<64> b = B_in.read();
    //     ap_uint<32> b_0_temp = b(31, 0);
    //     ap_uint<32> b_1_temp = b(63, 32);
    //     float b_0 = *(float*)(& b_0_temp);
    //     float b_1 = *(float*)(& b_1_temp);
    //     sum_0 += a * b_0;
    //     sum_1 += a * b_1;
    // }
    // C.write(sum_0);
    // C.write(sum_1);

    // ultrascale+ FPGA
    float p_sum_0[4];
    #pragma HLS ARRAY_PARTITION variable=p_sum_0 type=complete
    float p_sum_1[4];
    #pragma HLS ARRAY_PARTITION variable=p_sum_1 type=complete
    init_sum_loop: for(int i = 0; i < 4; i++){
    #pragma HLS unroll
        p_sum_0[i] = 0;
        p_sum_1[i] = 0;
    }

    // //u250 with vitis 2024.1
    // PE_loop: for(int k = 0; k < k_size; k++){
    // #pragma HLS PIPELINE II=1
    // #pragma HLS dependence variable=p_sum_0 inter false
    // #pragma HLS dependence variable=p_sum_1 inter false
    //     float a = A_in.read();
    //     if(!is_last_A) A_out.write(a);

    //     ap_uint<64> b = B_in.read();
    //     ap_uint<32> b_0_temp = b(31, 0);
    //     ap_uint<32> b_1_temp = b(63, 32);
    //     float b_0 = *(float*)(& b_0_temp);
    //     float b_1 = *(float*)(& b_1_temp);

    //     p_sum_0[k % 4] += a * b_0;
    //     p_sum_1[k % 4] += a * b_1;
    // }

    //u280 with vitis 2022.1
    PE_loop: for(int k = 0; k < k_size/4; k++){
    #pragma HLS PIPELINE II=4
        float      a0  = A_in.read();      if(!is_last_A) A_out.write(a0);
        ap_uint<64> b0 = B_in.read();      
        ap_uint<32> b0_lo = b0(31, 0);
        ap_uint<32> b0_hi = b0(63, 32);
        float      b0_0 = *(float*)(&b0_lo);
        float      b0_1 = *(float*)(&b0_hi);
        p_sum_0[0] += a0 * b0_0;
        p_sum_1[0] += a0 * b0_1;

        float      a1  = A_in.read();      if(!is_last_A) A_out.write(a1);
        ap_uint<64> b1 = B_in.read();      
        ap_uint<32> b1_lo = b1(31, 0);
        ap_uint<32> b1_hi = b1(63, 32);
        float      b1_0 = *(float*)(&b1_lo);
        float      b1_1 = *(float*)(&b1_hi);
        p_sum_0[1] += a1 * b1_0;
        p_sum_1[1] += a1 * b1_1;

        float      a2  = A_in.read();      if(!is_last_A) A_out.write(a2);
        ap_uint<64> b2 = B_in.read();      
        ap_uint<32> b2_lo = b2(31, 0);
        ap_uint<32> b2_hi = b2(63, 32);
        float      b2_0 = *(float*)(&b2_lo);
        float      b2_1 = *(float*)(&b2_hi);
        p_sum_0[2] += a2 * b2_0;
        p_sum_1[2] += a2 * b2_1;

        float      a3  = A_in.read();      if(!is_last_A) A_out.write(a3);
        ap_uint<64> b3 = B_in.read();      
        ap_uint<32> b3_lo = b3(31, 0);
        ap_uint<32> b3_hi = b3(63, 32);
        float      b3_0 = *(float*)(&b3_lo);
        float      b3_1 = *(float*)(&b3_hi);
        p_sum_0[3] += a3 * b3_0;
        p_sum_1[3] += a3 * b3_1;
    }

    float sum_0 = 0;
    float sum_1 = 0;
    #pragma HLS allocation function instances=tree_adder_4 limit=1
    tree_adder_4(p_sum_0, sum_0);
    tree_adder_4(p_sum_1, sum_1);
    C.write(sum_0);
    C.write(sum_1);
}


template <bool is_uint_A = false, int max_log2_k_size = 10, bool is_last_A = false>
void PE_i4xi4_pack_1x2_1D(
    hls::stream<ap_int<4>>& A_in, 
    hls::stream<ap_int<4>>& A_out,
    hls::stream<ap_uint<8>>& B_in, 
    hls::stream<ap_int<max_log2_k_size + 8>>& C_out, 
    int k_size
){
    ap_int<2*max_log2_k_size + 16> pack_c = 0;
    PE_LOOP: for (int k = 0; k < k_size; k++) {
    #pragma HLS PIPELINE II=1
        ap_int<4> a = A_in.read();
        ap_uint<8> b = B_in.read();
        ap_int<4> b0 = b(3, 0);
        ap_int<4> b1 = b(7, 4);
        ap_int<5> b1_temp = b[3] ? ap_int<5>(b1 + ap_int<4>(-1)) : ap_int<5>(b1);
        ap_int<max_log2_k_size + 4> b0_sign_ex = b[3] ? ap_int<max_log2_k_size + 4>(-1) : ap_int<max_log2_k_size + 4>(0);
        ap_int<max_log2_k_size + 13> pack_b = (b1_temp, b0_sign_ex, b0);
        
        if(is_uint_A)
            pack_c += ap_uint<4>(a) * pack_b;
        else
            pack_c += a * pack_b;
        if(!is_last_A) A_out.write(a);
    }
    ap_int<max_log2_k_size + 8> c0 = pack_c.range(max_log2_k_size + 7, 0);
    ap_int<max_log2_k_size + 8> c1 = pack_c.range(2*max_log2_k_size + 15, max_log2_k_size + 8);
    c1 = c1 + c0[max_log2_k_size + 7];
    C_out.write(c0);
    C_out.write(c1);
}



template <bool is_uint_A = false, int max_log2_k_size = 10, bool is_last_A = false>
void PE_i8xi8_pack_1x2_1xDSP_1D(
    hls::stream<ap_int<8>>& A_in, 
    hls::stream<ap_int<8>>& A_out,
    hls::stream<ap_uint<16>>& B_in, 
    hls::stream<ap_int<max_log2_k_size + 16>>& C_out, 
    int k_size
){
    ap_int<max_log2_k_size + 16> C_out_0 = 0;
    ap_int<max_log2_k_size + 16> C_out_1 = 0;
    PE_LOOP: for (int k = 0; k < k_size; k++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=1 max=(1<<max_log2_k_size)
        ap_int<8> a = A_in.read();
        ap_uint<16> b = B_in.read();
        if(!is_last_A) A_out.write(a);
        
        ap_int<8> b0 = b(7, 0);
        ap_int<8> b1 = b(15, 8);
        ap_int<9> b1_temp = b[7] ? ap_int<9>(b1 + ap_int<8>(-1)) : ap_int<9>(b1);
        ap_int<8> b0_sign_ex = b[7] ? ap_int<8>(-1) : ap_int<8>(0);
        ap_int<25> pack_b = (b1_temp, b0_sign_ex, b0);

        ap_int<32> pack_c;
        if(is_uint_A)
            pack_c = ap_uint<8>(a) * pack_b;
        else
            pack_c = a * pack_b;
        ap_int<16> c0 = pack_c.range(15, 0);
        ap_int<16> c1 = pack_c.range(31, 16);
        ap_uint<1> c0_b = c0[15];
        c1 += c0_b;
        C_out_0 += c0;
        C_out_1 += c1;
    }
    C_out.write(C_out_0);
    C_out.write(C_out_1);
}



#endif