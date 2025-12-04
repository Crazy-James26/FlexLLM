# 🔥 FlexLLM: A Composable HLS Library for Rapid LLM Accelerator Design

FlexLLM is a **composable High-Level Synthesis (HLS) library** for rapidly building **hybrid temporal–spatial accelerators** for Large Language Models (LLMs).  
It provides parameterized module templates, optimized memory-access/dataflow components, and a complete quantization suite, enabling FPGA-based LLM systems to be built with **minimal manual engineering effort**.

Using FlexLLM, we implemented a **full Llama-3.2-1B inference system**—including prefill, decode, tokenizer integration, and long-context memory—**in under two months with ~1K lines of code**.

---

## ✨ Key Features

- **Composable HLS Library** for LLM accelerator development  
- **Hybrid Temporal–Spatial Architecture**  
- **Hardware-Efficient Quantization Suite**  
- **Hierarchical Memory Transformer (HMT) Plug-In**  
- **FPGA Deployment Ready**

---

## 📊 Performance Summary

### AMD U280 FPGA (16nm) vs. NVIDIA A100 GPU (7nm)
- 1.29× end-to-end speedup  
- 1.64× higher decode throughput  
- 3.14× better energy efficiency  

### Projected V80 FPGA (7nm)
- 4.71× end-to-end speedup  
- 6.55× decode throughput  
- 4.13× energy efficiency  

### Long-Context (with HMT)
- 23.23× reduced prefill latency  
- 64× longer context window  

---

## 📁 Repository Layout

```
FlexLLM/
├─ Modules/                          # Core FlexLLM module library (compute, quant, memory, data movement)
│
├─ SpinQuant_Llama_32_1B_Ins/        # Llama-3.2-1B-Instruct accelerator (SpinQuant)
│  ├─ parameters/                    # Downloaded model parameters
│  ├─ RapidStream_pref_u280/         # Prefill RapidStream config (U280)
│  ├─ RapidStream_dec_u280/          # Decode RapidStream config (U280)
│  ├─ run/                           # Bitstreams, hosts, and test scripts
│  │  ├─ bitstreams/                 # FPGA .xclbin files
│  │  ├─ parameters/                 # Downloaded parameters
│  │  ├─ llama-3.2-1b-f16.gguf       # Tokenizer (download required)
│  │  ├─ SpinQuant_Prefilling_Decoding_mem_opt
│  │  ├─ SpinQuant_Prefilling_Decoding_mem_opt_demo
│  │  └─ test files (.py/.txt/.csv)
│  └─ TAPA files                     # TAPA HLS kernels, host code, memory configs
│
├─ SpinQuant_Llama_32_1B/            # Llama-3.2-1B accelerator (SpinQuant)
├─ HMT_SpinQuant_Llama_32_1B/        # Llama-3.2-1B-Instruct + SpinQuant + HMT
└─ README.md
```

---

## 📦 Download Required Files

Download parameters & GGUF from:

https://drive.google.com/drive/folders/149QLnEm-NT3fhCgB4Uy7Xda1oLu2zk-7?usp=sharing

Place them in:

```
FlexLLM/your_model/parameters/
FlexLLM/your_model/run/parameters/
FlexLLM/your_model/run/llama-3.2-1b-f16.gguf
```

---

## 🧰 Requirements

- Ubuntu 20.04 / 22.04  
- XRT installed  
- Vitis 2022.2  
- TAPA CLI  
- Compatible FPGA board  

Check FPGA:

```
xbutil examine
```

---

## 🛠 Build (Host Only)

```
export FLEXLLM_HOME=/path/to/FlexLLM
export LLAMA_CPP_ROOT=/path/to/llama.cpp

tapa g++ -- SpinQuant_Prefilling_Decoding_mem_opt_demo.cpp   -I$FLEXLLM_HOME/Modules   -I$LLAMA_CPP_ROOT   -I$LLAMA_CPP_ROOT/include   -I$LLAMA_CPP_ROOT/ggml/include   -I$LLAMA_CPP_ROOT/ggml/include/ggml   $LLAMA_CPP_ROOT/build/bin/libllama.so   -Wl,-rpath,$LLAMA_CPP_ROOT/build/bin   -lpthread -ldl -lm   -o run/SpinQuant_Prefilling_Decoding_mem_opt_demo
```

---

## 🚀 Run on U280

```
./SpinQuant_Prefilling_Decoding_mem_opt_demo   --bitstream_pref bitstreams/SpinQuant_Prefilling_mem_opt_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin   --bitstream_dec  bitstreams/SpinQuant_Decoding_mem_opt_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin   llama-3.2-1b-f16.gguf my_prompt.txt my_answer.txt
```

---

## 📝 Notes for V80 Support

V80 results are estimates. Full bitstreams coming soon.

---

## 🙏 Acknowledgments

We thank AMD — **Fraser Nicholas** and **Michaela Blott** — for support and guidance.