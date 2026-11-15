/*
 * Adapted from the Accel-Sim GPU microbenchmark suite:
 *   https://github.com/accel-sim/accel-sim-framework
 *   util/tuner/GPU_Microbenchmark/ubench/shd/shared_lat/shared_lat.cu
 *   (release branch, commit aa95f49; BSD 2-Clause, notice below)
 *
 * M. Khairy, Z. Shen, T. M. Aamodt, T. G. Rogers, "Accel-Sim: An Extensible
 * Simulation Framework for Validated GPU Modeling," ISCA 2020.
 *
 * BSD 2-Clause License
 *
 * Copyright (c) 2018-2021, Vijay Kandiah, Junrui Pan, Mahmoud Khairy, Scott
 * Peverelle, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
 * Northwestern University, Purdue University, The University of British Columbia
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer;
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution;
 * 3. Neither the names of Northwestern University, Purdue University,
 *    The University of British Columbia nor the names of their contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * EDIT [gasp-gpu]: pointer-chasing array moved from __shared__ to __constant__
 * to measure constant cache latency.
 * EDIT [gasp-gpu]: error checks, clock reads, metadata and output use common.h.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

#define CONST_MEM_SIZE 256 // to fit inside per-SM constant cache
#define THREADS_NUM 1
#define WARP_SIZE 4
#define ITERS 4096

// EDIT: use __constant__ instead of __shared__ [gasp-gpu]
__constant__ uint64_t s_a[CONST_MEM_SIZE];

__global__ void const_lat(uint64_t *startClk, uint64_t *stopClk, uint64_t *dsink) {
    // thread index
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t uid = bid * blockDim.x + tid;

    if (uid == 0) {
        // initalize pointer chaser
        uint64_t p_chaser = 0;

        // warm up constant cache so timed loop is all hits
        for (uint32_t i = 0; i < CONST_MEM_SIZE; i += WARP_SIZE) {
            p_chaser = s_a[p_chaser];
        }

        // start timing
        uint64_t start = read_clock64();

        // pointer-chasing loop
        #pragma unroll 16
        for (uint32_t i = 0; i < ITERS; ++i) {
            p_chaser = s_a[p_chaser];
        }

        // stop timing
        uint64_t stop = 0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop) : "l"(p_chaser) : "memory");

        // write time and data back to memory
        startClk[uid] = start;
        stopClk[uid] = stop;
        dsink[uid] = p_chaser;
    }
}

int main(int argc, char **argv) {
    std::string gpu_arg;
    std::string out_dir = executable_dir(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) gpu_arg = argv[++i];
        else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) out_dir = argv[++i];
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) ++i;
    }

    uint64_t *h_a = (uint64_t*) malloc(CONST_MEM_SIZE * sizeof(uint64_t));
    for (uint32_t i = 0; i < (CONST_MEM_SIZE - WARP_SIZE); i += WARP_SIZE) {
        h_a[i] = (i + WARP_SIZE) % CONST_MEM_SIZE;
    }
    CUDA_CHECK( cudaMemcpyToSymbol(s_a, h_a, CONST_MEM_SIZE * sizeof(uint64_t)) );

    uint64_t *startClk = (uint64_t*) malloc(THREADS_NUM * sizeof(uint64_t));
    uint64_t *stopClk = (uint64_t*) malloc(THREADS_NUM * sizeof(uint64_t));
    uint64_t *dsink = (uint64_t*) malloc(THREADS_NUM * sizeof(uint64_t));

    uint64_t *startClk_g;
    uint64_t *stopClk_g;
    uint64_t *dsink_g;

    CUDA_CHECK( cudaMalloc(&startClk_g, THREADS_NUM * sizeof(uint64_t)) ); // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
    CUDA_CHECK( cudaMalloc(&stopClk_g, THREADS_NUM * sizeof(uint64_t)) ); // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
    CUDA_CHECK( cudaMalloc(&dsink_g, THREADS_NUM * sizeof(uint64_t)) );

    const_lat<<<1, THREADS_NUM>>>(startClk_g, stopClk_g, dsink_g);
    CUDA_CHECK( cudaPeekAtLastError() );

    CUDA_CHECK( cudaMemcpy(startClk, startClk_g, THREADS_NUM * sizeof(uint64_t), cudaMemcpyDeviceToHost) ); // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
    CUDA_CHECK( cudaMemcpy(stopClk, stopClk_g, THREADS_NUM * sizeof(uint64_t), cudaMemcpyDeviceToHost) ); // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
    CUDA_CHECK( cudaMemcpy(dsink, dsink_g, THREADS_NUM * sizeof(uint64_t), cudaMemcpyDeviceToHost) );

    float latency;
    latency = ((float)(stopClk[0] - startClk[0])) / ((float)(ITERS));

    // EDIT: SM clock from common.h clock_kernel so we can report the global speed
    double sm_clock_ghz = 0.0;
    {
        unsigned long long *d_clock;
        CUDA_CHECK( cudaMalloc(&d_clock, 2 * sizeof(unsigned long long)) );
        clock_kernel<<<1, 1>>>(d_clock);
        CUDA_CHECK( cudaGetLastError() );
        CUDA_CHECK( cudaDeviceSynchronize() );
        unsigned long long c[2];
        CUDA_CHECK( cudaMemcpy(c, d_clock, sizeof(c), cudaMemcpyDeviceToHost) );
        CUDA_CHECK( cudaFree(d_clock) );
        if (c[1] > 0) sm_clock_ghz = (double)c[0] / (double)c[1];
    }

    //gasp-gpu output------------------------------------------------------------------------------
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int runtime_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    const std::string gpu = gpu_arg.empty() ? short_gpu_name(prop.name) : gpu_arg;

    std::string json;
    json += "{\n";
    json += "  \"benchmark\": \"constant_memory_load.cu\",\n";
    json += "  \"parameter\": \"const_mem_latency_cycles\",\n";
    json += strprintf("  \"value\": %.4f,\n", latency);
    json += "  \"unit\": \"cycles\",\n";
    json += strprintf("  \"gpu\": \"%s\",\n", gpu.c_str());
    json += strprintf("  \"gpu_name\": \"%s\",\n", json_escape(prop.name).c_str());
    json += strprintf("  \"arch\": \"sm_%d%d\",\n", prop.major, prop.minor);
    json += strprintf("  \"num_sms\": %d,\n", prop.multiProcessorCount);
    json += strprintf("  \"clock\": {\"sm_clock_ghz\": %.4f},\n", sm_clock_ghz);
    json += strprintf("  \"cuda_version\": \"%s\",\n", cuda_version_string(runtime_version).c_str());
    json += strprintf("  \"nvcc_version\": \"%d.%d\",\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
    json += strprintf("  \"driver_version\": \"%s\",\n", json_escape(driver_version()).c_str());
    json += strprintf("  \"git_commit\": \"%s\",\n", json_escape(git_commit()).c_str());
    json += strprintf("  \"timestamp\": \"%s\"\n", utc_timestamp().c_str());
    json += "}\n";

    const std::string out_path = write_run_json(out_dir, "const_mem_latency_cycles", gpu, json);
    if (out_path.empty()) return 1;

    printf("%s", json.c_str());
    fprintf(stderr, "Saved %s (%.3f cycles per const load)\n", out_path.c_str(), latency);

    CUDA_CHECK(cudaFree(startClk_g));
    CUDA_CHECK(cudaFree(stopClk_g));
    CUDA_CHECK(cudaFree(dsink_g));
    free(h_a);
    free(startClk);
    free(stopClk);
    free(dsink);
    return 0;
}
