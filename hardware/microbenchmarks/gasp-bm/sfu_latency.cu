/*
 * Adapted from the Accel-Sim GPU microbenchmark suite:
 *   https://github.com/accel-sim/gpu-app-collection
 *   src/cuda/GPU_Microbenchmark/ubench/core/sfu_lat_fsqrt/sfu_lat_fsqrt.h
 *   (dev branch, commit ded8dac)
 *
 * M. Khairy, Z. Shen, T. M. Aamodt, T. G. Rogers, "Accel-Sim: An Extensible
 * Simulation Framework for Validated GPU Modeling," ISCA 2020.
 *
 * EDIT [gasp-gpu]: the chained op is rsqrt.approx.ftz.f32 (one MUFU.RSQ), not
 * sin.approx, which adds an FMUL to every step of the chain on Volta+.
 * EDIT [gasp-gpu]: error checks, clock reads, metadata and output use common.h.
 * Output is written to <exe dir>/sfu_inst_latency_cycles.<gpu>.run.json.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

#define THREADS_PER_BLOCK 1
#define THREADS_PER_SM 1
#define BLOCKS_NUM 1
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
#define WARP_SIZE 32
#define REPEAT_TIMES 1024

__global__ void sfu_latency(uint64_t *startClk, uint64_t *stopClk, float *data1, float *res) {
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	register float s1 = data1[gid];
	register float result = s1;

	// synchronize all threads
	asm volatile ("bar.sync 0;");

	// start timing
	uint64_t start = read_clock64();

	// EDIT: rsqrt instead of sin [gasp-gpu]. sin.approx compiles to FMUL + MUFU.SIN
	// on Volta + arch.
	for (int j=0 ; j<REPEAT_TIMES ; ++j) {
		asm volatile ("{\t\n"
				"rsqrt.approx.ftz.f32 %0, %0;\n\t"
				"rsqrt.approx.ftz.f32 %0, %0;\n\t"
				"rsqrt.approx.ftz.f32 %0, %0;\n\t"
				"rsqrt.approx.ftz.f32 %0, %0;\n\t"
				"}" : "+f"(result)
		);
	}
	// synchronize all threads
	asm volatile("bar.sync 0;");

	// stop timing
	uint64_t stop = read_clock64();

	// write time and data back to memory
	startClk[gid] = start;
	stopClk[gid] = stop;
	res[gid] = result;
}

int main(int argc, char **argv){
	(void)argc;
	uint64_t *startClk = (uint64_t*) malloc(TOTAL_THREADS*sizeof(uint64_t));
	uint64_t *stopClk = (uint64_t*) malloc(TOTAL_THREADS*sizeof(uint64_t));
	float *data1 = (float*) malloc(TOTAL_THREADS*sizeof(float));
	float *res = (float*) malloc(TOTAL_THREADS*sizeof(float));

	uint64_t *startClk_g;
	uint64_t *stopClk_g;
	float *data1_g;
	float *res_g;

	for (uint32_t i=0; i<TOTAL_THREADS; i++) {
		data1[i] = 10.124234521;
	}

	CUDA_CHECK( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint64_t)) ); // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
	CUDA_CHECK( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint64_t)) ); // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
	CUDA_CHECK( cudaMalloc(&data1_g, TOTAL_THREADS*sizeof(int32_t)) );
	CUDA_CHECK( cudaMalloc(&res_g, TOTAL_THREADS*sizeof(int32_t)) );

	CUDA_CHECK( cudaMemcpy(data1_g, data1, TOTAL_THREADS*sizeof(int32_t), cudaMemcpyHostToDevice) );

	sfu_latency<<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, data1_g, res_g);
	CUDA_CHECK( cudaPeekAtLastError() );

	CUDA_CHECK( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint64_t), cudaMemcpyDeviceToHost) ); // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
	CUDA_CHECK( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint64_t), cudaMemcpyDeviceToHost) ); // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
	CUDA_CHECK( cudaMemcpy(res, res_g, TOTAL_THREADS*sizeof(float), cudaMemcpyDeviceToHost) );

	float latency;
	latency = ((float)(stopClk[0]-startClk[0]))/((float)(REPEAT_TIMES*4));

	// EDIT: SM clock from common.h clock_kernel so we can report the global speed
	double sm_clock_ghz = 0.0;
	{
		unsigned long long *d_clock;
		CUDA_CHECK( cudaMalloc(&d_clock, 2*sizeof(unsigned long long)) );
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
	const std::string gpu = short_gpu_name(prop.name);

	std::string json;
	json += "{\n";
	json += "  \"benchmark\": \"sfu_latency.cu\",\n";
	json += "  \"parameter\": \"sfu_inst_latency_cycles\",\n";
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

	const std::string out_path = write_run_json(executable_dir(argv[0]), "sfu_inst_latency_cycles", gpu, json);
	if (out_path.empty()) return 1;

	printf("%s", json.c_str());
	fprintf(stderr, "Saved %s (%.3f cycles per SFU op)\n", out_path.c_str(), latency);

	CUDA_CHECK(cudaFree(startClk_g));
	CUDA_CHECK(cudaFree(stopClk_g));
	CUDA_CHECK(cudaFree(data1_g));
	CUDA_CHECK(cudaFree(res_g));
	free(startClk);
	free(stopClk);
	free(data1);
	free(res);
	return 0;
}
