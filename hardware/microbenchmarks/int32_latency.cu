// Lines 3 - 60, 98 - 134 are from https://github.com/shen203/GPU_Microbenchmark/blob/master/alu_lat_int32/alu_lat_int32.cu

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define THREADS_PER_BLOCK 1
#define THREADS_PER_SM 1
#define BLOCKS_NUM 1
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
#define WARP_SIZE 32
#define REPEAT_TIMES 1024

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


template <class T>
__global__ void lat_int32(uint32_t *startClk, uint32_t *stopClk, T *data1, T *data2, T *res) {
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	register T s1 = data1[gid];
	register T s2 = data2[gid];
	register T result = 0;

	// synchronize all threads
	asm volatile ("bar.sync 0;");

	// start timing
	uint32_t start = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

	for (int j=0 ; j<REPEAT_TIMES ; ++j) {
		asm volatile ("{\t\n"
				"mad.lo.s32 %0, %1, %2 , %0;\n\t"
				"mad.lo.s32 %0, %1, %2 , %0;\n\t"
				"mad.lo.s32 %0, %1, %2 , %0;\n\t"
				"mad.lo.s32 %0, %1, %2 , %0;\n\t"
				"}" : "+r"(result),"+r"(s1),"+r"(s2)
		);
	}
	// synchronize all threads
	asm volatile("bar.sync 0;");

	// stop timing
	uint32_t stop = 0;
	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

	// write time and data back to memory
	startClk[gid] = start;
	stopClk[gid] = stop;
	res[gid] = result;
}

void print_output(float final_int32_clock_cycles, cudaDeviceProp prop) {
  // get timestamp
  char timestamp[32];
  time_t now = time(NULL);
  struct tm utc;
  gmtime_r(&now, &utc);
  strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &utc);

  // get runtime version
  int runtime_version = 0;
  gpuErrchk(cudaRuntimeGetVersion(&runtime_version));

  printf("{\n");
  printf("  \"benchmark\": \"int32_latency.cu\",\n");
  printf("  \"parameter\": \"int32_inst_latency_cycles\",\n");
  printf("  \"value\": %.4f,\n", final_int32_clock_cycles); // determine
  printf("  \"unit\": \"cycles\",\n");
  //printf("  \"gpu\": \"%s\",\n", gpu);
  printf("  \"gpu_name\": \"%s\",\n", prop.name);
  printf("  \"arch\": \"sm_%d%d\",\n", prop.major, prop.minor);
  printf("  \"num_sms\": %d,\n", prop.multiProcessorCount);
  //fprintf(output_file, "  \"stats\": {\n");
  //fprintf(output_file, "    \"median\": %.4f,\n", median);
  //fprintf(output_file, "    \"mean\": %.4f,\n", mean);
  //fprintf(output_file, "    \"std\": %.4f,\n", stddev);
  //fprintf(output_file, "    \"min\": %.4f,\n", mn);
  //fprintf(output_file, "    \"max\": %.4f\n", mx);
  //printf("  },\n");

  //printf("  \"clock\": {\"sm_clock_ghz\": %.4f},\n", sm_clock_ghz);
  printf("  \"cuda_version\": \"%d.%d\",\n", runtime_version / 1000, (runtime_version % 1000) / 10);
  printf("  \"nvcc_version\": \"%d.%d\",\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
  printf("  \"timestamp\": \"%s\"\n", timestamp);
  printf("}\n");
}

int main(int argc, char **argv){
	uint32_t *startClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	int32_t *data1 = (int32_t*) malloc(TOTAL_THREADS*sizeof(int32_t));
	int32_t *data2 = (int32_t*) malloc(TOTAL_THREADS*sizeof(int32_t));
	int32_t *res = (int32_t*) malloc(TOTAL_THREADS*sizeof(int32_t));

	uint32_t *startClk_g;
	uint32_t *stopClk_g;
	int32_t *data1_g;
	int32_t *data2_g;
	int32_t *res_g;

	for (uint32_t i=0; i<TOTAL_THREADS; i++) {
		data1[i] = (int32_t)i;
		data2[i] = (int32_t)i;
	}

	gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&data1_g, TOTAL_THREADS*sizeof(int32_t)) );
	gpuErrchk( cudaMalloc(&data2_g, TOTAL_THREADS*sizeof(int32_t)) );
	gpuErrchk( cudaMalloc(&res_g, TOTAL_THREADS*sizeof(int32_t)) );

	gpuErrchk( cudaMemcpy(data1_g, data1, TOTAL_THREADS*sizeof(int32_t), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(data2_g, data2, TOTAL_THREADS*sizeof(int32_t), cudaMemcpyHostToDevice) );

	lat_int32<int32_t><<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, data1_g, data2_g, res_g);
	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(res, res_g, TOTAL_THREADS*sizeof(int32_t), cudaMemcpyDeviceToHost) );

	float latency;
	latency = ((float)(stopClk[0]-startClk[0]))/((float)(REPEAT_TIMES*4));
  uint32_t total_clock = stopClk[0]-startClk[0];

  //gasp-gpu analysis----------------------------------------------------------------------------

  cudaDeviceProp prop;
  gpuErrchk(cudaGetDeviceProperties(&prop, 0));

  print_output(latency, prop);


	return 0;
}
