// Lines 3 - 60, 92 - 127 are from https://github.com/shen203/GPU_Microbenchmark/blob/master/alu_lat_int32/alu_lat_int32.cu

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
__global__ void lat_int32(uint64_t *startClk, uint64_t *stopClk, T *data1, T *data2, T *res) { // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	register T s1 = data1[gid];
	register T s2 = data2[gid];
	register T result = 0;

	// synchronize all threads
	asm volatile ("bar.sync 0;");

	// start timing
	uint64_t start = 0; // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
	asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory"); // EDIT: using clock64() instead of clock() [gasp-gpu]

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
	uint64_t stop = 0; // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
	asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop) :: "memory"); // EDIT: using clock64() instead of clock() [gasp-gpu]

	// write time and data back to memory
	startClk[gid] = start;
	stopClk[gid] = stop;
	res[gid] = result;
}

void print_output(float final_int32_clock_cycles) {
  // get timestamp
  char timestamp[32];
  time_t now = time(NULL);
  struct tm utc;
  gmtime_r(&now, &utc);
  strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &utc);

  // get runtime version
  int runtime_version = 0;
  gpuErrchk(cudaRuntimeGetVersion(&runtime_version));

	// get device properties
	cudaDeviceProp prop;
  gpuErrchk(cudaGetDeviceProperties(&prop, 0));

  printf("{\n");
  printf("  \"benchmark\": \"int32_latency.cu\",\n");
  printf("  \"parameter\": \"int32_inst_latency_cycles\",\n");
  printf("  \"value\": %.4f,\n", final_int32_clock_cycles); // <-- what we care about
  printf("  \"unit\": \"cycles\",\n");
  printf("  \"gpu_name\": \"%s\",\n", prop.name);
  printf("  \"arch\": \"sm_%d%d\",\n", prop.major, prop.minor);
  printf("  \"num_sms\": %d,\n", prop.multiProcessorCount);
  printf("  \"cuda_version\": \"%d.%d\",\n", runtime_version / 1000, (runtime_version % 1000) / 10);
  printf("  \"nvcc_version\": \"%d.%d\",\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
  printf("  \"timestamp\": \"%s\"\n", timestamp);
  printf("}\n");
}

int main(int argc, char **argv){
	uint64_t *startClk = (uint64_t*) malloc(TOTAL_THREADS*sizeof(uint64_t)); // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
	uint64_t *stopClk = (uint64_t*) malloc(TOTAL_THREADS*sizeof(uint64_t)); // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
	int32_t *data1 = (int32_t*) malloc(TOTAL_THREADS*sizeof(int32_t));
	int32_t *data2 = (int32_t*) malloc(TOTAL_THREADS*sizeof(int32_t));
	int32_t *res = (int32_t*) malloc(TOTAL_THREADS*sizeof(int32_t));

	uint64_t *startClk_g; // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
	uint64_t *stopClk_g; // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
	int32_t *data1_g;
	int32_t *data2_g;
	int32_t *res_g;

	for (uint32_t i=0; i<TOTAL_THREADS; i++) {
		data1[i] = (int32_t)i;
		data2[i] = (int32_t)i;
	}

	gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint64_t)) ); // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
	gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint64_t)) ); // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
	gpuErrchk( cudaMalloc(&data1_g, TOTAL_THREADS*sizeof(int32_t)) );
	gpuErrchk( cudaMalloc(&data2_g, TOTAL_THREADS*sizeof(int32_t)) );
	gpuErrchk( cudaMalloc(&res_g, TOTAL_THREADS*sizeof(int32_t)) );

	gpuErrchk( cudaMemcpy(data1_g, data1, TOTAL_THREADS*sizeof(int32_t), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(data2_g, data2, TOTAL_THREADS*sizeof(int32_t), cudaMemcpyHostToDevice) );

	lat_int32<int32_t><<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, data1_g, data2_g, res_g);
	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint64_t), cudaMemcpyDeviceToHost) ); // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint64_t), cudaMemcpyDeviceToHost) ); // EDIT: need to use uint64_t to store clock64() [gasp-gpu]
	gpuErrchk( cudaMemcpy(res, res_g, TOTAL_THREADS*sizeof(int32_t), cudaMemcpyDeviceToHost) );

	float latency;
	latency = ((float)(stopClk[0]-startClk[0]))/((float)(REPEAT_TIMES*4));

  //gasp-gpu analysis----------------------------------------------------------------------------
  print_output(latency);

	return 0;
}
