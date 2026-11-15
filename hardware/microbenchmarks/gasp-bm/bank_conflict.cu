// Authored by Josh Reynolds

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

#define THREADS_PER_BLOCK 32
#define THREADS_PER_SM 32
#define REPEAT_TIMES 1024
#define THREADS_PER_WARP 32
#define NUM_WARPS 1
#define BLOCKS_NUM 1
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)

__global__ void bank_conflict_latency(uint64_t *start_clock_penalty, uint64_t *stop_clock_penalty, uint64_t *start_clock_no_penalty, uint64_t *stop_clock_no_penalty, uint32_t *res) {
  int tid = threadIdx.x; // only using 1 block

  // shared memory
  __shared__ uint32_t shared_mem[32][32]; // 2D shared memory
  // there are 32 banks in shared memory that are 4 bytes wide
  // so shared_mem[0][0], shared_mem[1][0], ... , shared_mem[31][0] all map to bank 0
  // => our conflict will occur by making each thread access shared_mem[theadIdx.x][0]

  uint32_t ptr_penalty = (uint32_t)__cvta_generic_to_shared(&shared_mem[tid][0]); // built-in cuda command: https://docs.nvidia.com/cuda/archive/11.4.1/cuda-c-programming-guide/index.html#__cvta_generic_to_shared
  uint32_t ptr_no_penalty = (uint32_t)__cvta_generic_to_shared(&shared_mem[0][tid]);
  // initialize shared_mem[tid][0] = ptr so that when we run 'ld.shared.u32 %0 [%0]', the address stored in shared_mem[tid][0] is accessed at bank 0
  // same for ptr_no_penalty, except this will measure the accesses with no bank conflicts
  shared_mem[tid][0] = ptr_penalty;

  // synchronize all threads
	asm volatile ("bar.sync 0;");

  uint64_t start_penalty = read_clock64(); // test 1 with penalty

  // run a lot of operations that create bank conflicts for threads
  for (int i = 0; i < REPEAT_TIMES; i++) {
    asm volatile ("{\t\n"
      "ld.shared.u32 %0, [%0];\n\t"
      "ld.shared.u32 %0, [%0];\n\t"
      "ld.shared.u32 %0, [%0];\n\t"
      "ld.shared.u32 %0, [%0];\n\t"
      "}" : "+r"(ptr_penalty)::"memory"
    );
  }

  // synchronize all threads
	asm volatile ("bar.sync 0;");

  uint64_t stop_penalty = read_clock64();

  shared_mem[0][tid] = ptr_no_penalty; // getting ready for test 2

  // synchronize all threads
	asm volatile ("bar.sync 0;");

  uint64_t start_no_penalty = read_clock64(); // test 2 no penalty

  for (int i = 0; i < REPEAT_TIMES; i++) {
    asm volatile ("{\t\n"
      "ld.shared.u32 %0, [%0];\n\t"
      "ld.shared.u32 %0, [%0];\n\t"
      "ld.shared.u32 %0, [%0];\n\t"
      "ld.shared.u32 %0, [%0];\n\t"
      "}" : "+r"(ptr_no_penalty)::"memory"
    );
  }

  // synchronize all threads
	asm volatile ("bar.sync 0;");

  uint64_t stop_no_penalty = read_clock64();

  start_clock_penalty[tid] = start_penalty;
  stop_clock_penalty[tid] = stop_penalty;
  start_clock_no_penalty[tid] = start_no_penalty;
  stop_clock_no_penalty[tid] = stop_no_penalty;
  res[tid] = ptr_penalty + ptr_no_penalty; // need to use these values so that PTX does not throw them out in optimization
}

int main(int argc, char **argv) {
  (void)argc;
  uint64_t *start_clock_penalty = (uint64_t*) malloc(TOTAL_THREADS*sizeof(uint64_t));
	uint64_t *stop_clock_penalty = (uint64_t*) malloc(TOTAL_THREADS*sizeof(uint64_t));
  uint64_t *start_clock_no_penalty = (uint64_t*) malloc(TOTAL_THREADS*sizeof(uint64_t));
  uint64_t *stop_clock_no_penalty = (uint64_t*) malloc(TOTAL_THREADS*sizeof(uint64_t));
  uint32_t *res = (uint32_t*) malloc(TOTAL_THREADS*sizeof(int32_t));

  uint64_t *start_clock_penalty_g;
  uint64_t *stop_clock_penalty_g;
  uint64_t *start_clock_no_penalty_g;
  uint64_t *stop_clock_no_penalty_g;
  uint32_t *res_g;

  CUDA_CHECK( cudaMalloc(&start_clock_penalty_g, TOTAL_THREADS*sizeof(uint64_t)) );
  CUDA_CHECK( cudaMalloc(&stop_clock_penalty_g, TOTAL_THREADS*sizeof(uint64_t)) );
  CUDA_CHECK( cudaMalloc(&start_clock_no_penalty_g, TOTAL_THREADS*sizeof(uint64_t)) );
  CUDA_CHECK( cudaMalloc(&stop_clock_no_penalty_g, TOTAL_THREADS*sizeof(uint64_t)) );
  CUDA_CHECK( cudaMalloc(&res_g, TOTAL_THREADS*sizeof(uint32_t)) );

  bank_conflict_latency<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(start_clock_penalty_g, stop_clock_penalty_g, start_clock_no_penalty_g, stop_clock_no_penalty_g, res_g);
  CUDA_CHECK( cudaPeekAtLastError() );

  CUDA_CHECK( cudaMemcpy(start_clock_penalty, start_clock_penalty_g, TOTAL_THREADS*sizeof(uint64_t), cudaMemcpyDeviceToHost) );
	CUDA_CHECK( cudaMemcpy(stop_clock_penalty, stop_clock_penalty_g, TOTAL_THREADS*sizeof(uint64_t), cudaMemcpyDeviceToHost) );
  CUDA_CHECK( cudaMemcpy(start_clock_no_penalty, start_clock_no_penalty_g, TOTAL_THREADS*sizeof(uint64_t), cudaMemcpyDeviceToHost) );
	CUDA_CHECK( cudaMemcpy(stop_clock_no_penalty, stop_clock_no_penalty_g, TOTAL_THREADS*sizeof(uint64_t), cudaMemcpyDeviceToHost) );
  CUDA_CHECK( cudaMemcpy(res, res_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );


  float latency_penalty = ((float)(stop_clock_penalty[0]-start_clock_penalty[0]))/((float)(REPEAT_TIMES*4));
  float latency_no_penalty = ((float)(stop_clock_no_penalty[0]-start_clock_no_penalty[0])/((float)(REPEAT_TIMES*4)));
  float latency = (latency_penalty - latency_no_penalty) / (float)(TOTAL_THREADS-1); // there will be a conflict for each thread except the first one, so we must divide by the number of threads - 1

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
	json += "  \"benchmark\": \"bank_conflict.cu\",\n";
	json += "  \"parameter\": \"bank_penalty_cycles\",\n";
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

	CUDA_CHECK(cudaFree(start_clock_penalty_g));
  CUDA_CHECK(cudaFree(stop_clock_penalty_g));
  CUDA_CHECK(cudaFree(start_clock_no_penalty_g));
  CUDA_CHECK(cudaFree(stop_clock_no_penalty_g));

	free(start_clock_penalty);
	free(stop_clock_penalty);
  free(start_clock_no_penalty);
  free(stop_clock_no_penalty);

	return 0;
}
