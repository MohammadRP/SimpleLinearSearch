/*
 * LinearSearch_GPU.h
 *
 *  Created on: Oct 22, 2016
 *      Author: mrp
 */

#ifndef LINEARSEARCH_GPU_H_
#define LINEARSEARCH_GPU_H_

#include "LinearSearch.h"

#define BATCH_SIZE			32
#define NUM_ITER			(NUM_TUPLES / BATCH_SIZE)

#define THREADS_PER_BLOCK		32
#define BLOCKS_PER_GRID			16
#define RULES_PER_BLOCK			32

rule_t *rules_dev;
tuple_t *tuples_dev;
int *results_dev, *results_host;

void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__global__ void LinearSearch_Kernel(rule_t *rules, tuple_t *tuples,
		int *results) {

	__shared__ rule_t block_rules[RULES_PER_BLOCK];
	tuple_t thread_tuple;
	int rIdx, rPri;

	int r, t, f;
	uint8_t match;
	int rule_offset = blockIdx.x * RULES_PER_BLOCK;

	while (rule_offset < NUM_RULES) {

		if (threadIdx.x == 0) {
			memcpy(block_rules, &rules[rule_offset],
			RULES_PER_BLOCK * sizeof(rule_t));
		}
		__syncthreads();

		t = threadIdx.x;

		while (t < BATCH_SIZE) {
			memcpy(&thread_tuple, &tuples[t], sizeof(tuple_t));
			rIdx = -1;
			rPri = -1;
			for (r = 0; r < RULES_PER_BLOCK && (r + rule_offset) < NUM_RULES;
					r++) {
				match = 1;
				for (f = 0; f < NUM_FIELDS; f++) {
					if ((thread_tuple[f] & block_rules[r].mask[f])
							!= block_rules[r].tuple[f]) {
						match = 0;
						break;
					}
				}
				if (match == 1) {
					if (rPri < block_rules[r].priority) {
						rPri = block_rules[r].priority;
						rIdx = r + rule_offset;
					}
				}
			}
			if (rIdx >= 0) {
				atomicMax(&results[t], (rPri << 16) | rIdx);
			}
			t += blockDim.x;
		}

		__syncthreads();
		rule_offset += (gridDim.x * RULES_PER_BLOCK);
	}

}

int search_gpu(void) {

	printf("\n\nSearching %d Tuple(s) in %d Rule(s) on GPU ...\n", NUM_TUPLES,
	NUM_RULES);
	sleep(1);

	int i;

	cudaEvent_t start, stop;
	cudaStream_t stream;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaStreamCreate(&stream);

	// allocate memory on host
	CUDA_CHECK_RETURN(
			cudaMallocHost((void **) &results_host, NUM_RESULTS * sizeof(int), cudaHostAllocDefault));
	for (i = 0; i < NUM_RESULTS; i++) {
		results_host[i] = LOOKUP_MISS;
	}

	// allocate memory on device
	CUDA_CHECK_RETURN(
			cudaMalloc((void **)&rules_dev, NUM_RULES * sizeof(rule_t)));
	CUDA_CHECK_RETURN(
			cudaMalloc((void **)&tuples_dev, BATCH_SIZE * sizeof(tuple_t)));
	CUDA_CHECK_RETURN(
			cudaMalloc((void **)&results_dev, BATCH_SIZE * sizeof(int)));

	CUDA_CHECK_RETURN(
			cudaMemcpy(results_dev, results_host, BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));

	// copy rules to device
	CUDA_CHECK_RETURN(
			cudaMemcpy(rules_dev, rules, NUM_RULES * sizeof(rule_t), cudaMemcpyHostToDevice));

	sleep(1);

	// set time stamp t1
	cudaEventRecord(start, stream);

	for (i = 0; i < NUM_ITER; i++) {

		// copy tuples to device
		CUDA_CHECK_RETURN(
				cudaMemcpyAsync(tuples_dev, &tuples[i * BATCH_SIZE], BATCH_SIZE * sizeof(tuple_t), cudaMemcpyHostToDevice, stream));

		// Launch Kernel Function
		LinearSearch_Kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, 0, stream>>>(
				rules_dev, tuples_dev, results_dev);

		// copy results to host
		CUDA_CHECK_RETURN(
				cudaMemcpyAsync(&results_host[i*BATCH_SIZE], results_dev, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream));
	}

	cudaStreamSynchronize(stream);

	// set time stamp t2
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

#ifdef SHOW_SEARCH_RESULTS
	for (i = 0; i < NUM_RESULTS; i++)
		if (results_host[i] >= 0)
			printf("tuple %d matched with rule %d\n", i,
					results_host[i] & 0x0000FFFF);
#endif

	printf("Done. Linear Search takes %f ms on GPU, per Batch = %f ms\n",
			elapsedTime, elapsedTime / NUM_ITER);

	// free host memory
	CUDA_CHECK_RETURN(cudaFreeHost(results_host));

	// free device memory
	CUDA_CHECK_RETURN(cudaFree(rules_dev));
	CUDA_CHECK_RETURN(cudaFree(tuples_dev));
	CUDA_CHECK_RETURN(cudaFree(results_dev));

	CUDA_CHECK_RETURN(cudaStreamDestroy(stream));

	return 0;

}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
void CheckCudaErrorAux(const char *file, unsigned line, const char *statement,
		cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

#endif /* LINEARSEARCH_GPU_H_ */
