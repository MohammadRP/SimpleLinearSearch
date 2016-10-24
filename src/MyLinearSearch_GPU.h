/*
 * MyLinearSearch_GPU.h
 *
 *  Created on: Oct 23, 2016
 *      Author: mrp
 */

#ifndef MYLINEARSEARCH_GPU_H_
#define MYLINEARSEARCH_GPU_H_

#include <inttypes.h>

#include "LinearSearch.h"
#include "LinearSearch_Common.h"

typedef struct myRule {
	uint32_t tuple[NUM_FIELDS][NUM_RULES];
	uint32_t mask[NUM_FIELDS][NUM_RULES];
	int priority[NUM_RULES];
	void *action[NUM_RULES];
} myRule_t;

myRule_t *myRules_host;
myRule_t *myRules_dev;

typedef struct myHostTuples {
	uint32_t value[NUM_FIELDS][BATCH_SIZE];
} myTuples_t;

myTuples_t *myTuples_host[NUM_BATCH];
myTuples_t *myTuples_dev;

__global__ void MyLinearSearch_Kernel(myRule_t *rules, myTuples_t *tuples,
		int *results) {

	__shared__ uint8_t matches[BATCH_SIZE][RULES_PER_BLOCK];
	__shared__ int block_pri[RULES_PER_BLOCK];
	uint32_t thread_rules_tuple[RULES_PER_BLOCK];
	uint32_t thread_rules_mask[RULES_PER_BLOCK];

	uint32_t thread_fields[NUM_FIELDS];

	int rule_offset = blockIdx.x * RULES_PER_BLOCK;
	int f = threadIdx.x % NUM_FIELDS;
	int t, t1, r;
	int rIdx, rPri;

	int i;

	while (rule_offset < NUM_RULES) {

		for (i = 0; i < RULES_PER_BLOCK; i++) {
			thread_rules_tuple[i] = rules->tuple[f][rule_offset + i];
			thread_rules_mask[i] = rules->mask[f][rule_offset + i];
		}
		r = threadIdx.x;
		while (r < RULES_PER_BLOCK) {
			block_pri[r] = rules->priority[rule_offset + r];
			r += blockDim.x;
		}

		t = threadIdx.x;
		while (t < BATCH_SIZE) {
			for (r = 0; r < RULES_PER_BLOCK; r++)
				matches[t][r] = 1;
			t += blockDim.x;
		}
		__syncthreads();

		t1 = (threadIdx.x / NUM_FIELDS) * NUM_FIELDS;
		while (t1 < BATCH_SIZE) {
			memcpy(&thread_fields, &tuples->value[f][t1],
			NUM_FIELDS * sizeof(uint32_t));

			for (r = 0; r < RULES_PER_BLOCK && (r + rule_offset) < NUM_RULES;
					r++) {
				for (t = 0; t < NUM_FIELDS; t++) {
					if ((thread_fields[t] & thread_rules_mask[r])
							!= thread_rules_tuple[r]) {
						matches[t + t1][r] = 0;
					}
				}
			}
			t1 += (blockDim.x / NUM_FIELDS) * NUM_FIELDS;
		}
		__syncthreads();

// check matches
		t = threadIdx.x;
		while (t < BATCH_SIZE) {
			rIdx = -1;
			rPri = -1;
			for (r = 0; r < RULES_PER_BLOCK && (r + rule_offset) < NUM_RULES;
					r++) {
				if (matches[t][r] == 1) {
					if (rPri < block_pri[r]) {
						rPri = block_pri[r];
						rIdx = r + rule_offset;
					}
				}
			}
			if (rIdx > 0)
				atomicMax(&results[t], (rPri << 16) | rIdx);
			t += blockDim.x;
		}
		__syncthreads();

		rule_offset += (gridDim.x * RULES_PER_BLOCK);
	}

}

int mySearch_gpu(void) {

	printf("\n\nSearching %d Tuple(s) in %d Rule(s) on myGPU ...\n", NUM_TUPLES,
	NUM_RULES);
	sleep(1);

	int i, j, k;

	cudaEvent_t start, stop;
	cudaStream_t stream;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaStreamCreate(&stream);

// allocate memory on host
	CUDA_CHECK_RETURN(
			cudaMallocHost((void ** )&myRules_host, sizeof(myRule_t),
					cudaHostAllocDefault));
	for (i = 0; i < NUM_RULES; i++) {
		for (j = 0; j < NUM_FIELDS; j++) {
			myRules_host->tuple[j][i] = rules[i].tuple[j];
			myRules_host->mask[j][i] = rules[i].mask[j];
		}
		myRules_host->priority[i] = rules[i].priority;
	}

	for (i = 0; i < NUM_BATCH; i++)
		CUDA_CHECK_RETURN(
				cudaMallocHost((void ** ) &myTuples_host[i],
						NUM_BATCH * sizeof(myTuples_t), cudaHostAllocDefault));

	for (i = 0; i < NUM_BATCH; i++) {
		for (j = 0; j < BATCH_SIZE; j++) {
			for (k = 0; k < NUM_FIELDS; k++) {
				myTuples_host[i]->value[k][j] = tuples[j + i * BATCH_SIZE][k];
			}
		}
	}

	CUDA_CHECK_RETURN(
			cudaMallocHost((void **) &results_host, NUM_RESULTS * sizeof(int), cudaHostAllocDefault));
	for (i = 0; i < NUM_RESULTS; i++) {
		results_host[i] = LOOKUP_MISS;
	}

// allocate memory on device
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&myRules_dev, sizeof(myRule_t)));

	CUDA_CHECK_RETURN(cudaMalloc((void ** )&myTuples_dev, sizeof(myTuples_t)));

	CUDA_CHECK_RETURN(
			cudaMalloc((void **)&results_dev, BATCH_SIZE * sizeof(int)));

	CUDA_CHECK_RETURN(
			cudaMemcpy(results_dev, results_host, BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));

// copy rules to device
	CUDA_CHECK_RETURN(
			cudaMemcpy(myRules_dev, myRules_host, sizeof(myRule_t),
					cudaMemcpyHostToDevice));

	sleep(1);

// set time stamp t1
	cudaEventRecord(start, stream);

	for (i = 0; i < NUM_BATCH; i++) {

// copy tuples to device
		CUDA_CHECK_RETURN(
				cudaMemcpyAsync(myTuples_dev, myTuples_host[i],
						sizeof(myTuples_t), cudaMemcpyHostToDevice, stream));

// Launch Kernel Function
		MyLinearSearch_Kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, 0, stream>>>(
				myRules_dev, myTuples_dev, results_dev);

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
	for (i = 0; i < NUM_BATCH; i++)
		cudaFreeHost(myTuples_host[i]);

// free device memory
	CUDA_CHECK_RETURN(cudaFree(myRules_dev));
	CUDA_CHECK_RETURN(cudaFree(myTuples_dev));
	CUDA_CHECK_RETURN(cudaFree(results_dev));

	CUDA_CHECK_RETURN(cudaStreamDestroy(stream));

	return 0;

}

#endif /* MYLINEARSEARCH_GPU_H_ */
