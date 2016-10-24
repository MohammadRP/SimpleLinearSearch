/*
 * LinearSearch_CPU.h
 *
 *  Created on: Oct 22, 2016
 *      Author: mrp
 */

#ifndef LINEARSEARCH_CPU_H_
#define LINEARSEARCH_CPU_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>
#include <memory.h>
#include <time.h>

#include "LinearSearch_Common.h"

int search_cpu(void) {
	printf("\n\nSearching %d Tuple(s) in %d Rule(s) on CPU ...\n", NUM_TUPLES,
	NUM_RULES);
	sleep(1);

	struct timespec t1, t2;
	uint8_t match;
	int r, t, f;

	clock_gettime(CLOCK, &t1);
	for (t = 0; t < NUM_TUPLES; t++) {
		for (r = 0; r < NUM_RULES; r++) {
			match = 1;
			for (f = 0; f < NUM_FIELDS; f++) {
				if ((tuples[t][f] & rules[r].mask[f]) != rules[r].tuple[f]) {
					match = 0;
					break;
				}
			}
			if (match == 1) {
				if (results[t] == LOOKUP_MISS) {
					results[t] = r;
				} else if (rules[results[t]].priority <= rules[r].priority) {
					results[t] = r;
				}
			}
		}
	}
	clock_gettime(CLOCK, &t2);

#ifdef SHOW_SEARCH_RESULTS
	for (i = 0; i < NUM_RESULTS; i++) {
		if (results[i] >= 0)
		printf("tuple %d matched with rule %d\n", i, results[i]);
	}
#endif

	long time_spent_us = (t2.tv_sec - t1.tv_sec) * 1e6
			+ (t2.tv_nsec - t1.tv_nsec) / 1e3;
	long time_spent_per_tuple_us = time_spent_us / NUM_TUPLES;

	printf("Done. Linear Search takes %ld us on CPU, time per tuple: %ld us\n",
			time_spent_us, time_spent_per_tuple_us);

	return 0;
}

#endif /* LINEARSEARCH_CPU_H_ */
