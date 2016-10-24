/*
 * LinearSearch.h
 *
 *  Created on: Oct 9, 2016
 *      Author: mrp
 */

#ifndef LINEARSEARCH_H_
#define LINEARSEARCH_H_

#include "LinearSearch_Common.h"
#include <inttypes.h>


void init_rules(void) {
	int i, j;
	for (i = 0; i < NUM_RULES; i++) {
		for (j = 0; j < NUM_FIELDS; j++) {
			rules[i].tuple[j] = i;
			rules[i].mask[j] = i;
		}
		rules[i].priority = i;
	}
}

void init_tuples(void) {
	int i, j;
	for (i = 0; i < NUM_TUPLES; i++) {
		for (j = 0; j < NUM_FIELDS; j++) {
			tuples[i][j] = i;
		}
	}
}

void init_results(void) {
	int i;
	for (i = 0; i < NUM_RESULTS; i++) {
		results[i] = LOOKUP_MISS;
	}
}

#endif /* LINEARSEARCH_H_ */
