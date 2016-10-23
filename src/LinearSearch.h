/*
 * LinearSearch.h
 *
 *  Created on: Oct 9, 2016
 *      Author: mrp
 */

#ifndef LINEARSEARCH_H_
#define LINEARSEARCH_H_

#include <inttypes.h>

//#define DEBUG_LOAD_RULES
//#define DEBUG_LOAD_TUPLES
//#define DEBUG_SEARCH_RULES
//#define SHOW_SEARCH_RESULTS

#define CLOCK CLOCK_REALTIME

#define NUM_RULES	10000
#define NUM_TUPLES	1024
#define NUM_FIELDS	15
#define NUM_RESULTS	NUM_TUPLES
#define LOOKUP_MISS	-1

typedef uint32_t tuple_t[NUM_FIELDS];
typedef uint32_t mask_t[NUM_FIELDS];

typedef struct rule {
	tuple_t tuple;
	mask_t mask;
	int priority;
	void *action;
} rule_t;

static rule_t rules[NUM_RULES];
static tuple_t tuples[NUM_TUPLES];
static int results[NUM_RESULTS];

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
