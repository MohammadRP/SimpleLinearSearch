/*
 * LinearSearch_Common.h
 *
 *  Created on: Oct 25, 2016
 *      Author: mrp
 */

#ifndef LINEARSEARCH_COMMON_H_
#define LINEARSEARCH_COMMON_H_

#include <inttypes.h>

#define NUM_FIELDS			15
#define NUM_TUPLES			1020
#define NUM_RULES			10000
#define NUM_RESULTS			NUM_TUPLES

#define THREADS_PER_BLOCK	30
#define BLOCKS_PER_GRID		512
#define RULES_PER_BLOCK		32

#define BATCH_SIZE			THREADS_PER_BLOCK
#define NUM_BATCH			(NUM_TUPLES / BATCH_SIZE)
#define NUM_ITER			NUM_BATCH

#define LOOKUP_MISS			-1

//#define DEBUG_LOAD_RULES
//#define DEBUG_LOAD_TUPLES
//#define DEBUG_SEARCH_RULES
//#define SHOW_SEARCH_RESULTS

#define CLOCK CLOCK_REALTIME

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

#endif /* LINEARSEARCH_COMMON_H_ */
