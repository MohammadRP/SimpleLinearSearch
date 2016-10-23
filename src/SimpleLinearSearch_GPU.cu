/*
 ============================================================================
 Name        : SimpleLinearSearch_GPU.cu
 Author      : mrp
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>

#include "LinearSearch.h"
#include "LinearSearch_CPU.h"
#include "LinearSearch_GPU.h"

int main(void) {

	init_rules();

	init_tuples();

	init_results();

	search_cpu();

	search_gpu();

	return 0;
}
