/*
	This file is part of Task-Aware CUDA and is licensed under the terms contained in the COPYING and COPYING.LESSER files.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#include <cuda.h>
#include <TACUDA.h>

#include "common/Environment.hpp"

using namespace tacuda;

#pragma GCC visibility push(default)

extern "C" {

CUresult
tacudaInit(unsigned int flags)
{
	CUresult eret = cuInit(flags);
	if (eret == CUDA_SUCCESS) {
		Environment::initialize();
	}
	return eret;
}

CUresult
tacudaFinalize()
{
	Environment::finalize();
	return CUDA_SUCCESS;
}

} // extern C

#pragma GCC visibility pop
