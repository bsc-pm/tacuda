/*
	This file is part of Task-Aware CUDA and is licensed under the terms contained in the COPYING and COPYING.LESSER files.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#include <cuda_runtime.h>
#include <TACUDA.h>

#include "common/Environment.hpp"
#include "common/TaskingModel.hpp"
#include "common/util/ErrorHandler.hpp"

using namespace tacuda;

#pragma GCC visibility push(default)

extern "C" {

cudaError_t
tacudaLaunchKernel(
    const void* func, dim3 gridDim, dim3 blockDim, 
    void** args, size_t sharedMem, cudaStream_t stream,
    tacudaRequest *requestPtr)
{

	cudaError_t eret;
    
	eret = cudaLaunchKernel(func, gridDim, blockDim, args,
		sharedMem, stream);
	if (eret != cudaSuccess)
		return eret;

	Request *request = RequestManager::generateRequest(stream, (requestPtr == nullptr));
	assert(request != nullptr);

	if (requestPtr != nullptr)
		*requestPtr = request;

	return cudaSuccess;
}

} // extern C

#pragma GCC visibility pop
