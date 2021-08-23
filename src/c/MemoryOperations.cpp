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
tacudaMemcpyAsync(
	void *dst, const void *src, size_t sizeBytes,
	enum cudaMemcpyKind kind, cudaStream_t stream,
	tacudaRequest *requestPtr)
{
	cudaError_t eret;
	eret = cudaMemcpyAsync(dst, src, sizeBytes, kind, stream);
	if (eret != cudaSuccess)
		return eret;

	Request *request = RequestManager::generateRequest(stream, (requestPtr == nullptr));
	assert(request != nullptr);

	if (requestPtr != nullptr)
		*requestPtr = request;

	return cudaSuccess;
}

cudaError_t
tacudaMemsetAsync(
	void *devPtr, int value, size_t sizeBytes, cudaStream_t stream,
	tacudaRequest *requestPtr)
{
	cudaError_t eret;
	eret = cudaMemsetAsync(devPtr, value, sizeBytes, stream);
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
