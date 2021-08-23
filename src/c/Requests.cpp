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
tacudaWaitRequestAsync(tacudaRequest *request)
{
	assert(request != nullptr);

	if (*request != TACUDA_REQUEST_NULL)
		RequestManager::processRequest((Request *) *request);

	*request = TACUDA_REQUEST_NULL;

	return cudaSuccess;
}

cudaError_t
tacudaWaitallRequestsAsync(size_t count, tacudaRequest requests[])
{
	if (count == 0)
		return cudaSuccess;

	assert(requests != nullptr);

	RequestManager::processRequests(count, (Request * const *) requests);

	for (size_t r = 0; r < count; ++r) {
		requests[r] = TACUDA_REQUEST_NULL;
	}

	return cudaSuccess;
}

} // extern C

#pragma GCC visibility pop
