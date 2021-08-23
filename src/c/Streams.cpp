/*
	This file is part of Task-Aware CUDA and is licensed under the terms contained in the COPYING and COPYING.LESSER files.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#include <cuda_runtime.h>
#include <TACUDA.h>

#include "common/StreamPool.hpp"
#include "common/Environment.hpp"
#include "common/TaskingModel.hpp"
#include "common/util/ErrorHandler.hpp"

using namespace tacuda;

#pragma GCC visibility push(default)

extern "C" {

cudaError_t
tacudaCreateStreams(size_t count)
{
	if (count == TACUDA_STREAMS_AUTO)
		count = TaskingModel::getNumCPUs();
	assert(count > 0);

	StreamPool::initialize(count);

	return cudaSuccess;
}

cudaError_t
tacudaDestroyStreams()
{
	StreamPool::finalize();

	return cudaSuccess;
}

cudaError_t
tacudaGetStream(cudaStream_t *stream)
{
	assert(stream != nullptr);

	*stream = StreamPool::getStream(TaskingModel::getCurrentCPU());

	return cudaSuccess;
}

cudaError_t
tacudaReturnStream(cudaStream_t)
{
	return cudaSuccess;
}

cudaError_t
tacudaSynchronizeStreamAsync(cudaStream_t stream)
{
	RequestManager::generateRequest(stream, true);

	return cudaSuccess;
}

} // extern C

#pragma GCC visibility pop
