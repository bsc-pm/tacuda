/*
	This file is part of Task-Aware CUDA and is licensed under the terms contained in the COPYING and COPYING.LESSER files.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef STREAM_POOL_HPP
#define STREAM_POOL_HPP

#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <vector>

#include "TaskingModel.hpp"
#include "util/ErrorHandler.hpp"


namespace tacuda {

//! Class that manages the TACUDA streams
class StreamPool {
private:
	//! Array of streams
	static std::vector<cudaStream_t> _streams;

	//! Context of the streams
	static CUcontext _context;

public:
	//! \brief Initialize the pool of streams
	//!
	//! \param nstreams The number of streams to create
	static inline void initialize(size_t nstreams)
	{
		assert(nstreams > 0);

		CUresult eret = cuCtxGetCurrent(&_context);
		if (eret != CUDA_SUCCESS)
			ErrorHandler::fail("Failed in cuCtxGetCurrent: ", eret);

		_streams.resize(nstreams);
		for (size_t s = 0; s < nstreams; ++s) {
			cudaError_t eret2 = cudaStreamCreate(&_streams[s]);
			if (eret2 != cudaSuccess)
				ErrorHandler::fail("Failed in cudaStreamCreate: ", eret2);
		}
	}

	//! \brief Finalize the pool of streams
	static inline void finalize()
	{
		for (size_t s = 0; s < _streams.size(); ++s) {
			cudaError_t eret = cudaStreamDestroy(_streams[s]);
			if (eret != cudaSuccess)
				ErrorHandler::fail("Failed in cudaStreamDestroy: ", eret);
		}
	}

	//! \brief Get stream within pool
	//!
	//! \param streamId The stream identifier within the pool
	static inline cudaStream_t getStream(size_t streamId)
	{
		assert(streamId < _streams.size());

		CUresult eret = cuCtxSetCurrent(_context);
		if (eret != CUDA_SUCCESS)
			ErrorHandler::fail("Failed in cuCtxSetCurrent: ", eret);

		return _streams[streamId];
	}
};

} // namespace tacuda

#endif // STREAM_POOL_HPP
