/*
	This file is part of Task-Aware CUDA and is licensed under the terms contained in the COPYING and COPYING.LESSER files.

	Copyright (C) 2021-2025 Barcelona Supercomputing Center (BSC)
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
	static std::vector<std::vector<cudaStream_t>> _streams;

	//! Array of Stream Selectors
	static std::vector<int> _stream_selectors;

	//! Context of the streams
	static CUcontext _context;

public:
	//! \brief Initialize the pool of streams
	//!
	//! \param nstreams The number of streams to create
	static inline void initialize(size_t nstreams)
	{
		assert(nstreams > 0);

		const size_t totalStreams = nstreams * TaskingModel::getNumCPUs();

		CUresult eret = cuCtxGetCurrent(&_context);
		if (eret != CUDA_SUCCESS)
			ErrorHandler::fail("Failed in cuCtxGetCurrent: ", eret);

		_streams.resize(totalStreams);
		_stream_selectors.resize(totalStreams);

		for (size_t s = 0; s < totalStreams; ++s) {
			_streams[s].resize(nstreams);
			_stream_selectors[s] = 0;

			for (size_t ss = 0; ss < nstreams; ++ss) {
				cudaError_t eret2 = cudaStreamCreate(&_streams[s][ss]);
				if (eret2 != cudaSuccess)
					ErrorHandler::fail("Failed in cudaStreamCreate: ", eret2);
			}
		}
	}

	//! \brief Finalize the pool of streams
	static inline void finalize()
	{
		for (size_t s = 0; s < _streams.size(); ++s) {
			for (size_t ss = 0; s < _streams[s].size(); ++ss) {
				cudaError_t eret = cudaStreamDestroy(_streams[s][ss]);
				if (eret != cudaSuccess)
					ErrorHandler::fail("Failed in cudaStreamDestroy: ", eret);
			}
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

		int stream = ((_stream_selectors[streamId]++) % _streams[streamId].size());
		return _streams[streamId][stream];
	}
};

} // namespace tacuda

#endif // STREAM_POOL_HPP
