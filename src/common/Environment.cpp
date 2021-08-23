/*
	This file is part of Task-Aware CUDA and is licensed under the terms contained in the COPYING and COPYING.LESSER files.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#include <cstdio>

#include "Environment.hpp"
#include "StreamPool.hpp"
#include "util/ErrorHandler.hpp"


namespace tacuda {

TaskingModel::polling_handle_t Environment::_pollingHandle;

EnvironmentVariable<uint64_t> Environment::_pollingFrequency("TACUDA_POLLING_FREQUENCY", 500);

SpinLock ErrorHandler::_lock;

RequestManager::add_queue_t RequestManager::_addQueue;
SpinLock RequestManager::_addQueueLock;
RequestManager::list_t RequestManager::_pendingRequests;

std::vector<cudaStream_t> StreamPool::_streams;
CUcontext StreamPool::_context;

} // namespace tacuda

#if !defined(NDEBUG)
namespace boost {
	void assertion_failed_msg(char const * expr, char const * msg, char const * function, char const * file, long line)
	{
		fprintf(stderr, "%s:%ld %s Boost assertion failure: %s when evaluating %s\n", file, line, function, msg, expr);
		abort();
	}

	void assertion_failed(char const * expr, char const * function, char const * file, long line)
	{
		fprintf(stderr, "%s:%ld %s Boost assertion failure when evaluating %s\n", file, line, function, expr);
		abort();
	}
}
#endif
