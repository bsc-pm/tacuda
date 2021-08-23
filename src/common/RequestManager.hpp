/*
	This file is part of Task-Aware CUDA and is licensed under the terms contained in the COPYING and COPYING.LESSER files.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef REQUEST_MANAGER_HPP
#define REQUEST_MANAGER_HPP

#include <cuda_runtime.h>

#include <boost/intrusive/list.hpp>
#include <boost/lockfree/spsc_queue.hpp>

#include <cassert>

#include "util/ErrorHandler.hpp"


namespace tacuda {

//! Struct that represents a TACUDA request
struct Request {
	typedef boost::intrusive::link_mode<boost::intrusive::normal_link> link_mode_t;
	typedef boost::intrusive::list_member_hook<link_mode_t> links_t;

	//! The event of the request
    cudaEvent_t _event;

	//! The stream where the operation was posted
    cudaStream_t _stream;

	//! The event counter of the calling task
	void *_eventCounter;

	//! Support for Boost intrusive lists
	links_t _listLinks;

	inline Request() :
		_eventCounter(nullptr)
	{
	}
};


//! Class that manages the TACUDA requests
class RequestManager {
private:
	typedef boost::lockfree::spsc_queue<Request*, boost::lockfree::capacity<63*1024> > add_queue_t;
	typedef boost::intrusive::member_hook<Request, typename Request::links_t, &Request::_listLinks> hook_option_t;
	typedef boost::intrusive::list<Request, hook_option_t> list_t;

	//! Fast queues to add requests concurrently
	static add_queue_t _addQueue;

	//! Lock to add requests
	static SpinLock _addQueueLock;

	//! List of pending requests
	static list_t _pendingRequests;

	//! \brief Add a TACUDA request
	//!
	//! \param request The request to add
	static void addRequest(Request *request)
	{
		_addQueueLock.lock();
		while (!_addQueue.push(request)) {
			util::spinWait();
		}
		_addQueueLock.unlock();
	}

	//! \brief Add multiple TACUDA requests
	//!
	//! \param requests The requests to add
	static void addRequests(size_t count, Request *const requests[])
	{
		size_t added = 0;
		_addQueueLock.lock();
		do {
			added += _addQueue.push(requests+added, count-added);
		} while (added < count);
		_addQueueLock.unlock();
	}

public:
	//! \brief Generate a TACUDA request waiting for stream completion
	//!
	//! \param stream The stream to wait
	//! \param bind Whether should be bound to the current task
	static Request *generateRequest(cudaStream_t stream, bool bind)
	{
		Request *request = Allocator<Request>::allocate();
		assert(request != nullptr);

		cudaError_t eret;
		eret = cudaEventCreate(&request->_event); 
		if (eret != cudaSuccess) {
			ErrorHandler::fail("Failed in cudaEventCreate: ", eret);
		}
        
		eret = cudaEventRecord(request->_event, stream);
		if (eret != cudaSuccess) {
			ErrorHandler::fail("Failed in cudaEventRecord: ", eret);
		}

		// Bind the request to the calling task if needed
		if (bind) {
			void *counter = TaskingModel::getCurrentEventCounter();
			assert(counter != nullptr);

			request->_eventCounter = counter;

			TaskingModel::increaseCurrentTaskEventCounter(counter, 1);

			RequestManager::addRequest(request);
		}
		return request;
	}

	static void processRequest(Request *request)
	{
		assert(request != nullptr);
		assert(request->_eventCounter == nullptr);

		void *counter = TaskingModel::getCurrentEventCounter();
		assert(counter != nullptr);

		request->_eventCounter = counter;

		TaskingModel::increaseCurrentTaskEventCounter(counter, 1);

		addRequest(request);
	}

	static void processRequests(size_t count, Request *const requests[])
	{
		assert(count > 0);
		assert(requests != nullptr);

		void *counter = TaskingModel::getCurrentEventCounter();
		assert(counter != nullptr);

		size_t nactive = 0;
		for (size_t r = 0; r < count; ++r) {
			if (requests[r] != nullptr) {
				assert(requests[r]->_eventCounter == nullptr);
				requests[r]->_eventCounter = counter;
				++nactive;
			}
		}
		TaskingModel::increaseCurrentTaskEventCounter(counter, nactive);

		addRequests(count, requests);
	}

	static void checkRequests()
	{
		if (!_addQueue.empty()) {
			_addQueue.consume_all(
				[&](Request *request) {
					if (request != nullptr)
						_pendingRequests.push_back(*request);
				}
			);
		}
    
		cudaError_t eret;

		auto it = _pendingRequests.begin();
		while (it != _pendingRequests.end()) {
			Request &request = *it;
                
			eret = cudaEventQuery(request._event);
			if (eret != cudaSuccess && eret != cudaErrorNotReady) {
				ErrorHandler::fail("Failed in cudaEventQuery: ", eret);
			} else if (eret == cudaSuccess) {
                assert(request._eventCounter != nullptr);
				TaskingModel::decreaseTaskEventCounter(request._eventCounter, 1);

				eret = cudaEventDestroy(request._event);
				if (eret != cudaSuccess) {
					ErrorHandler::fail("Failed in cudaEventDestroy: ", eret);
				}

				Allocator<Request>::free(&request);

				it = _pendingRequests.erase(it);
				continue;
            }
			++it;
		}
	}
};

} // namespace tacuda

#endif // REQUEST_MANAGER_HPP
