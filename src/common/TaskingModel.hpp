/*
	This file is part of Task-Aware CUDA and is licensed under the terms contained in the COPYING and COPYING.LESSER files.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKING_MODEL_HPP
#define TASKING_MODEL_HPP

#include <pthread.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

#include "Symbol.hpp"
#include "TaskingModelAPI.hpp"
#include "util/EnvironmentVariable.hpp"
#include "util/ErrorHandler.hpp"


namespace tacuda {

//! Class that gives access to the tasking model features
class TaskingModel {
public:
	//! Prototype of a polling instance function
	typedef void (*polling_function_t)(void *args);

	//! Handle of polling instances
	typedef size_t polling_handle_t;

private:
	//! Pointers to the tasking model functions
	static get_current_event_counter_t *_getCurrentEventCounter;
	static increase_current_task_event_counter_t *_increaseCurrentTaskEventCounter;
	static decrease_task_event_counter_t *_decreaseTaskEventCounter;
	static notify_task_event_counter_api_t *_notifyTaskEventCounterAPI;
	static spawn_function_t *_spawnFunction;
	static wait_for_t *_waitFor;
	static get_total_num_cpus_t *_getTotalNumCPUs;
	static get_current_virtual_cpu_t *_getCurrentVirtualCPU;

	//! Actual names of the tasking model functions
	static const std::string _getCurrentEventCounterName;
	static const std::string _increaseCurrentTaskEventCounterName;
	static const std::string _decreaseTaskEventCounterName;
	static const std::string _notifyTaskEventCounterAPIName;
	static const std::string _spawnFunctionName;
	static const std::string _waitForName;
	static const std::string _getTotalNumCPUsName;
	static const std::string _getCurrentVirtualCPUName;

	//! Internal structure to represent a polling instance
	struct PollingInfo {
		std::string _name;
		polling_function_t _function;
		void *_args;
		uint64_t _frequency;
		std::atomic<bool> _mustFinish;
		std::atomic<bool> _finished;

		inline PollingInfo(
			const std::string &name,
			polling_function_t function,
			void *args,
			uint64_t frequency
		) :
			_name(name),
			_function(function),
			_args(args),
			_frequency(frequency),
			_mustFinish(false),
			_finished(false)
		{
		}
	};

public:
	//! \brief Initialize and load the symbols of the tasking model
	static void initialize();

	//! \brief Register a polling instance
	//!
	//! This function registers a polling instance, which will call
	//! the specified function with its argument periodically. The
	//! polling function must accept a pointer to void and should
	//! not return anything
	//!
	//! \param name The name of the polling instance
	//! \param function The function to be called periodically
	//! \param args The arguments of the function
	//! \param frequency The frequency at which to call the function
	//!                  in microseconds
	//!
	//! \returns A polling handle to unregister the instance once
	//!          the polling should finish
	static inline polling_handle_t registerPolling(
		const std::string &name,
		polling_function_t function,
		void *args,
		uint64_t frequency
	) {
		PollingInfo *info = new PollingInfo(name, function, args, frequency);
		assert(info != nullptr);

		assert(_spawnFunction);
		(*_spawnFunction)(pollingTaskFunction, info, pollingTaskCompletes, info, info->_name.c_str());

		return (polling_handle_t) info;
	}

	//! \brief Unregister a polling instance
	//!
	//! This function unregisters a polling instance, which will
	//! prevent the associated polling function from being further
	//! called. The polling function is guaranteed to not be called
	//! after returing from this function. Note that other registered
	//! polling instances may continue calling that function
	//!
	//! \param handle The handle of the polling instance to unregister
	static inline void unregisterPolling(polling_handle_t handle)
	{
		PollingInfo *info = (PollingInfo *) handle;
		assert(info != nullptr);
		assert(_waitFor);

		// Notify that the polling should stop
		info->_mustFinish = true;

		// Wait until the spawned task completes
		while (!info->_finished) {
			// Yield in case there is a single available CPU
			(*_waitFor)(1000);
		}
		delete info;
	}

	//! \brief Get the event counter of the current task
	//!
	//! \returns An opaque pointer of the event counter
	static inline void *getCurrentEventCounter()
	{
		assert(_getCurrentEventCounter);
		return (*_getCurrentEventCounter)();
	}

	//! \brief Increase the event counter of the current task
	//!
	//! \param counter The event counter of the current task
	//! \param increment The amount of events to increase
	static inline void increaseCurrentTaskEventCounter(void *counter, unsigned int increment)
	{
		assert(_increaseCurrentTaskEventCounter);
		(*_increaseCurrentTaskEventCounter)(counter, increment);
	}

	//! \brief Decrease the event counter of a task
	//!
	//! \param counter The event counter of the target task
	//! \param decrement The amount of events to decrease
	static inline void decreaseTaskEventCounter(void *counter, unsigned int decrement)
	{
		assert(_decreaseTaskEventCounter);
		(*_decreaseTaskEventCounter)(counter, decrement);
	}

	//! \brief Get total number of CPUs used by the tasking runtime
	static inline size_t getNumCPUs()
	{
		assert(_getTotalNumCPUs);
		return (size_t) (*_getTotalNumCPUs)();
	}

	//! \brief Get the current CPU identifier
	static inline size_t getCurrentCPU()
	{
		assert(_getCurrentVirtualCPU);
		return (size_t) (*_getCurrentVirtualCPU)();
	}

private:
	//! \brief Function called by all polling tasks
	//!
	//! This is called only once per polling task and
	//! represents the body of any polling tasks
	//!
	//! \param args An opaque pointer to the polling info
	static inline void pollingTaskFunction(void *args)
	{
		PollingInfo *info = (PollingInfo *) args;
		assert(info != nullptr);
		assert(_waitFor);

		// Poll until it is externally notified to stop
		while (!info->_mustFinish) {
			// Call the actual polling function
			info->_function(info->_args);

			// Pause the polling task for some microseconds
			(*_waitFor)(info->_frequency);
		}
	}

	//! \brief Function called by a polling task when completes
	//!
	//! This function is called when a polling task fully
	//! completes (e.g., all child tasks have completed)
	//!
	//! \param args An opaque pointer to the polling info
	static inline void pollingTaskCompletes(void *args)
	{
		PollingInfo *info = (PollingInfo *) args;
		assert(info != nullptr);

		// The polling task has completed
		info->_finished = true;
	}
};

} // namespace tacuda

#endif // TASKING_MODEL_HPP

