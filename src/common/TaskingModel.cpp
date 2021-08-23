/*
	This file is part of Task-Aware CUDA and is licensed under the terms contained in the COPYING and COPYING.LESSER files.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#include <string>

#include "Symbol.hpp"
#include "TaskingModel.hpp"
#include "util/ErrorHandler.hpp"


namespace tacuda {

void TaskingModel::initialize()
{
	// Load the functions required by polling tasks
	_waitFor = (wait_for_t *) Symbol::load(_waitForName);
	_spawnFunction = (spawn_function_t *) Symbol::load(_spawnFunctionName);

	// Load the task events API functions
	_getCurrentEventCounter = (get_current_event_counter_t *)
		Symbol::load(_getCurrentEventCounterName);
	_increaseCurrentTaskEventCounter = (increase_current_task_event_counter_t *)
		Symbol::load(_increaseCurrentTaskEventCounterName);
	_decreaseTaskEventCounter = (decrease_task_event_counter_t *)
		Symbol::load(_decreaseTaskEventCounterName);

	_notifyTaskEventCounterAPI = (notify_task_event_counter_api_t *)
		Symbol::load(_notifyTaskEventCounterAPIName, false);

	_getTotalNumCPUs = (get_total_num_cpus_t *)
		Symbol::load(_getTotalNumCPUsName);
	_getCurrentVirtualCPU = (get_current_virtual_cpu_t *)
		Symbol::load(_getCurrentVirtualCPUName);

	// Notify the tasking runtime that the event counters
	// API may be used during the execution. This function
	// is not required so call it only if defined
	if (_notifyTaskEventCounterAPI) {
		(*_notifyTaskEventCounterAPI)();
	}
}

//! The pointers to the tasking model API functions
get_current_event_counter_t *TaskingModel::_getCurrentEventCounter = nullptr;
increase_current_task_event_counter_t *TaskingModel::_increaseCurrentTaskEventCounter = nullptr;
decrease_task_event_counter_t *TaskingModel::_decreaseTaskEventCounter = nullptr;
notify_task_event_counter_api_t *TaskingModel::_notifyTaskEventCounterAPI = nullptr;
spawn_function_t *TaskingModel::_spawnFunction = nullptr;
wait_for_t *TaskingModel::_waitFor = nullptr;
get_total_num_cpus_t *TaskingModel::_getTotalNumCPUs = nullptr;
get_current_virtual_cpu_t *TaskingModel::_getCurrentVirtualCPU = nullptr;

//! Actual names of the tasking model API functions
const std::string TaskingModel::_getCurrentEventCounterName("nanos6_get_current_event_counter");
const std::string TaskingModel::_increaseCurrentTaskEventCounterName("nanos6_increase_current_task_event_counter");
const std::string TaskingModel::_decreaseTaskEventCounterName("nanos6_decrease_task_event_counter");
const std::string TaskingModel::_notifyTaskEventCounterAPIName("nanos6_notify_task_event_counter_api");
const std::string TaskingModel::_spawnFunctionName("nanos6_spawn_function");
const std::string TaskingModel::_waitForName("nanos6_wait_for");
const std::string TaskingModel::_getTotalNumCPUsName("nanos6_get_total_num_cpus");
const std::string TaskingModel::_getCurrentVirtualCPUName("nanos6_get_current_virtual_cpu");

} // namespace tacuda
