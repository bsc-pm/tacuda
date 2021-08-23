/*
	This file is part of Task-Aware CUDA and is licensed under the terms contained in the COPYING and COPYING.LESSER files.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKING_MODEL_API_HPP
#define TASKING_MODEL_API_HPP

#include <cstdint>


extern "C" {
	/* External Events API */

	//! \brief Get the event counter associated with the current task
	//!
	//! \returns the associated event counter with the executing task
	void *nanos6_get_current_event_counter(void);

	//! \brief Increase the counter of events of the current task to prevent the release of dependencies
	//!
	//! This function atomically increases the counter of events of a task
	//!
	//! \param[in] event_counter The event counter according with the current task
	//! \param[in] value The value to be incremented (must be positive or zero)
	void nanos6_increase_current_task_event_counter(void *event_counter, unsigned int increment);

	//! \brief Decrease the counter of events of a task and release the dependencies if required
	//!
	//! This function atomically decreases the counter of events of a task and
	//! it releases the depencencies once the number of events becomes zero
	//! and the task has completed its execution
	//!
	//! \param[in] event_counter The event counter of the task
	//! \param[in] value The value to be decremented (must be positive or zero)
	void nanos6_decrease_task_event_counter(void *event_counter, unsigned int decrement);

	//! \brief Notify that the external events API could be used
	//!
	//! This function notifies the runtime system as soon as possible
	//! that the external events API could be used from now on. This
	//! function may be defined or not depending on the runtime system
	void nanos6_notify_task_event_counter_api(void);

	//! \brief Spawn asynchronously a function
	//!
	//! \param function the function to be spawned
	//! \param args a parameter that is passed to the function
	//! \param completion_callback an optional function that will be called when the function finishes
	//! \param completion_args a parameter that is passed to the completion callback
	//! \param label an optional name for the function
	void nanos6_spawn_function(
		void (*function)(void *),
		void *args,
		void (*completion_callback)(void *),
		void *completion_args,
		char const *label);

	//! \brief Pause the current task for an amount of microseconds
	//!
	//! The task is paused for approximately the amount of microseconds
	//! passed as a parameter. The runtime may choose to execute other
	//! tasks within the execution scope of this call
	//!
	//! \param time_us the time that should be spent while paused
	//! in microseconds
	//!
	//! \returns the actual time spent during the pause
	uint64_t nanos6_wait_for(uint64_t time_us);

	//! \brief Get the total number of CPUs available to the runtime
	unsigned int nanos6_get_total_num_cpus(void);

	//! \brief Get the current virtual CPU identifier
	unsigned int nanos6_get_current_virtual_cpu(void);

	//! Prototypes of the tasking model API functions
	typedef void *get_current_event_counter_t(void);
	typedef void increase_current_task_event_counter_t(void *, unsigned int);
	typedef void decrease_task_event_counter_t(void *, unsigned int);
	typedef void notify_task_event_counter_api_t(void);
	typedef void spawn_function_t(void (*)(void *), void *, void (*)(void *), void *, char const *);
	typedef uint64_t wait_for_t(uint64_t);
	typedef unsigned int get_total_num_cpus_t(void);
	typedef unsigned int get_current_virtual_cpu_t(void);
}

#endif // TASKING_MODEL_API_HPP
