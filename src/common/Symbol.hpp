/*
	This file is part of Task-Aware CUDA and is licensed under the terms contained in the COPYING and COPYING.LESSER files.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef SYMBOL_HPP
#define SYMBOL_HPP

#include <dlfcn.h>
#include <string>

#include "util/ErrorHandler.hpp"


namespace tacuda {

//! Class that allows the dynamic loading of symbols at run-time
class Symbol {
public:
	//! \brief Load a symbol from the subsequent libraries
	//!
	//! \param symbolName The name of the symbol to load
	//! \param mandatory Whether should abort the program if not found
	//!
	//! \returns An opaque pointer to the symbol or null if not found
	static inline void *load(const std::string &symbolName, bool mandatory = true)
	{
		void *symbol = dlsym(RTLD_NEXT, symbolName.c_str());
		if (symbol == nullptr && mandatory) {
			ErrorHandler::fail("Could not find symbol ", symbolName);
		}
		return symbol;
	}
};

} // namespace tacuda

#endif // SYMBOL_HPP
