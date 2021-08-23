#	This file is part of Task-Aware CUDA and is licensed under the terms contained in the COPYING and COPYING.LESSER files.
#
#	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)

ACLOCAL_AMFLAGS=-I m4

AM_CPPFLAGS=\
 $(cudainc) \
 $(BOOST_CPPFLAGS) -DBOOST_ENABLE_ASSERT_DEBUG_HANDLER \
 -I$(top_srcdir)/src \
 -I$(top_srcdir)/src/common \
 -I$(top_srcdir)/src/include \
 -include "config.h" \
 $(tacuda_CPPFLAGS)

AM_CXXFLAGS=$(tacuda_CXXFLAGS)

AM_LDFLAGS=$(cudalib) $(BOOST_LDFLAGS) -ldl
LIBS=$(cudalibs)

include_HEADERS= \
 src/include/TACUDA.h
pkginclude_HEADERS= # This library does not provide any additional header

c_api_sources= \
 src/c/cuBLASOperations.cpp \
 src/c/Initialization.cpp \
 src/c/MemoryOperations.cpp \
 src/c/Operations.cpp \
 src/c/Requests.cpp \
 src/c/Streams.cpp

common_sources = \
 src/common/Environment.cpp \
 src/common/TaskingModel.cpp

noinst_HEADERS = \
 src/common/Allocator.hpp \
 src/common/Environment.hpp \
 src/common/RequestManager.hpp \
 src/common/StreamPool.hpp \
 src/common/Symbol.hpp \
 src/common/TaskingModelAPI.hpp \
 src/common/TaskingModel.hpp \
 src/common/util/EnvironmentVariable.hpp \
 src/common/util/ErrorHandler.hpp \
 src/common/util/LockFreeQueue.hpp \
 src/common/util/SpinLock.hpp \
 src/common/util/Utils.hpp

lib_LTLIBRARIES = libtacuda.la

libtacuda_la_CPPFLAGS=$(AM_CPPFLAGS)
libtacuda_la_SOURCES=$(common_sources) $(c_api_sources)
