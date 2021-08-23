#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

#	This file is part of Task-Aware CUDA and is licensed under the terms contained in the COPYING and COPYING.LESSER files.
#
#	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)

AC_PREREQ([2.69])
AC_INIT([tacuda], [1.0.0], [pm-tools@bsc.es])

AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_HEADERS([config.h])

AC_LANG([C++])

# Prepare compilation flags before AC_PROG_CXX
AX_COMPILE_FLAGS

AC_PROG_CXX

# Retrieve compiler version
AX_CHECK_CXX_VERSION

# AM_PROG_AR must be called before LT_INIT or a warning ensues
AM_PROG_AR

# Automake initialization
AM_INIT_AUTOMAKE([foreign -Wall dist-bzip2 -Wno-portability subdir-objects silent-rules])
AM_SILENT_RULES([yes])

# Checks for header files.
AC_CHECK_HEADERS([unistd.h])

# Check for CUDA
AX_CHECK_CUDA

# Boost is no longer required
AX_BOOST_BASE([1.59], [], [ AC_MSG_ERROR([Boost library not found])])

# Checks for typedefs, structures, and compiler characteristics.
AX_CXX_COMPILE_STDCXX_11([noext],[mandatory])
AC_CHECK_HEADER_STDBOOL
AC_TYPE_SIZE_T

AX_CHECK_CPUS

# Turn off shared libraries.
LT_INIT([static])

# Enable/disable architecture specific code
AS_CASE([$host],
  [x86_64*],[ARCH=x86_64])
AS_IF([test "$ARCH" = "x86_64"],[
  AC_DEFINE([X86_64_ARCH],[1],[Compiling for x86_64 host])
])

AX_VAR_PUSHVALUE([prefix],[/gpfs/scratch/bsc15/bsc15963/tacuda_proj])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

echo ""
echo ""
echo "Configuration summary:"
echo "    Compiler version... ${CXX_VERSION}"
echo ""
echo "    Installation prefix... ${prefix}"
echo ""
echo "    CUDA CPPFLAGS... ${cudainc}"
echo "    CUDA LDFLAGS... ${cudalibs} ${cudalib}"
echo ""
echo "    CXXFLAGS... ${tacuda_CXXFLAGS} ${CXXFLAGS}"
echo "    CPPFLAGS... ${tacuda_CPPFLAGS} ${CPPFLAGS}"
echo ""
echo "    Boost CPPFLAGS... ${BOOST_CPPFLAGS}"
echo "    Boost LDFLAGS... ${BOOST_LDFLAGS}"
