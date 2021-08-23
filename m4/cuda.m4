AC_DEFUN([AX_CHECK_CUDA],[

#Check if an CUDA implementation is installed.
AC_ARG_WITH(cuda,
[AS_HELP_STRING([--with-cuda,--with-cuda=PATH],
                [search in system directories or specify prefix directory for installed CUDA package])])
AC_ARG_WITH(cuda-include,
[AS_HELP_STRING([--with-cuda-include=PATH],
                [specify directory for installed CUDA include files])])
AC_ARG_WITH(cuda-lib,
[AS_HELP_STRING([--with-cuda-lib=PATH],
                [specify directory for the installed CUDA library])])

# Search for CUDA by default
AS_IF([test "$with_cuda" != yes],[
  cudainc="-I$with_cuda/include"
  AS_IF([test -d $with_cuda/lib64/stubs],
    [olibdir=$with_cuda/lib64/stubs],
    [olibdir=$with_cuda/lib/stub])

  cudalib="-L$olibdir -Wl,-rpath,$olibdir"
])

AS_IF([test "x$with_cuda_include" != x],[
  cudainc="-I$with_cuda_include"
])
AS_IF([test "x$with_cuda_lib" != x],[
  cudalib="-L$with_cuda_lib -Wl,-rpath,$with_cuda_lib"
])

# Tests if provided headers and libraries are usable and correct
AX_VAR_PUSHVALUE([CPPFLAGS],[$CPPFLAGS $cudainc])
AX_VAR_PUSHVALUE([CFLAGS])
AX_VAR_PUSHVALUE([LDFLAGS],[$LDFLAGS $cudalib])
AX_VAR_PUSHVALUE([LIBS],[])


AC_CHECK_HEADERS([cuda.h cuda_runtime.h cublas_v2.h library_types.h], [cuda=yes], [cuda=no])
search_libs="cuda cublas cudart"
required_libs=""

m4_foreach([function],
           [cuInit,
	    cublasSgemm,
            cudaStreamCreate,
            cudaLaunchKernel,
	    cudaMemcpyAsync],
           [
             AS_IF([test "$cuda" = "yes"],[
               AC_SEARCH_LIBS(function,
                              [$search_libs],
                              [cuda=yes],
                              [cuda=no],
                              [$required_libs])dnl
             ])
           ])dnl

cudalibs=$LIBS

AX_VAR_POPVALUE([CPPFLAGS])
AX_VAR_POPVALUE([CFLAGS])
AX_VAR_POPVALUE([LDFLAGS])
AX_VAR_POPVALUE([LIBS])

AS_IF([test "$cuda" != "yes"],[
    AC_MSG_ERROR([
------------------------------
CUDA path was not correctly specified.
Please, check that the provided directories are correct.
------------------------------])
])

AC_SUBST([cuda])
AC_SUBST([cudainc])
AC_SUBST([cudalib])
AC_SUBST([cudalibs])

])dnl AX_CHECK_CUDA

