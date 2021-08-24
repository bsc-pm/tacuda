# Task-Aware CUDA Library

## Documentation and programming model
The Task-Aware CUDA (TACUDA) library extends the functionality of the CUDA language by providing new mechanisms for improving the interoperability between parallel task-based programming models, such as OpenMP and OmpSs-2, and the offloading of kernels to NVIDIA GPUs. The library allows the efficient offloading of CUDA operations from concurrent tasks in an asynchronous manner. TACUDA manages most of the low-level aspects, so the developers can focus in exploiting the parallelism os the application.

The TACUDA library contains both exclusive TACUDA functions and wrapper functions of the CUDA language and its APIs. All the wrapper functions refer to the most relevant CUDA asynchronous operations, so if more information about the original functions is needed it can be found in the [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda).

The idea behind the library is to let the tasks inside a taskified application to offload multiple CUDA operations into a given set of CUDA streams that run on the NVIDIA GPU, and then these tasks should bind their completion to the finalization of their operations in an asynchronous way. All of this happens without blocking the host CPUs while the CUDA operations are running on the GPU, so this allows host CPUs to execute other ready tasks meanwhile. In general, tasks will use standard CUDA functions to offload operations and use TACUDA functions to bind and "synchronize" with their compeltion.

The library is initialized by calling tacudaInit, which is a wrapper function of the cuInit function of the Driver API. The TACUDA library is finalized by calling tacudaFinalize. In the TACUDA applications is important to use multiple stream, as multiple tasks will submit operations on CUDA streams concurrently and we want to distribute the operations and reduce unnecessary dependencies (which are implicitly created). For this reason, the TACUDA library generates a pool of streams that can be used by the tasks during the execution of the application. This pool can be created with the function ```tacudaCreateStreams(size_t count)```, and must be destroyed before the finalization of the library with ```tacudaDestroyStreams```. 
When a task wants to offload an operation to the GPU, it can retrieve on of these streams for exclusive use until it is returned. When the stream is not needed anymore, the task can return it to the pool. The functions corresponding to these actions are the following:

```
tacudaGetStream(cudaStream_t *stream);
tacudaReturnStream(cudaStream_t stream);
```

When operations have been offloaded from a task, it may be interesting for the task to synchronize with these operations. That's when the function ```tacudaSynchronizeStreamAsync(cudaStream_t stream)``` comes to use. Given the asynchronous nature of CUDA operations, this function allows the task to synchronize with the operations submitted to a specific stream, but in an asynchronous manner. It binds the completion of the task to the finalization of the submitted operations on the stream, and so the calling task will be able to continue and finish its execution but will delay its fully completion (and the release of data dependencies) until all the operations in the stream finalize.

TACUDA also defines another way to asynchronously wait for CUDA operations that allows a more fine-grained waiting. As it was mentioned earlier, TACUDA defines a wrapper function for each of the most relevant CUDA asynchronous operations. The wrappers have the same behaviour and parameters as the originals, plus an additional parameter that returns a TACUDA request (tacudaRequest). If the request pointer parameter is not NULL, TACUDA generates a tacudaRequest and saves a pointer to it in the output parameter after executing the corresponding operation. Then, the task can bind itself to the request calling the functions ```tacudaWaitRequestAsync(tacudaRequest *request)``` or ```tacudaWaitallRequestsAsync(size_t count, tacudaRequest requests[])```. If the request pointer is NULL, TACUDA directly binds the desired operation to the calling task without generating any request. As an example, the following codes do exactly the same:

**1:**

```
tacudaRequest request;
tacudaMemcpyAsync(..., stream, &request);
tacudaWaitRequestAsync(&request);
```

**2:**

```
tacudaMemcpyAsync(..., stream, NULL);
```

Finally, inside the header file src/include/TACUDA.h, it can be found a brief explanation of every function in the library and their functionality (the remaining functions of the library are all wrapper functions), which can be used as a "cheat sheet" and it is very helpful while using the TACUDA library. 

## Building and installing
TACUDA uses the standard GNU automake and libtool toolchain. When cloning from a repository, the building environment must be prepared executing the bootstrap script:

```
$ ./bootstrap
```

Then, to configure, compile and install the library execute the following commands:

```
$ ./configure --prefix=$INSTALL_PREFIX --with-cuda=$CUDA_HOME ..other options..
$ make
$ make install
```

where $INSTALL_PREFIX is the directory into which to install TACUDA, and $CUDA_HOME is the prefix of the CUDA installation. There are two other optional configuration flags, which probably will not be needed if the user has a normal CUDA setup:

- `--with-cuda-include` : It lets the user specify the directory where the CUDA include files are installed.
- `--with-cuda-lib` : It lets the user specify the directory where the CUDA libraries are installed.

Once TACUDA is built and installed, e.g, in $TACUDA_HOME, the installation folder will contain the library in $TACUDA_HOME/lib (or $TACUDA_HOME/lib64) and the header in $TACUDA_HOME/include.

## Requirements
In order to install the TACUDA library, the main requirements are the following:

- Automake, autoconf, libtool, make and a C and C++ compiler.
- CUDA, including the Runtime, Driver and cuBLAS APIs (with a version compatible with the compiler version).
- Boost library 1.59 or greater
- OmpSs-2 (version 2018.11 or greater)
- ...

The cuBLAS API is not essential to the library, so if the user does the proper changes to the Makefile.am file (basically removing or commenting all the lines containing `src/c/cuBLASOperations.cpp` and a few compiler flags) it can be compiled and installed without it.
