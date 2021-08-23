/*
	This file is part of Task-Aware CUDA and is licensed under the terms contained in the COPYING and COPYING.LESSER files.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef TACUDA_H
#define TACUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <library_types.h>

#include <stddef.h>
#include <stdint.h>

#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif

typedef void *tacudaRequest;

static const tacudaRequest TACUDA_REQUEST_NULL = NULL;
static const size_t TACUDA_STREAMS_AUTO = 0;

//! \brief Initialization
//!
//! Initializes the TACUDA environment and the CUDA driver API
CUresult
tacudaInit(unsigned int flags);

//! \brief Finalization
//!
//! Finalizes the TACUDA environment
CUresult
tacudaFinalize();

//! \brief Initialization of the pool of streams
//!
//! Initlializes the pool of stream with count asynchronous stream
cudaError_t
tacudaCreateStreams(size_t count);

//! \brief Finalization of the pool of streams
cudaError_t
tacudaDestroyStreams();

//! \brief Getting a stream
//!
//!  Gets a stream from the pool by passing a pointer to a previously declarated stream
cudaError_t
tacudaGetStream(cudaStream_t *stream);

//! \brief Returning a stream to the pool
cudaError_t
tacudaReturnStream(cudaStream_t stream);

//! \brief Binding the calling task to a stream
//!
//! Asynchronous function, binds the completion of the calling task
//! to the finalization of the submitted operations on the stream
cudaError_t
tacudaSynchronizeStreamAsync(cudaStream_t stream);


//! The following four functions are wrapper functions for some CUDA asynchronous operations.
//! Apart form their standard behaviour, they also return a TACUDA request.
//! 	- If the request pointer parameter is not NULL, TACUDA generates a tacudaRequest and saves a
//! 	pointer to it in the output parameter after executing the corresponding operation
//! 	- If the parameter is NULL, TACUDA directly binds the desired operation to the calling task without generating any request

//! \brief Copying of data between host and device
//!
//! Asynchronous wrapper function, copies data between host and device
__host__ __device__ cudaError_t
tacudaMemcpyAsync(
	void *dst, const void *src, size_t sizeBytes,
	enum cudaMemcpyKind kind, cudaStream_t stream,
	tacudaRequest *request);

//! \brief Initialization of device memory
//! 
//! Asynchronous wrapper function, initializes or sets device memory to a value
__host__ __device__ cudaError_t
tacudaMemsetAsync(
	void *devPtr, int value, size_t sizeBytes,
	cudaStream_t stream,
	tacudaRequest *request);

//! \brief Launching a device function 
//!
//! Asynchronous wrapper function (from the CPU view), launches a kernel or device function "func"
__host__ cudaError_t
tacudaLaunchKernel(
    const void* func, dim3 gridDim, dim3 blockDim, 
    void** args, size_t sharedMem, cudaStream_t stream,
    tacudaRequest *request);

//! \brief Performs a matrix-matrix multiplication
//! 
//! Asynchronous wrapper function (from the CPU view) from the cuBLAS library,
//! performs the following matrix-matrix multiplication: C := alpha*op(A)*op(B) + beta*C
__device__ cublasStatus_t
tacublasGemmEx(cublasHandle_t handle,
            cublasOperation_t transa, cublasOperation_t transb, int m, 
            int n, int k, const void *alpha, const void *matA, enum cudaDataType_t Atype,
            int lda, const void *matB, enum cudaDataType_t Btype, int ldb,
            const void *beta, void *matC, enum cudaDataType_t Ctype, int ldc, enum cudaDataType_t computeType,
            cublasGemmAlgo_t algo, cudaStream_t stream, tacudaRequest *requestPtr);

//! \brief Binding a request 
//!
//! Asynchronous and non-blocking operation, binds a request to the calling task
cudaError_t
tacudaWaitRequestAsync(tacudaRequest *request);

//! \brief Bindig multiple requests
//!
//! Asynchronous and non-blocking operation, binds the calling task 
//! to count requests, all of them stored in "requests"
cudaError_t
tacudaWaitallRequestsAsync(size_t count, tacudaRequest requests[]);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif /* TACUDA_H */
