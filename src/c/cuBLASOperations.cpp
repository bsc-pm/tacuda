/*
	This file is part of Task-Aware CUDA and is licensed under the terms contained in the COPYING and COPYING.LESSER files.

	Copyright (C) 2021-2023 Barcelona Supercomputing Center (BSC)
*/

#include <cuda_runtime.h>
#include <TACUDA.h>

#include "common/Environment.hpp"
#include "common/TaskingModel.hpp"
#include "common/util/ErrorHandler.hpp"

using namespace tacuda;

#pragma GCC visibility push(default)

extern "C" {

cublasStatus_t
tacublasGemmEx(cublasHandle_t handle,
            cublasOperation_t transa, cublasOperation_t transb, int m,
            int n, int k, const void *alpha, const void *matA, enum cudaDataType_t Atype,
            int lda, const void *matB, enum cudaDataType_t Btype, int ldb,
            const void *beta, void *matC, enum cudaDataType_t Ctype, int ldc, enum cudaDataType_t computeType,
            cublasGemmAlgo_t algo, cudaStream_t stream, tacudaRequest *requestPtr)
{
    cublasStatus_t stat;
    stat = cublasSetStream(handle, stream);
    if (stat != CUBLAS_STATUS_SUCCESS)
        return stat;

    stat = cublasGemmEx(handle, transa, transb,
        m, n, k, alpha, matA, Atype, lda, matB, Btype, ldb,
        beta, matC, Ctype, ldc, computeType, algo);
    if (stat != CUBLAS_STATUS_SUCCESS)
        return stat;

    Request *request = RequestManager::generateRequest(stream, (requestPtr == nullptr));
	assert(request != nullptr);

    if (requestPtr != nullptr)
		*requestPtr = request;

	return CUBLAS_STATUS_SUCCESS;
}

} // extern C

#pragma GCC visibility pop
