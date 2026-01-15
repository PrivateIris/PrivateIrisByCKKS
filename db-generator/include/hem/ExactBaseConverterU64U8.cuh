////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/HEaaN.hpp"
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

using namespace HEaaN;

template <typename T, typename U>
__global__ void signedMulModKernel(const T *op, const u64 mult_const, U *res,
                                   const u64 moduli, size_t size);

template <typename T>
__global__ void signedRNSToCRTKernel(T **op, const __uint128_t *mod_hat,
                                     const __uint128_t prod_mod,
                                     __int128_t *res, size_t num_arr,
                                     size_t size_arr);

template <typename T>
__global__ void signedCRTToRNSKernel(const __int128_t *op, const u64 *mod,
                                     T **res, size_t num_arr, size_t size_arr);

__global__ void embedI64toI8Mod(i64 *op, std::int8_t *res, const u64 modulus,
                                const size_t size);
__global__ void embedI8toI64Mod(std::int8_t *op, i64 *res, const u64 modulus,
                                const size_t size);
