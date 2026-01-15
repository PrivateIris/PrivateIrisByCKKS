////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "HEaaN/HEaaN.hpp"
#include "HEaaN/device/Device.hpp"
#include <vector>

namespace HEaaN {
class MatrixMultiplier {
public:
    MatrixMultiplier(const int m, const int k, const int n, const u64 degree,
                     const u64 modulus, const u64 word_size = 20,
                     DeviceType device_type = DeviceType::GPU);
    ~MatrixMultiplier();
    MatrixMultiplier(const MatrixMultiplier &) = delete;
    MatrixMultiplier &operator=(const MatrixMultiplier &) = delete;
    MatrixMultiplier(MatrixMultiplier &&) = delete;
    MatrixMultiplier &operator=(MatrixMultiplier &&) = delete;
    // Multiply two matrices A and B and store the result in C
    // A: m x k matrix
    // B: k x n matrix
    // C: m x n matrix
    void multiplyMatrix(Real *A, Real *B, Real *C, int m, int n, int k,
                        Real alpha = 1.0, Real beta = 0.0);
    void multiplyMatrixAndMod(i64 *A, i64 *B, i64 *C, int m, int n, int k);
    // C = A x B for input matrices A, B in mod q=moduli.
    // DOES NOT WORK FOR NTT FORMATS.
    // word_size_A is determined as word_size_A = 55 - log(k) - word_size_B
    void singleRNSMultiplyMatrixAndMod(i64 *A, i64 *B, i64 *C, int m, int n,
                                       int k);
    // Convert a row-major RLWE ciphertexts to column-major matrices.
    // The coefficients of the matrices are centered i.e. in the range
    // [-(q-1)/2, (q-1)/2] Only works for the ciphertexts with level 0.
    void convertRowMajorRLWEsToColumnMajorMatrices(
        const std::vector<Ciphertext> &ctxt, i64 *A, i64 *B, const u64 K);
    void convertColumnMajorMatricesToRowMajorRLWEs(
        const i64 *A, const i64 *B, const u64 K, const u64 N, const u64 q,
        std::vector<Ciphertext> &ctxt);

    void multiplyMatrixAndModNative(i64 *A, i64 *B, i64 *C, int m, int n,
                                    int k);

private:
    void initializeCUDA();

    void multiplyMatrixCUDA(Real *A, Real *B, Real *C, int m, int n, int k,
                            Real alpha, Real beta, size_t B_idx = 0);
    void multiplyMatrixAndModCUDA(i64 *A, i64 *B, i64 *C, int m, int n, int k);
    void singleRNSMultiplyMatrixAndModCUDA(i64 *A, i64 *B, i64 *C, int m, int n,
                                           int k, const u64 word_size_A,
                                           const u64 word_size_B);
    void convertRowMajorRLWEsToColumnMajorMatricesCUDA(
        const std::vector<Ciphertext> &ctxt, i64 *A, i64 *B, const u64 K);
    void convertColumnMajorMatricesToRowMajorRLWEsCUDA(
        const i64 *A, const i64 *B, const u64 K, const u64 N, const u64 q,
        std::vector<Ciphertext> &ctxt);
    void multiplyMatrixAndModNativeCUDA(i64 *A, i64 *B, i64 *C, int m, int n,
                                        int k);

    void multiplyMatrixCPU(Real *A, Real *B, Real *C, int m, int n, int k,
                           Real alpha, Real beta, size_t B_idx = 0);
    void multiplyMatrixAndModCPU(i64 *A, i64 *B, i64 *C, int m, int n, int k);
    void singleRNSMultiplyMatrixAndModCPU(i64 *A, i64 *B, i64 *C, int m, int n,
                                          int k, const u64 word_size_A,
                                          const u64 word_size_B);
    void convertRowMajorRLWEsToColumnMajorMatricesCPU(
        const std::vector<Ciphertext> &ctxt, i64 *A, i64 *B, const u64 K);
    void convertColumnMajorMatricesToRowMajorRLWEsCPU(
        const i64 *A, const i64 *B, const u64 K, const u64 N, const u64 q,
        std::vector<Ciphertext> &ctxt);
    void multiplyMatrixAndModNativeCPU(const i64 *A, const i64 *B, i64 *C,
                                       int m, int n, int k) const;

    u64 degree_;
    u64 modulus_;
    u64 two_modulus_;
    u64 barr_for_64_;
    u64 two_to_64_;
    u64 two_to_64_shoup_;
    u64 word_size_;
    i64 *a_int_;
    i64 *b_int_;
    i64 *c_int_;
    Real *a_real_;
    Real *b_real_;
    Real *c_real_;

    DeviceType device_type_;
};
} // namespace HEaaN
