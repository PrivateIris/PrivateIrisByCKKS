////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2023 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "HEaaN/HEaaN.hpp"
#include "HEaaN/Integers.hpp"
#include "HEaaN/Real.hpp"
#include <vector>

namespace HEaaN {
class HEAAN_API MatrixMultiplier {
public:
    MatrixMultiplier(const int m, const int k, const int n);
    ~MatrixMultiplier();
    // Multiply two matrices A and B and store the result in C
    // A: m x k matrix
    // B: k x n matrix
    // C: m x n matrix
    void multiplyMatrix(Real *A, Real *B, Real *C, int m, int n, int k,
                        Real alpha = 1.0, Real beta = 0.0);
    void addMatrixAndMod(i64 *A, i64 *B, int m, int n, const u64 moduli);
    void subMatrixAndMultConstMod(i64 *A, i64 *B, int m, int n,
                                  const i64 mult_const, const u64 moduli);
    void multiplyMatrixAndMod(i64 *A, i64 *B, i64 *C, int m, int n, int k,
                              const u64 moduli, const u64 word_size = 20);
    // C = A x B for input matrices A, B in mod q=moduli.
    // DOES NOT WORK FOR NTT FORMATS.
    // word_size_A is determined as word_size_A = 55 - log(k) - word_size_B
    void singleRNSMultiplyMatrixAndMod(i64 *A, i64 *B, i64 *C, int m, int n,
                                       int k, const u64 moduli,
                                       const u64 word_size_B = 20);
    // Convert a row-major RLWE ciphertexts to column-major matrices.
    // The coefficients of the matrices are centered i.e. in the range
    // [-(q-1)/2, (q-1)/2] Only works for the ciphertexts with level 0.
    void convertRowMajorRLWEsToColumnMajorMatrices(
        const std::vector<Ciphertext> &ctxt, i64 *A, i64 *B, const u64 K);
    void convertColumnMajorMatricesToRowMajorRLWEs(
        const i64 *A, const i64 *B, const u64 K, const u64 N, const u64 q,
        const Context &context, std::vector<Ciphertext> &ctxt);

private:
    void initializeCUDA();

    void addMatrixAndModCUDA(i64 *A, i64 *B, int m, int n, const u64 moduli);
    void subMatrixAndMultConstModCUDA(i64 *A, i64 *B, int m, int n,
                                      const i64 mult_const, const u64 moduli);

    void multiplyMatrixCUDA(Real *A, Real *B, Real *C, int m, int n, int k,
                            Real alpha, Real beta);
    void multiplyMatrixAndModCUDA(i64 *A, i64 *B, i64 *C, int m, int n, int k,
                                  const u64 moduli, const u64 word_size);
    void singleRNSMultiplyMatrixAndModCUDA(i64 *A, i64 *B, i64 *C, int m, int n,
                                           int k, const u64 moduli,
                                           const u64 word_size_A,
                                           const u64 word_size_B);
    void convertRowMajorRLWEsToColumnMajorMatricesCUDA(
        const std::vector<Ciphertext> &ctxt, i64 *A, i64 *B, const u64 K);
    void convertColumnMajorMatricesToRowMajorRLWEsCUDA(
        const i64 *A, const i64 *B, const u64 K, const u64 N, const u64 q,
        const Context &context, std::vector<Ciphertext> &ctxt);

    i64 *a_int_;
    i64 *b_int_;
    i64 *c_int_;
    Real *a_real_;
    Real *b_real_;
    Real *c_real_;
};
} // namespace HEaaN
